from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common_benchmark import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    NER_LABELS,
    RESULTS,
    bootstrap_ner,
    bootstrap_re,
    choose_threshold,
    load_task,
    ner_metrics,
    re_metrics,
    save_gzip_json,
    save_json,
    seed_everything,
)

BASE_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
BIOREX_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/pretrained_model.zip"
SPECIAL_MARKERS = ["<C1>", "</C1>", "<C2>", "</C2>"]


def freeze_lower_layers(model, number_to_freeze=6):
    base = None
    for attribute in ["bert", "roberta", "deberta", "electra"]:
        if hasattr(model, attribute):
            base = getattr(model, attribute)
            break
    if base is None:
        return {"frozen_layers": 0, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if hasattr(base, "embeddings"):
        for parameter in base.embeddings.parameters():
            parameter.requires_grad = False
    layers = None
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        layers = base.encoder.layer
    if layers is not None:
        for layer in layers[:number_to_freeze]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    return {
        "frozen_layers": number_to_freeze if layers is not None else 0,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


class TransformerNerDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=256):
        self.rows = rows
        self.items = []
        self.truncated_rows = 0
        for index, row in enumerate(rows):
            encoded = tokenizer(
                row["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            word_ids = encoded.word_ids()
            aligned = []
            seen = set()
            for word_id in word_ids:
                if word_id is None or word_id in seen:
                    aligned.append(-100)
                else:
                    aligned.append(LABEL_TO_ID[row["labels"][word_id]])
                    seen.add(word_id)
            if len(seen) < len(row["tokens"]):
                self.truncated_rows += 1
            item = {key: value for key, value in encoded.items()}
            item["labels"] = aligned
            item["word_ids"] = [-1 if value is None else value for value in word_ids]
            item["row_index"] = index
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class NerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        model_items = [{key: value for key, value in item.items() if key not in {"labels", "word_ids", "row_index"}} for item in items]
        batch = self.tokenizer.pad(model_items, padding=True, return_tensors="pt")
        maximum = batch["input_ids"].shape[1]
        labels = torch.full((len(items), maximum), -100, dtype=torch.long)
        word_ids = torch.full((len(items), maximum), -1, dtype=torch.long)
        indices = []
        for row_index, item in enumerate(items):
            length = len(item["labels"])
            labels[row_index, :length] = torch.tensor(item["labels"], dtype=torch.long)
            word_ids[row_index, :length] = torch.tensor(item["word_ids"], dtype=torch.long)
            indices.append(item["row_index"])
        batch["labels"] = labels
        batch["word_ids"] = word_ids
        batch["row_indices"] = indices
        return batch


def predict_ner(model, loader, rows, device):
    predictions = [["O"] * len(row["tokens"]) for row in rows]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            word_ids = batch.pop("word_ids")
            row_indices = batch.pop("row_indices")
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.argmax(dim=-1).cpu()
            for item_index, source_index in enumerate(row_indices):
                previous = -1
                for position, word_id in enumerate(word_ids[item_index].tolist()):
                    if word_id < 0 or word_id == previous:
                        continue
                    if word_id < len(predictions[source_index]):
                        predictions[source_index][word_id] = ID_TO_LABEL[int(logits[item_index, position])]
                    previous = word_id
    return predictions


def ner_loss(logits, labels, class_weights):
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        weight=class_weights,
        ignore_index=-100,
    )


def train_ner() -> None:
    seed_everything(42)
    rows = load_task("ner")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    datasets = {split: TransformerNerDataset(rows[split], tokenizer) for split in rows}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=8 if split == "train" else 24,
            shuffle=split == "train",
            collate_fn=NerCollator(tokenizer),
            num_workers=2,
        )
        for split, dataset in datasets.items()
    }
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(NER_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    freeze_info = freeze_lower_layers(model, 6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    counts = Counter(label for row in rows["train"] for label in row["labels"])
    total = sum(counts.values())
    weights = np.asarray([np.sqrt(total / max(1, counts[label])) for label in NER_LABELS], dtype=np.float32)
    weights = np.clip(weights / weights.mean(), 0.25, 3.5)
    class_weights = torch.tensor(weights, device=device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-5, weight_decay=0.01)
    accumulation = 2
    number_of_epochs = 4
    steps_per_epoch = int(np.ceil(len(loaders["train"]) / accumulation))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * steps_per_epoch * number_of_epochs)),
        num_training_steps=steps_per_epoch * number_of_epochs,
    )
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience = 0
    history = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, number_of_epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(loaders["train"], 1):
            batch.pop("word_ids")
            batch.pop("row_indices")
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = ner_loss(logits, labels, class_weights) / accumulation
            loss.backward()
            losses.append(float(loss.detach().cpu()) * accumulation)
            if step % accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        dev_predictions = predict_ner(model, loaders["dev"], rows["dev"], device)
        dev_metrics = ner_metrics(rows["dev"], dev_predictions)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **dev_metrics})
        print("PubMedBERT NER epoch", epoch, history[-1], flush=True)
        if dev_metrics["exact_f1"] > best_f1 + 1e-4:
            best_f1 = dev_metrics["exact_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Transformer NER training failed")
    model.load_state_dict(best_state)
    model.to(device)
    torch.save(
        {
            "model_state": best_state,
            "model_name": BASE_MODEL,
            "labels": NER_LABELS,
            "best_epoch": best_epoch,
        },
        RESULTS / "pubmedbert_ner_model.pt",
    )
    result_rows = []
    prediction_payload = {}
    for split in ["dev", "id_test", "temporal_ood_test"]:
        predictions = predict_ner(model, loaders[split], rows[split], device)
        metrics = ner_metrics(rows[split], predictions)
        confidence_interval = bootstrap_ner(rows[split], predictions, iterations=500) if split != "dev" else (float("nan"), float("nan"))
        result_rows.append(
            {
                "model": "PubMedBERT token classifier",
                "family": "domain Transformer / SOTA-class",
                "split": split,
                **metrics,
                "ci95_low": confidence_interval[0],
                "ci95_high": confidence_interval[1],
                "train_seconds": train_seconds,
                "best_epoch": best_epoch,
                "best_dev_f1": best_f1,
                "truncated_rows": datasets[split].truncated_rows,
                "hyperparameters": json.dumps({"base_model": BASE_MODEL, "frozen_lower_layers": 6, "lr": 3e-5, "effective_batch": 16}),
            }
        )
        prediction_payload[split] = predictions
    pd.DataFrame(result_rows).to_csv(RESULTS / "transformer_ner_results.csv", index=False)
    save_gzip_json(RESULTS / "transformer_ner_predictions.json.gz", {"PubMedBERT token classifier": prediction_payload})
    save_json(RESULTS / "transformer_ner_history.json", history)
    save_json(
        RESULTS / "transformer_ner_complete.json",
        {
            "status": "success",
            "device": str(device),
            "freeze_info": freeze_info,
            "class_weights": {label: float(weight) for label, weight in zip(NER_LABELS, weights)},
            "truncated_rows": {split: dataset.truncated_rows for split, dataset in datasets.items()},
        },
    )


class TransformerRelationDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=256):
        self.rows = rows
        self.encodings = tokenizer(
            [row["marked_sentence"] for row in rows],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.rows[index]["binary_label"])
        item["row_index"] = index
        return item


class RelationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        indices = [item["row_index"] for item in items]
        labels = torch.tensor([item["labels"] for item in items], dtype=torch.long)
        model_items = [{key: value for key, value in item.items() if key not in {"labels", "row_index"}} for item in items]
        batch = self.tokenizer.pad(model_items, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["row_indices"] = indices
        return batch


def locate_biorex_model() -> Path | None:
    cache = Path.home() / ".cache" / "enzchemred_biorex"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "pretrained_model.zip"
    extracted = cache / "model"
    if not archive.exists():
        try:
            urllib.request.urlretrieve(BIOREX_URL, archive)
        except Exception as error:
            print("BioREx download failed; falling back to PubMedBERT:", error, flush=True)
            return None
    if not extracted.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extracted)
        except Exception as error:
            print("BioREx extraction failed; falling back to PubMedBERT:", error, flush=True)
            return None
    candidates = []
    for config in extracted.rglob("config.json"):
        parent = config.parent
        if any((parent / name).exists() for name in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]):
            candidates.append(parent)
    return sorted(candidates, key=lambda path: len(str(path)))[0] if candidates else None


def predict_relation(model, loader, number_of_rows, device):
    probabilities = np.zeros(number_of_rows, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            indices = batch.pop("row_indices")
            batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            probability = torch.softmax(model(**inputs).logits, dim=-1)[:, 1].cpu().numpy()
            for source_index, value in zip(indices, probability):
                probabilities[source_index] = value
    return probabilities


def train_relation() -> None:
    seed_everything(42)
    rows = load_task("re")
    biorex_directory = locate_biorex_model()
    tokenizer_source = biorex_directory if biorex_directory is not None else BASE_MODEL
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_MARKERS})
    datasets = {split: TransformerRelationDataset(rows[split], tokenizer) for split in rows}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=12 if split == "train" else 32,
            shuffle=split == "train",
            collate_fn=RelationCollator(tokenizer),
            num_workers=2,
        )
        for split, dataset in datasets.items()
    }
    model_source = biorex_directory if biorex_directory is not None else BASE_MODEL
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        backbone = "official BioREx pretrained model" if biorex_directory is not None else "PubMedBERT fallback"
    except Exception as error:
        print("BioREx model load failed; using PubMedBERT:", error, flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True)
        backbone = "PubMedBERT fallback"
    model.resize_token_embeddings(len(tokenizer))
    freeze_info = freeze_lower_layers(model, 6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    counts = Counter(row["binary_label"] for row in rows["train"])
    weights = torch.tensor([1.0, math.sqrt(counts[0] / counts[1])], dtype=torch.float, device=device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2.5e-5, weight_decay=0.01)
    accumulation = 2
    number_of_epochs = 4
    steps_per_epoch = int(np.ceil(len(loaders["train"]) / accumulation))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * steps_per_epoch * number_of_epochs)),
        num_training_steps=steps_per_epoch * number_of_epochs,
    )
    best_f1 = -1.0
    best_threshold = 0.5
    best_state = None
    best_epoch = 0
    patience = 0
    history = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, number_of_epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(loaders["train"], 1):
            batch.pop("row_indices")
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = nn.functional.cross_entropy(logits, labels, weight=weights) / accumulation
            loss.backward()
            losses.append(float(loss.detach().cpu()) * accumulation)
            if step % accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        dev_scores = predict_relation(model, loaders["dev"], len(rows["dev"]), device)
        dev_truth = np.asarray([row["binary_label"] for row in rows["dev"]])
        threshold, dev_f1 = choose_threshold(dev_truth, dev_scores)
        dev_metrics = re_metrics(dev_truth, dev_scores >= threshold, dev_scores)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "threshold": threshold, **dev_metrics})
        print("BioREx/PubMedBERT RE epoch", epoch, history[-1], flush=True)
        if dev_f1 > best_f1 + 1e-4:
            best_f1 = dev_f1
            best_threshold = threshold
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Transformer relation training failed")
    model.load_state_dict(best_state)
    model.to(device)
    torch.save(
        {
            "model_state": best_state,
            "model_source": str(model_source),
            "backbone": backbone,
            "best_epoch": best_epoch,
            "threshold": best_threshold,
        },
        RESULTS / "biorex_relation_model.pt",
    )
    result_rows = []
    prediction_payload = {}
    model_name = "BioREx/PubMedBERT entity-marker classifier"
    for split in ["dev", "id_test", "temporal_ood_test"]:
        scores = predict_relation(model, loaders[split], len(rows[split]), device)
        truth = np.asarray([row["binary_label"] for row in rows[split]])
        predictions = (scores >= best_threshold).astype(int)
        metrics = re_metrics(truth, predictions, scores)
        confidence_interval = bootstrap_re(rows[split], truth, predictions, iterations=1000) if split != "dev" else (float("nan"), float("nan"))
        result_rows.append(
            {
                "model": model_name,
                "family": "BioREx domain Transformer / SOTA-class",
                "split": split,
                **metrics,
                "ci95_low": confidence_interval[0],
                "ci95_high": confidence_interval[1],
                "threshold": best_threshold,
                "train_seconds": train_seconds,
                "best_epoch": best_epoch,
                "best_dev_f1": best_f1,
                "backbone": backbone,
                "hyperparameters": json.dumps({"frozen_lower_layers": 6, "lr": 2.5e-5, "effective_batch": 24}),
            }
        )
        prediction_payload[split] = {"predictions": predictions.tolist(), "scores": scores.tolist()}
    pd.DataFrame(result_rows).to_csv(RESULTS / "transformer_re_results.csv", index=False)
    save_gzip_json(RESULTS / "transformer_re_predictions.json.gz", {model_name: prediction_payload})
    save_json(RESULTS / "transformer_re_history.json", history)
    save_json(
        RESULTS / "transformer_re_complete.json",
        {"status": "success", "device": str(device), "backbone": backbone, "freeze_info": freeze_info},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ner", "re"], required=True)
    arguments = parser.parse_args()
    if arguments.task == "ner":
        train_ner()
    else:
        train_relation()


if __name__ == "__main__":
    main()
