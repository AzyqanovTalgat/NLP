from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from common import (
    BIOMED_MODEL,
    NER_LABELS,
    RESULTS,
    bootstrap_ner_f1,
    flatten_metric_row,
    load_ner_bundle,
    ner_metrics,
    prepare_data,
    repair_bio,
    seed_everything,
    write_json,
)

OUT = RESULTS / "stage_ner_sota"
OUT.mkdir(parents=True, exist_ok=True)
LABEL_TO_ID = {label: i for i, label in enumerate(NER_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


class EncodedNerDataset(Dataset):
    def __init__(self, sequences: list[dict[str, Any]], tokenizer, max_length: int = 512):
        self.sequences = sequences
        self.encodings: list[dict[str, Any]] = []
        for seq in sequences:
            encoding = tokenizer(
                seq["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            word_ids = encoding.word_ids()
            covered = [word_id for word_id in word_ids if word_id is not None]
            if covered and max(covered) + 1 != len(seq["tokens"]):
                raise RuntimeError(
                    f"Sequence {seq['uid']} was truncated: covered {max(covered)+1} of {len(seq['tokens'])} words"
                )
            labels = []
            previous_word = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != previous_word:
                    labels.append(LABEL_TO_ID[seq["labels"][word_id]])
                else:
                    labels.append(-100)
                previous_word = word_id
            item = {key: value for key, value in encoding.items()}
            item["labels"] = labels
            self.encodings.append(item)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        return self.encodings[index]


class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def decode_predictions(predictions: np.ndarray, labels: np.ndarray) -> tuple[list[list[str]], list[list[str]]]:
    pred_ids = predictions.argmax(axis=-1)
    truth_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []
    for pred_row, label_row in zip(pred_ids, labels):
        truth: list[str] = []
        pred: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if int(label_id) == -100:
                continue
            truth.append(ID_TO_LABEL[int(label_id)])
            pred.append(ID_TO_LABEL[int(pred_id)])
        truth_sequences.append(repair_bio(truth))
        pred_sequences.append(repair_bio(pred))
    return truth_sequences, pred_sequences


def compute_metrics(eval_prediction):
    truth, pred = decode_predictions(eval_prediction.predictions, eval_prediction.label_ids)
    metrics = ner_metrics(truth, pred)
    return {key: value for key, value in metrics.items() if not key.startswith("entity_t") and not key.startswith("entity_f")}


def main() -> None:
    seed_everything(42)
    paths = prepare_data()
    bundle, audit = load_ner_bundle(paths)
    tokenizer = AutoTokenizer.from_pretrained(BIOMED_MODEL, use_fast=True)

    train_ds = EncodedNerDataset(bundle["train"], tokenizer)
    dev_ds = EncodedNerDataset(bundle["dev"], tokenizer)
    test_datasets = {
        split: EncodedNerDataset(sequences, tokenizer)
        for split, sequences in bundle.items()
        if split.startswith("test_")
    }

    counts = {label: 0 for label in NER_LABELS}
    for sequence in bundle["train"]:
        for label in sequence["labels"]:
            counts[label] += 1
    total = sum(counts.values())
    weights = np.array([np.sqrt(total / max(1, counts[label])) for label in NER_LABELS], dtype=np.float32)
    weights = np.clip(weights / weights.mean(), 0.35, 3.0)
    class_weights = torch.tensor(weights, dtype=torch.float)

    model = AutoModelForTokenClassification.from_pretrained(
        BIOMED_MODEL,
        num_labels=len(NER_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=42,
        data_seed=42,
        report_to=[],
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        disable_tqdm=False,
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer, padding=True),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-4)],
        class_weights=class_weights,
    )

    started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started
    trainer.save_state()

    rows = []
    prediction_store: dict[str, Any] = {}
    for split, dataset in test_datasets.items():
        output = trainer.predict(dataset, metric_key_prefix=split)
        truth, pred = decode_predictions(output.predictions, output.label_ids)
        metrics = ner_metrics(truth, pred)
        ci = bootstrap_ner_f1(truth, pred)
        rows.append(
            flatten_metric_row(
                "BiomedBERT token classifier",
                "Domain-pretrained SOTA-class Transformer",
                split,
                metrics,
                ci,
                train_seconds,
                {
                    "pretrained_model": BIOMED_MODEL,
                    "epochs_requested": 4,
                    "best_checkpoint": trainer.state.best_model_checkpoint,
                    "best_dev_metric": trainer.state.best_metric,
                    "learning_rate": 3e-5,
                    "effective_batch_size": 16,
                    "class_weights": {label: float(weight) for label, weight in zip(NER_LABELS, weights)},
                },
            )
        )
        prediction_store[split] = pred

    pd.DataFrame(rows).to_csv(OUT / "ner_metrics.csv", index=False)
    with gzip.open(OUT / "predictions.json.gz", "wt", encoding="utf-8") as fp:
        json.dump({"ner": {"BiomedBERT token classifier": prediction_store}}, fp)
    write_json(
        OUT / "training_state.json",
        {
            "model": BIOMED_MODEL,
            "device": str(trainer.args.device),
            "train_seconds": train_seconds,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "log_history": trainer.state.log_history,
            "train_metrics": train_result.metrics,
            "audit_summary": audit["modeling"],
            "class_counts": counts,
            "class_weights": {label: float(weight) for label, weight in zip(NER_LABELS, weights)},
        },
    )
    write_json(OUT / "complete.json", {"status": "success", "model": BIOMED_MODEL})


if __name__ == "__main__":
    main()
