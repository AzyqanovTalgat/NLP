from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common_benchmark import bootstrap_re, choose_threshold, load_task, re_metrics, seed_everything
from run_transformer import RelationCollator, SPECIAL_MARKERS, TransformerRelationDataset, predict_relation

MODEL_NAME = "dmis-lab/TinyPubMedBERT-v1.0"
SEED = 42


def contiguous_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("enzchemred_q1/prepared"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    # common_benchmark reads the same immutable prepared directory used by all other models.
    rows = load_task("re")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_MARKERS})
    datasets = {split: TransformerRelationDataset(value, tokenizer, max_length=160) for split, value in rows.items()}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=32 if split == "train" else 64,
            shuffle=split == "train",
            collate_fn=RelationCollator(tokenizer),
            num_workers=2,
        )
        for split, dataset in datasets.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    counts = Counter(int(row["binary_label"]) for row in rows["train"])
    class_weights = torch.tensor([1.0, math.sqrt(counts[0] / max(1, counts[1]))], dtype=torch.float, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.08 * total_steps)), total_steps)

    best_f1 = -1.0
    best_threshold = 0.5
    best_state = None
    best_epoch = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(loaders["train"], start=1):
            batch.pop("row_indices")
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            ce = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="none")
            pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
            loss = (((1.0 - pt) ** 0.75) * ce).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if step % 150 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": float(np.mean(losses))}), flush=True)

        dev_scores = predict_relation(model, loaders["dev"], len(rows["dev"]), device)
        dev_truth = np.asarray([int(row["binary_label"]) for row in rows["dev"]])
        threshold, dev_f1 = choose_threshold(dev_truth, dev_scores)
        dev_metrics = re_metrics(dev_truth, dev_scores >= threshold, dev_scores)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "threshold": threshold, **dev_metrics})
        print(json.dumps(history[-1]), flush=True)
        if dev_f1 > best_f1:
            best_f1 = float(dev_f1)
            best_threshold = float(threshold)
            best_epoch = epoch
            best_state = contiguous_state(model)

    if best_state is None:
        raise RuntimeError("No valid TinyPubMedBERT checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)
    train_seconds = time.perf_counter() - started

    results = []
    predictions = {}
    for split in ["dev", "id_test", "temporal_ood_test"]:
        scores = predict_relation(model, loaders[split], len(rows[split]), device)
        truth = np.asarray([int(row["binary_label"]) for row in rows[split]])
        pred = (scores >= best_threshold).astype(int)
        metrics = re_metrics(truth, pred, scores)
        ci = bootstrap_re(rows[split], truth, pred, iterations=500) if split != "dev" else (float("nan"), float("nan"))
        results.append({
            "model": f"{args.model_name} entity-marker classifier",
            "family": "compact biomedical Transformer",
            "split": split,
            **metrics,
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "threshold": best_threshold,
            "train_seconds": train_seconds,
            "best_epoch": best_epoch,
            "best_dev_f1": best_f1,
        })
        predictions[split] = {"predictions": pred.tolist(), "scores": scores.tolist()}

    with (args.out_dir / "re_tiny_transformer_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with gzip.open(args.out_dir / "re_tiny_transformer_predictions.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({results[0]["model"]: predictions}, fh)
    (args.out_dir / "re_tiny_training_state.json").write_text(json.dumps({
        "model": args.model_name,
        "seed": SEED,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_dev_f1": best_f1,
        "best_threshold": best_threshold,
        "history": history,
        "class_counts": dict(counts),
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_seconds": train_seconds,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
