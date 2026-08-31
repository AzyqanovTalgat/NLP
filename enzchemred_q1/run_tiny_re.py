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

from run_transformer import (
    REDataset,
    bootstrap_re,
    choose_threshold,
    load_jsonl,
    make_re_collate,
    relation_metrics,
    score_re_texts,
)

MODEL_NAME = "dmis-lab/TinyPubMedBERT-v1.0"
SPECIAL_MARKERS = ["<C1>", "</C1>", "<C2>", "</C2>"]
SEED = 42


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def clone_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_all()

    rows = {
        split: load_jsonl(args.data_dir / f"re_{split}.jsonl")
        for split in ["train", "dev", "id_test", "temporal_ood_test"]
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_MARKERS})
    datasets = {split: REDataset(value, tokenizer, max_length=160) for split, value in rows.items()}
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(tokenizer))

    counts = Counter(int(row["binary_label"]) for row in rows["train"])
    class_weights = torch.tensor([1.0, math.sqrt(counts[0] / max(1, counts[1]))], dtype=torch.float)
    loader = DataLoader(
        datasets["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=make_re_collate(tokenizer),
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 6
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.08 * total_steps), total_steps)
    best_f1 = -1.0
    best_threshold = 0.5
    best_state = None
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            batch.pop("indices")
            labels = batch.pop("labels")
            logits = model(**batch).logits
            ce = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="none")
            pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
            loss = (((1.0 - pt) ** 0.75) * ce).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())
            if step % 150 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": running / step}), flush=True)

        dev_scores = score_re_texts(model, tokenizer, [x["marked_sentence"] for x in rows["dev"]], batch_size=64, max_length=160)
        y_dev = np.asarray([int(x["binary_label"]) for x in rows["dev"]])
        threshold, dev_f1 = choose_threshold(y_dev, dev_scores)
        dev_pred = (dev_scores >= threshold).astype(int)
        dev_metrics = relation_metrics(y_dev, dev_pred, dev_scores)
        history.append({"epoch": epoch, "train_loss": running / max(1, len(loader)), "threshold": threshold, "dev": dev_metrics})
        print(json.dumps(history[-1]), flush=True)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_threshold = threshold
            best_state = clone_state(model)

    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - started
    results = []
    predictions = {}
    for split in ["dev", "id_test", "temporal_ood_test"]:
        scores = score_re_texts(model, tokenizer, [x["marked_sentence"] for x in rows[split]], batch_size=64, max_length=160)
        truth = np.asarray([int(x["binary_label"]) for x in rows[split]])
        pred = (scores >= best_threshold).astype(int)
        metrics = relation_metrics(truth, pred, scores)
        ci = bootstrap_re(rows[split], truth, pred, n=500) if split != "dev" else (float("nan"), float("nan"))
        results.append({
            "model": "TinyPubMedBERT full fine-tuning",
            "family": "compact biomedical Transformer",
            "split": split,
            **metrics,
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "threshold": best_threshold,
            "train_seconds": train_seconds,
            "best_dev_f1": best_f1,
        })
        predictions[split] = {"pred": pred.tolist(), "score": scores.tolist()}

    with (args.out_dir / "re_tiny_transformer_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with gzip.open(args.out_dir / "re_tiny_transformer_predictions.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(predictions, fh)
    (args.out_dir / "re_tiny_training_state.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "seed": SEED,
        "epochs": epochs,
        "best_dev_f1": best_f1,
        "best_threshold": best_threshold,
        "history": history,
        "class_counts": dict(counts),
        "class_weights": class_weights.tolist(),
        "train_seconds": train_seconds,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
