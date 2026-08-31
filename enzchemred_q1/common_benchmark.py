from __future__ import annotations

import gzip
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ENZCHEMRED_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("ENZCHEMRED_RESULTS", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

NER_LABELS = ["O", "B-Chemical", "I-Chemical", "B-Protein", "I-Protein"]
LABEL_TO_ID = {label: i for i, label in enumerate(NER_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
ENTITY_TYPES = ["Chemical", "Protein"]
SPLITS = ["train", "dev", "id_test", "temporal_ood_test"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except Exception:
        pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_task(task: str) -> dict[str, list[dict[str, Any]]]:
    return {split: load_jsonl(DATA / f"{task}_{split}.jsonl") for split in SPLITS}


def repair_bio(labels: Sequence[str]) -> list[str]:
    repaired: list[str] = []
    previous = "O"
    for raw in labels:
        tag = raw
        if tag.startswith("I-"):
            entity_type = tag[2:]
            if previous not in {f"B-{entity_type}", f"I-{entity_type}"}:
                tag = f"B-{entity_type}"
        repaired.append(tag)
        previous = tag
    return repaired


def spans_from_labels(row: dict[str, Any], labels: Sequence[str]) -> set[tuple[int, int, str]]:
    labels = repair_bio(labels)
    offsets = row["token_offsets"]
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    end: int | None = None
    entity_type: str | None = None
    for i, tag in enumerate([*labels, "O"]):
        if tag.startswith("B-"):
            if start is not None:
                spans.add((start, int(end), str(entity_type)))
            entity_type = tag[2:]
            start, end = offsets[i]
        elif tag.startswith("I-"):
            current_type = tag[2:]
            if start is None or current_type != entity_type:
                if start is not None:
                    spans.add((start, int(end), str(entity_type)))
                entity_type = current_type
                start, end = offsets[i]
            else:
                end = offsets[i][1]
        else:
            if start is not None:
                spans.add((start, int(end), str(entity_type)))
            start = end = entity_type = None
    return spans


def ner_metrics(rows: list[dict[str, Any]], predictions: list[list[str]]) -> dict[str, float]:
    tp = fp = fn = 0
    per_type = {entity_type: [0, 0, 0] for entity_type in ENTITY_TYPES}
    token_true: list[str] = []
    token_pred: list[str] = []
    for row, predicted in zip(rows, predictions):
        truth = spans_from_labels(row, row["labels"])
        guess = spans_from_labels(row, predicted)
        tp += len(truth & guess)
        fp += len(guess - truth)
        fn += len(truth - guess)
        for entity_type in ENTITY_TYPES:
            true_type = {span for span in truth if span[2] == entity_type}
            pred_type = {span for span in guess if span[2] == entity_type}
            per_type[entity_type][0] += len(true_type & pred_type)
            per_type[entity_type][1] += len(pred_type - true_type)
            per_type[entity_type][2] += len(true_type - pred_type)
        token_true.extend(repair_bio(row["labels"]))
        token_pred.extend(repair_bio(predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    exact_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    type_f1: dict[str, float] = {}
    for entity_type, (type_tp, type_fp, type_fn) in per_type.items():
        p = type_tp / (type_tp + type_fp) if type_tp + type_fp else 0.0
        r = type_tp / (type_tp + type_fn) if type_tp + type_fn else 0.0
        type_f1[entity_type] = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "exact_f1": exact_f1,
        "macro_type_f1": float(np.mean(list(type_f1.values()))),
        "chemical_f1": type_f1["Chemical"],
        "protein_f1": type_f1["Protein"],
        "token_accuracy": accuracy_score(token_true, token_pred),
        "token_macro_f1": f1_score(
            token_true, token_pred, labels=NER_LABELS, average="macro", zero_division=0
        ),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def split_flat_predictions(rows: list[dict[str, Any]], flat: Sequence[str]) -> list[list[str]]:
    predictions: list[list[str]] = []
    position = 0
    for row in rows:
        length = len(row["tokens"])
        predictions.append(list(flat[position : position + length]))
        position += length
    if position != len(flat):
        raise ValueError("Flat NER prediction length mismatch")
    return predictions


def bootstrap_ner(
    rows: list[dict[str, Any]], predictions: list[list[str]], iterations: int = 500, seed: int = 2026
) -> tuple[float, float]:
    by_document: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_document[row["pmid"]].append(i)
    documents = list(by_document)
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        sampled = generator.choice(documents, size=len(documents), replace=True)
        sampled_rows: list[dict[str, Any]] = []
        sampled_predictions: list[list[str]] = []
        for document in sampled:
            for index in by_document[document]:
                sampled_rows.append(rows[index])
                sampled_predictions.append(predictions[index])
        scores.append(ner_metrics(sampled_rows, sampled_predictions)["exact_f1"])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def re_metrics(y_true: Sequence[int], y_pred: Sequence[int], scores: Sequence[float] | None = None) -> dict[str, float]:
    precision, recall, positive_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "positive_f1": float(positive_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if scores is not None:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return result


def choose_threshold(y_true: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    scores_array = np.asarray(scores)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        current = f1_score(y_true, scores_array >= threshold, zero_division=0)
        if current > best_f1:
            best_threshold, best_f1 = float(threshold), float(current)
    return best_threshold, best_f1


def bootstrap_re(
    rows: list[dict[str, Any]], y_true: Sequence[int], y_pred: Sequence[int], iterations: int = 1000, seed: int = 2026
) -> tuple[float, float]:
    by_document: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_document[row["pmid"]].append(i)
    documents = list(by_document)
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        sampled = generator.choice(documents, size=len(documents), replace=True)
        indices = np.concatenate([np.asarray(by_document[document]) for document in sampled])
        scores.append(re_metrics(y_true_array[indices], y_pred_array[indices])["positive_f1"])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)


def normalized_model_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
