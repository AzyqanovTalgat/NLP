from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
BIOMED_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
BLUE_BERT_URL = "https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/bert_data.zip"
BLUE_DATA_URL = "https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/data_v0.2.zip"
NCBI_URLS = {
    "train": "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/train.tsv",
    "dev": "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/devel.tsv",
    "test": "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/test.tsv",
}
POSITIVE_DDI_LABELS = ["DDI-advice", "DDI-effect", "DDI-int", "DDI-mechanism"]
ALL_DDI_LABELS = ["DDI-false", *POSITIVE_DDI_LABELS]
NER_LABELS = ["O", "B-Disease", "I-Disease"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, destination: Path, retries: int = 4) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "Q1-biomedical-validation/1.0"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            tmp.replace(destination)
            return destination
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def extract_zip(path: Path, target: Path) -> Path:
    marker = target / ".complete"
    if marker.exists():
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(target)
    marker.write_text("ok\n", encoding="utf-8")
    return target


def prepare_data() -> dict[str, Any]:
    blue_zip = download(BLUE_BERT_URL, CACHE / "bert_data.zip")
    blue_data_zip = download(BLUE_DATA_URL, CACHE / "data_v0.2.zip")
    blue_root = extract_zip(blue_zip, CACHE / "blue_bert")
    blue_data_root = extract_zip(blue_data_zip, CACHE / "blue_data")

    ncbi_paths: dict[str, Path] = {}
    for split, url in NCBI_URLS.items():
        ncbi_paths[split] = download(url, CACHE / "ncbi_disease" / f"{split}.tsv")

    return {
        "blue_zip": blue_zip,
        "blue_data_zip": blue_data_zip,
        "blue_root": blue_root,
        "blue_data_root": blue_data_root,
        "ncbi": ncbi_paths,
        "hashes": {
            "bert_data.zip": sha256_file(blue_zip),
            "data_v0.2.zip": sha256_file(blue_data_zip),
            **{f"ncbi_{k}.tsv": sha256_file(v) for k, v in ncbi_paths.items()},
        },
    }


def normalize_ner_label(label: str) -> str:
    value = label.strip()
    if value == "O":
        return "O"
    if value == "B" or value.startswith("B-"):
        return "B-Disease"
    if value == "I" or value.startswith("I-"):
        return "I-Disease"
    raise ValueError(f"Unexpected NER label: {label!r}")


def load_conll(path: Path, source: str, split: str) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        nonlocal tokens, labels
        if tokens:
            uid = f"{source}:{split}:{len(sequences)}"
            sequences.append({"uid": uid, "source": source, "split": split, "tokens": tokens, "labels": labels})
            tokens, labels = [], []

    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for raw in fp:
            line = raw.rstrip("\n\r")
            if not line.strip():
                flush()
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.rsplit(maxsplit=1)
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[-1]
            if token.lower() in {"token", "word"} and label.lower() in {"label", "tag"}:
                continue
            if token == "-DOCSTART-":
                flush()
                continue
            tokens.append(token)
            labels.append(normalize_ner_label(label))
    flush()
    return sequences


def sequence_hash(sequence: dict[str, Any]) -> str:
    normalized = " ".join(sequence["tokens"]).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_sequences(
    sequences: list[dict[str, Any]],
    forbidden_hashes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    forbidden_hashes = forbidden_hashes or set()
    seen: set[str] = set()
    clean: list[dict[str, Any]] = []
    removed_duplicate = 0
    removed_forbidden = 0
    conflicts = 0
    labels_by_hash: dict[str, tuple[str, ...]] = {}
    for seq in sequences:
        digest = sequence_hash(seq)
        labels = tuple(seq["labels"])
        if digest in forbidden_hashes:
            removed_forbidden += 1
            continue
        if digest in seen:
            removed_duplicate += 1
            if labels_by_hash.get(digest) != labels:
                conflicts += 1
            continue
        seen.add(digest)
        labels_by_hash[digest] = labels
        clean.append(seq)
    return clean, {
        "removed_duplicate": removed_duplicate,
        "removed_forbidden": removed_forbidden,
        "annotation_conflicts": conflicts,
    }


def count_entities(label_sequences: Sequence[Sequence[str]]) -> int:
    total = 0
    for labels in label_sequences:
        total += sum(1 for tag in repair_bio(labels) if tag.startswith("B-"))
    return total


def ner_split_summary(sequences: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(x["tokens"]) for x in sequences]
    label_counts = Counter(tag for x in sequences for tag in x["labels"])
    return {
        "sequences": len(sequences),
        "tokens": int(sum(lengths)),
        "entities": count_entities([x["labels"] for x in sequences]),
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_median": float(np.median(lengths)) if lengths else 0.0,
        "length_p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
        "length_max": int(max(lengths)) if lengths else 0,
        "labels": dict(label_counts),
    }


def load_ner_bundle(paths: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base = paths["blue_root"] / "bert_data" / "BC5CDR" / "disease"
    bc5 = {
        "train": load_conll(base / "train.tsv", "BC5CDR-Disease", "train"),
        "dev": load_conll(base / "devel.tsv", "BC5CDR-Disease", "dev"),
        "test": load_conll(base / "test.tsv", "BC5CDR-Disease", "test"),
    }
    ncbi = {
        "train": load_conll(paths["ncbi"]["train"], "NCBI-Disease", "train"),
        "dev": load_conll(paths["ncbi"]["dev"], "NCBI-Disease", "dev"),
        "test": load_conll(paths["ncbi"]["test"], "NCBI-Disease", "test"),
    }

    test_hashes = {sequence_hash(x) for x in bc5["test"] + ncbi["test"]}
    raw_train = bc5["train"] + ncbi["train"]
    raw_dev = bc5["dev"] + ncbi["dev"]
    train, train_dedup = deduplicate_sequences(raw_train, forbidden_hashes=test_hashes)
    train_hashes = {sequence_hash(x) for x in train}
    dev, dev_dedup = deduplicate_sequences(raw_dev, forbidden_hashes=test_hashes | train_hashes)

    bundle = {
        "train": train,
        "dev": dev,
        "test_bc5cdr": bc5["test"],
        "test_ncbi": ncbi["test"],
        "test_combined": bc5["test"] + ncbi["test"],
    }
    audit = {
        "raw": {
            "BC5CDR-Disease": {k: ner_split_summary(v) for k, v in bc5.items()},
            "NCBI-Disease": {k: ner_split_summary(v) for k, v in ncbi.items()},
        },
        "modeling": {k: ner_split_summary(v) for k, v in bundle.items()},
        "deduplication": {"train": train_dedup, "dev": dev_dedup},
        "test_hash_overlap_between_corpora": len(
            {sequence_hash(x) for x in bc5["test"]} & {sequence_hash(x) for x in ncbi["test"]}
        ),
    }
    return bundle, audit


def normalize_ddi_label(label: str) -> str:
    value = str(label).strip()
    aliases = {
        "false": "DDI-false",
        "advice": "DDI-advice",
        "effect": "DDI-effect",
        "int": "DDI-int",
        "mechanism": "DDI-mechanism",
    }
    value = aliases.get(value.lower(), value)
    if value not in ALL_DDI_LABELS:
        raise ValueError(f"Unexpected DDI label: {label!r}")
    return value


def load_ddi_tsv(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    lower = {c.lower(): c for c in df.columns}
    idx_col = lower.get("index", df.columns[0])
    sent_col = lower.get("sentence", df.columns[1])
    label_col = lower.get("label", df.columns[2])
    out = df[[idx_col, sent_col, label_col]].copy()
    out.columns = ["index", "sentence", "label"]
    out["label"] = out["label"].map(normalize_ddi_label)
    out["split"] = split
    return out


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def prepare_entity_markers(text: str) -> str:
    pattern = re.compile(r"@[A-Za-z0-9_\-]+\$")
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    if len(matches) == 1 and "-" in matches[0].group(0).strip("@$ "):
        start, end = matches[0].span()
        return text[:start] + " [E1] DRUG [/E1] [E2] DRUG [/E2] " + text[end:]
    replacements = [" [E1] DRUG [/E1] ", " [E2] DRUG [/E2] "]
    pieces: list[str] = []
    last = 0
    for i, match in enumerate(matches):
        pieces.append(text[last:match.start()])
        pieces.append(replacements[i] if i < 2 else " DRUG ")
        last = match.end()
    pieces.append(text[last:])
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def ddi_split_summary(df: pd.DataFrame) -> dict[str, Any]:
    lengths = df["sentence"].str.split().map(len).to_numpy() if len(df) else np.array([])
    duplicate_texts = int(df["sentence"].map(normalized_text).duplicated().sum()) if len(df) else 0
    conflicting = 0
    if len(df):
        grouped = df.assign(_norm=df["sentence"].map(normalized_text)).groupby("_norm")["label"].nunique()
        conflicting = int((grouped > 1).sum())
    return {
        "instances": int(len(df)),
        "labels": {k: int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "positive_instances": int(df["label"].isin(POSITIVE_DDI_LABELS).sum()),
        "negative_instances": int((df["label"] == "DDI-false").sum()),
        "positive_rate": float(df["label"].isin(POSITIVE_DDI_LABELS).mean()) if len(df) else 0.0,
        "length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "length_median": float(np.median(lengths)) if len(lengths) else 0.0,
        "length_p95": float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
        "length_max": int(lengths.max()) if len(lengths) else 0,
        "duplicate_indices": int(df["index"].duplicated().sum()),
        "duplicate_normalized_texts": duplicate_texts,
        "conflicting_normalized_text_labels": conflicting,
    }


def load_re_bundle(paths: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    base = paths["blue_root"] / "bert_data" / "ddi2013-type"
    train = load_ddi_tsv(base / "train.tsv", "train")
    dev = load_ddi_tsv(base / "dev.tsv", "dev")
    test = load_ddi_tsv(base / "test.tsv", "test")

    test_ids = set(test["index"])
    test_texts = set(test["sentence"].map(normalized_text))
    before_train = len(train)
    train = train.loc[~train["index"].isin(test_ids) & ~train["sentence"].map(normalized_text).isin(test_texts)].copy()
    train_ids = set(train["index"])
    train_texts = set(train["sentence"].map(normalized_text))
    before_dev = len(dev)
    dev = dev.loc[
        ~dev["index"].isin(test_ids | train_ids)
        & ~dev["sentence"].map(normalized_text).isin(test_texts | train_texts)
    ].copy()

    for frame in (train, dev, test):
        frame["marked_sentence"] = frame["sentence"].map(prepare_entity_markers)

    bundle = {"train": train.reset_index(drop=True), "dev": dev.reset_index(drop=True), "test": test.reset_index(drop=True)}
    audit = {
        "splits": {k: ddi_split_summary(v) for k, v in bundle.items()},
        "removed_for_leakage": {"train": before_train - len(train), "dev": before_dev - len(dev)},
        "cross_split_index_overlap": {
            "train_dev": len(set(train["index"]) & set(dev["index"])),
            "train_test": len(set(train["index"]) & set(test["index"])),
            "dev_test": len(set(dev["index"]) & set(test["index"])),
        },
        "cross_split_exact_text_overlap": {
            "train_dev": len(set(train["sentence"].map(normalized_text)) & set(dev["sentence"].map(normalized_text))),
            "train_test": len(set(train["sentence"].map(normalized_text)) & set(test["sentence"].map(normalized_text))),
            "dev_test": len(set(dev["sentence"].map(normalized_text)) & set(test["sentence"].map(normalized_text))),
        },
    }
    return bundle, audit


def repair_bio(labels: Sequence[str]) -> list[str]:
    repaired: list[str] = []
    previous = "O"
    for raw in labels:
        tag = normalize_ner_label(raw)
        if tag.startswith("I-") and previous == "O":
            tag = "B-" + tag[2:]
        repaired.append(tag)
        previous = tag
    return repaired


def labels_to_spans(labels: Sequence[str]) -> set[tuple[int, int, str]]:
    labels = repair_bio(labels)
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    entity_type: str | None = None
    for i, tag in enumerate([*labels, "O"]):
        if tag.startswith("B-"):
            if start is not None and entity_type is not None:
                spans.add((start, i, entity_type))
            start = i
            entity_type = tag[2:]
        elif tag.startswith("I-"):
            current = tag[2:]
            if start is None or current != entity_type:
                if start is not None and entity_type is not None:
                    spans.add((start, i, entity_type))
                start = i
                entity_type = current
        else:
            if start is not None and entity_type is not None:
                spans.add((start, i, entity_type))
            start = None
            entity_type = None
    return spans


def ner_metrics(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("NER prediction and target sequence counts differ")
    tp = fp = fn = 0
    flat_true: list[str] = []
    flat_pred: list[str] = []
    for truth, pred in zip(y_true, y_pred):
        n = min(len(truth), len(pred))
        truth_r = repair_bio(truth[:n])
        pred_r = repair_bio(pred[:n])
        true_spans = labels_to_spans(truth_r)
        pred_spans = labels_to_spans(pred_r)
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
        flat_true.extend(truth_r)
        flat_pred.extend(pred_r)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "entity_tp": float(tp),
        "entity_fp": float(fp),
        "entity_fn": float(fn),
        "token_accuracy": accuracy_score(flat_true, flat_pred) if flat_true else 0.0,
        "token_macro_f1": f1_score(flat_true, flat_pred, labels=NER_LABELS, average="macro", zero_division=0)
        if flat_true
        else 0.0,
    }


def bootstrap_ner_f1(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    n_bootstrap: int = 500,
    seed: int = 2026,
) -> tuple[float, float]:
    if not y_true:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        truth = [y_true[i] for i in idx]
        pred = [y_pred[i] for i in idx]
        scores.append(ner_metrics(truth, pred)["entity_f1"])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def re_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=POSITIVE_DDI_LABELS,
        average="micro",
        zero_division=0,
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=POSITIVE_DDI_LABELS,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "positive_micro_precision": p_micro,
        "positive_micro_recall": r_micro,
        "positive_micro_f1": f_micro,
        "positive_macro_precision": p_macro,
        "positive_macro_recall": r_macro,
        "positive_macro_f1": f_macro,
        "macro_f1_all": f1_score(y_true, y_pred, labels=ALL_DDI_LABELS, average="macro", zero_division=0),
        "weighted_f1_all": f1_score(y_true, y_pred, labels=ALL_DDI_LABELS, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def bootstrap_re_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 2026,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not len(y_true):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(re_metrics(y_true[idx].tolist(), y_pred[idx].tolist())["positive_macro_f1"])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def paired_bootstrap_pvalue_ner(
    y_true: Sequence[Sequence[str]],
    pred_a: Sequence[Sequence[str]],
    pred_b: Sequence[Sequence[str]],
    n_bootstrap: int = 1000,
    seed: int = 2027,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        truth = [y_true[i] for i in idx]
        a = [pred_a[i] for i in idx]
        b = [pred_b[i] for i in idx]
        diffs.append(ner_metrics(truth, a)["entity_f1"] - ner_metrics(truth, b)["entity_f1"])
    diffs = np.asarray(diffs)
    return float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))


def paired_bootstrap_pvalue_re(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    n_bootstrap: int = 2000,
    seed: int = 2027,
) -> float:
    truth = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(truth), len(truth))
        diffs.append(
            re_metrics(truth[idx].tolist(), a[idx].tolist())["positive_macro_f1"]
            - re_metrics(truth[idx].tolist(), b[idx].tolist())["positive_macro_f1"]
        )
    diffs = np.asarray(diffs)
    return float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))


def token_shape(token: str) -> str:
    chars = []
    for ch in token:
        if ch.isupper():
            chars.append("X")
        elif ch.islower():
            chars.append("x")
        elif ch.isdigit():
            chars.append("d")
        else:
            chars.append(ch)
    compact = []
    for ch in chars:
        if not compact or compact[-1] != ch:
            compact.append(ch)
    return "".join(compact)[:16]


def token_features(tokens: Sequence[str], i: int) -> dict[str, Any]:
    word = tokens[i]
    lower = word.lower()
    features: dict[str, Any] = {
        "bias": 1.0,
        "word.lower": lower,
        "word.shape": token_shape(word),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.hasdigit": any(ch.isdigit() for ch in word),
        "word.hashyphen": "-" in word,
        "word.len_bin": min(len(word), 15),
        "prefix1": lower[:1],
        "prefix2": lower[:2],
        "prefix3": lower[:3],
        "suffix1": lower[-1:],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],
        "suffix4": lower[-4:],
    }
    if i > 0:
        prev = tokens[i - 1]
        features.update(
            {
                "-1.lower": prev.lower(),
                "-1.shape": token_shape(prev),
                "-1.istitle": prev.istitle(),
                "-1.isupper": prev.isupper(),
            }
        )
    else:
        features["BOS"] = True
    if i + 1 < len(tokens):
        nxt = tokens[i + 1]
        features.update(
            {
                "+1.lower": nxt.lower(),
                "+1.shape": token_shape(nxt),
                "+1.istitle": nxt.istitle(),
                "+1.isupper": nxt.isupper(),
            }
        )
    else:
        features["EOS"] = True
    return features


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_metric_row(
    model: str,
    family: str,
    split: str,
    metrics: dict[str, Any],
    ci: tuple[float, float] | None,
    train_seconds: float,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "family": family,
        "split": split,
        "train_seconds": train_seconds,
        "hyperparameters": json.dumps(hyperparameters or {}, sort_keys=True),
    }
    row.update(metrics)
    if ci is not None:
        row["f1_ci95_low"] = ci[0]
        row["f1_ci95_high"] = ci[1]
    return row
