from __future__ import annotations

import copy
import gzip
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import sklearn_crfsuite

from common import (
    ALL_DDI_LABELS,
    NER_LABELS,
    POSITIVE_DDI_LABELS,
    RESULTS,
    bootstrap_ner_f1,
    bootstrap_re_f1,
    flatten_metric_row,
    load_ner_bundle,
    load_re_bundle,
    ner_metrics,
    prepare_data,
    re_metrics,
    repair_bio,
    seed_everything,
    token_features,
    write_json,
)

OUT = RESULTS / "stage_baselines"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reshape_predictions(flat: Sequence[str], lengths: Sequence[int]) -> list[list[str]]:
    predictions: list[list[str]] = []
    offset = 0
    for length in lengths:
        predictions.append(repair_bio(list(flat[offset : offset + length])))
        offset += length
    if offset != len(flat):
        raise ValueError("Prediction reshape length mismatch")
    return predictions


def fit_token_baselines(bundle: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = bundle["train"]
    dev = bundle["dev"]
    tests = {k: v for k, v in bundle.items() if k.startswith("test_")}

    train_features = [token_features(seq["tokens"], i) for seq in train for i in range(len(seq["tokens"]))]
    dev_features = [token_features(seq["tokens"], i) for seq in dev for i in range(len(seq["tokens"]))]
    train_labels = [tag for seq in train for tag in seq["labels"]]
    dev_labels_seq = [seq["labels"] for seq in dev]
    dev_lengths = [len(seq["tokens"]) for seq in dev]

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_features)
    x_dev = vectorizer.transform(dev_features)
    joblib.dump(vectorizer, OUT / "ner_dict_vectorizer.joblib", compress=3)

    test_matrices: dict[str, Any] = {}
    for split, sequences in tests.items():
        features = [token_features(seq["tokens"], i) for seq in sequences for i in range(len(seq["tokens"]))]
        test_matrices[split] = vectorizer.transform(features)

    rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}

    specifications = {
        "Logistic Regression": (
            lambda c: LogisticRegression(
                C=c,
                solver="saga",
                max_iter=300,
                tol=1e-3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            [0.25, 1.0, 4.0],
        ),
        "Linear SVM": (
            lambda c: LinearSVC(C=c, class_weight="balanced", random_state=42, max_iter=5000),
            [0.25, 1.0, 4.0],
        ),
    }

    for model_name, (factory, candidates) in specifications.items():
        start = time.perf_counter()
        best_model = None
        best_c = None
        best_score = -1.0
        tuning: list[dict[str, float]] = []
        for c in candidates:
            model = factory(c)
            model.fit(x_train, train_labels)
            dev_pred = reshape_predictions(model.predict(x_dev), dev_lengths)
            score = ner_metrics(dev_labels_seq, dev_pred)["entity_f1"]
            tuning.append({"C": c, "dev_entity_f1": score})
            if score > best_score:
                best_score = score
                best_model = model
                best_c = c
        assert best_model is not None
        train_seconds = time.perf_counter() - start
        joblib.dump(best_model, OUT / f"ner_{model_name.lower().replace(' ', '_')}.joblib", compress=3)
        model_predictions: dict[str, Any] = {}
        for split, sequences in tests.items():
            lengths = [len(seq["tokens"]) for seq in sequences]
            pred = reshape_predictions(best_model.predict(test_matrices[split]), lengths)
            truth = [seq["labels"] for seq in sequences]
            metrics = ner_metrics(truth, pred)
            ci = bootstrap_ner_f1(truth, pred)
            rows.append(
                flatten_metric_row(
                    model_name,
                    "Classical baseline",
                    split,
                    metrics,
                    ci,
                    train_seconds,
                    {"C": best_c, "dev_entity_f1": best_score, "tuning": tuning},
                )
            )
            model_predictions[split] = pred
        predictions[model_name] = model_predictions

    # Sequence baseline: CRF.
    train_x_seq = [[token_features(seq["tokens"], i) for i in range(len(seq["tokens"]))] for seq in train]
    train_y_seq = [seq["labels"] for seq in train]
    dev_x_seq = [[token_features(seq["tokens"], i) for i in range(len(seq["tokens"]))] for seq in dev]
    start = time.perf_counter()
    best_crf = None
    best_params = None
    best_score = -1.0
    tuning = []
    for c1, c2 in [(0.05, 0.05), (0.1, 0.1), (0.1, 0.01), (0.01, 0.1)]:
        model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=100,
            all_possible_transitions=True,
        )
        model.fit(train_x_seq, train_y_seq)
        dev_pred = [repair_bio(x) for x in model.predict(dev_x_seq)]
        score = ner_metrics(dev_labels_seq, dev_pred)["entity_f1"]
        tuning.append({"c1": c1, "c2": c2, "dev_entity_f1": score})
        if score > best_score:
            best_score = score
            best_crf = model
            best_params = {"c1": c1, "c2": c2}
    assert best_crf is not None
    train_seconds = time.perf_counter() - start
    joblib.dump(best_crf, OUT / "ner_crf.joblib", compress=3)
    model_predictions = {}
    for split, sequences in tests.items():
        x = [[token_features(seq["tokens"], i) for i in range(len(seq["tokens"]))] for seq in sequences]
        pred = [repair_bio(x) for x in best_crf.predict(x)]
        truth = [seq["labels"] for seq in sequences]
        rows.append(
            flatten_metric_row(
                "Linear-chain CRF",
                "Classical sequence baseline",
                split,
                ner_metrics(truth, pred),
                bootstrap_ner_f1(truth, pred),
                train_seconds,
                {**(best_params or {}), "dev_entity_f1": best_score, "tuning": tuning},
            )
        )
        model_predictions[split] = pred
    predictions["Linear-chain CRF"] = model_predictions
    return rows, predictions


class NerDataset(Dataset):
    def __init__(self, sequences: list[dict[str, Any]], word_to_id: dict[str, int]):
        self.items = []
        label_to_id = {label: i for i, label in enumerate(NER_LABELS)}
        for seq in sequences:
            ids = [word_to_id.get(token.lower(), word_to_id["<UNK>"]) for token in seq["tokens"]]
            labels = [label_to_id[tag] for tag in seq["labels"]]
            self.items.append((ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def ner_collate(batch):
    lengths = torch.tensor([len(ids) for ids, _ in batch], dtype=torch.long)
    max_len = int(lengths.max())
    words = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, (ids, tags) in enumerate(batch):
        words[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(tags)] = torch.tensor(tags, dtype=torch.long)
        mask[i, : len(ids)] = True
    return words, labels, mask, lengths


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 160, dropout: float = 0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, len(NER_LABELS))

    def forward(self, words: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(words))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=words.size(1))
        return self.classifier(self.dropout(output))


def predict_ner_lstm(model: nn.Module, loader: DataLoader) -> list[list[str]]:
    model.eval()
    id_to_label = dict(enumerate(NER_LABELS))
    all_pred: list[list[str]] = []
    with torch.no_grad():
        for words, labels, mask, lengths in loader:
            logits = model(words.to(DEVICE), lengths.to(DEVICE))
            pred = logits.argmax(-1).cpu()
            for i, length in enumerate(lengths.tolist()):
                all_pred.append(repair_bio([id_to_label[int(x)] for x in pred[i, :length]]))
    return all_pred


def fit_ner_bilstm(bundle: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    counter = Counter(token.lower() for seq in bundle["train"] for token in seq["tokens"])
    vocab = ["<PAD>", "<UNK>"] + [word for word, _ in counter.most_common(50000)]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    write_json(OUT / "ner_bilstm_vocab.json", word_to_id)

    train_ds = NerDataset(bundle["train"], word_to_id)
    dev_ds = NerDataset(bundle["dev"], word_to_id)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=ner_collate, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=ner_collate, num_workers=0)
    test_loaders = {
        split: DataLoader(NerDataset(sequences, word_to_id), batch_size=64, shuffle=False, collate_fn=ner_collate)
        for split, sequences in bundle.items()
        if split.startswith("test_")
    }

    counts = Counter(tag for seq in bundle["train"] for tag in seq["labels"])
    weights = torch.tensor(
        [math.sqrt(sum(counts.values()) / max(1, counts[label])) for label in NER_LABELS],
        dtype=torch.float,
        device=DEVICE,
    )
    weights = weights / weights.mean()

    model = BiLSTMTagger(len(vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    best_state = None
    best_dev = -1.0
    patience = 0
    curve: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, 21):
        model.train()
        losses = []
        for words, labels, mask, lengths in train_loader:
            words = words.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(words, lengths)
            loss = criterion(logits.view(-1, len(NER_LABELS)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_pred = predict_ner_lstm(model, dev_loader)
        dev_truth = [seq["labels"] for seq in bundle["dev"]]
        dev_f1 = ner_metrics(dev_truth, dev_pred)["entity_f1"]
        curve.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_entity_f1": dev_f1})
        if dev_f1 > best_dev + 1e-4:
            best_dev = dev_f1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 4:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - start
    torch.save(model.state_dict(), OUT / "ner_bilstm.pt")

    rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    for split, loader in test_loaders.items():
        pred = predict_ner_lstm(model, loader)
        truth = [seq["labels"] for seq in bundle[split]]
        rows.append(
            flatten_metric_row(
                "BiLSTM",
                "Deep NLP",
                split,
                ner_metrics(truth, pred),
                bootstrap_ner_f1(truth, pred),
                train_seconds,
                {"embedding_dim": 128, "hidden_dim": 160, "best_dev_entity_f1": best_dev, "epochs": len(curve)},
            )
        )
        predictions[split] = pred
    return rows, predictions, {"curve": curve, "best_dev_entity_f1": best_dev}


def make_re_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    sublinear_tf=True,
                    max_features=70000,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
        ]
    )


def fit_re_baselines(bundle: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    train, dev, test = bundle["train"], bundle["dev"], bundle["test"]
    vectorizer = make_re_vectorizer()
    x_train = vectorizer.fit_transform(train["marked_sentence"])
    x_dev = vectorizer.transform(dev["marked_sentence"])
    x_test = vectorizer.transform(test["marked_sentence"])
    joblib.dump(vectorizer, OUT / "re_tfidf.joblib", compress=3)

    model_specs = {
        "Complement Naive Bayes": (lambda value: ComplementNB(alpha=value), [0.1, 0.5, 1.0]),
        "Logistic Regression": (
            lambda value: LogisticRegression(
                C=value,
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            ),
            [0.25, 1.0, 4.0],
        ),
        "Linear SVM": (
            lambda value: LinearSVC(C=value, class_weight="balanced", random_state=42, max_iter=10000),
            [0.25, 1.0, 4.0],
        ),
    }

    rows: list[dict[str, Any]] = []
    predictions: dict[str, list[str]] = {}
    for model_name, (factory, candidates) in model_specs.items():
        start = time.perf_counter()
        best_model = None
        best_value = None
        best_score = -1.0
        tuning = []
        for value in candidates:
            model = factory(value)
            model.fit(x_train, train["label"])
            pred = model.predict(x_dev)
            score = re_metrics(dev["label"], pred)["positive_macro_f1"]
            tuning.append({"value": value, "dev_positive_macro_f1": score})
            if score > best_score:
                best_score = score
                best_value = value
                best_model = model
        assert best_model is not None
        train_seconds = time.perf_counter() - start
        pred = best_model.predict(x_test).tolist()
        rows.append(
            flatten_metric_row(
                model_name,
                "Classical baseline",
                "test",
                re_metrics(test["label"], pred),
                bootstrap_re_f1(test["label"], pred),
                train_seconds,
                {"selected_value": best_value, "dev_positive_macro_f1": best_score, "tuning": tuning},
            )
        )
        predictions[model_name] = pred
        joblib.dump(best_model, OUT / f"re_{model_name.lower().replace(' ', '_')}.joblib", compress=3)
    return rows, predictions


TOKEN_PATTERN = re.compile(r"\[/?E[12]\]|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]")


def re_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


class ReDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, word_to_id: dict[str, int], label_to_id: dict[str, int], max_len: int = 192):
        self.items = []
        for text, label in zip(frame["marked_sentence"], frame["label"]):
            tokens = re_tokenize(text)[:max_len]
            ids = [word_to_id.get(token.lower(), word_to_id["<UNK>"]) for token in tokens]
            self.items.append((ids, label_to_id[label]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def re_collate(batch):
    lengths = torch.tensor([len(ids) for ids, _ in batch], dtype=torch.long)
    max_len = int(lengths.max())
    words = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, (ids, _) in enumerate(batch):
        words[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = True
    return words, labels, mask, lengths


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, embedding_dim: int = 128, hidden_dim: int = 160):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, words, mask, lengths):
        embedded = self.dropout(self.embedding(words))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=words.size(1))
        scores = self.attention(output).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), output).squeeze(1)
        return self.classifier(self.dropout(pooled))


def predict_re_lstm(model, loader, id_to_label):
    model.eval()
    predictions = []
    with torch.no_grad():
        for words, labels, mask, lengths in loader:
            logits = model(words.to(DEVICE), mask.to(DEVICE), lengths.to(DEVICE))
            ids = logits.argmax(-1).cpu().tolist()
            predictions.extend(id_to_label[i] for i in ids)
    return predictions


def fit_re_bilstm(bundle: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    counter = Counter(token.lower() for text in bundle["train"]["marked_sentence"] for token in re_tokenize(text))
    vocab = ["<PAD>", "<UNK>"] + [word for word, _ in counter.most_common(40000)]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    label_to_id = {label: i for i, label in enumerate(ALL_DDI_LABELS)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    write_json(OUT / "re_bilstm_vocab.json", word_to_id)

    train_loader = DataLoader(
        ReDataset(bundle["train"], word_to_id, label_to_id),
        batch_size=64,
        shuffle=True,
        collate_fn=re_collate,
    )
    dev_loader = DataLoader(ReDataset(bundle["dev"], word_to_id, label_to_id), batch_size=128, collate_fn=re_collate)
    test_loader = DataLoader(ReDataset(bundle["test"], word_to_id, label_to_id), batch_size=128, collate_fn=re_collate)

    counts = bundle["train"]["label"].value_counts().to_dict()
    weights = torch.tensor(
        [math.sqrt(len(bundle["train"]) / max(1, counts.get(label, 0))) for label in ALL_DDI_LABELS],
        dtype=torch.float,
        device=DEVICE,
    )
    weights = weights / weights.mean()

    model = BiLSTMAttention(len(vocab), len(ALL_DDI_LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_state = None
    best_dev = -1.0
    patience = 0
    curve = []
    start = time.perf_counter()
    for epoch in range(1, 26):
        model.train()
        losses = []
        for words, labels, mask, lengths in train_loader:
            words = words.to(DEVICE)
            labels = labels.to(DEVICE)
            mask = mask.to(DEVICE)
            lengths = lengths.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(words, mask, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_pred = predict_re_lstm(model, dev_loader, id_to_label)
        dev_f1 = re_metrics(bundle["dev"]["label"], dev_pred)["positive_macro_f1"]
        curve.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_positive_macro_f1": dev_f1})
        if dev_f1 > best_dev + 1e-4:
            best_dev = dev_f1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - start
    torch.save(model.state_dict(), OUT / "re_bilstm_attention.pt")
    pred = predict_re_lstm(model, test_loader, id_to_label)
    row = flatten_metric_row(
        "BiLSTM + Attention",
        "Deep NLP",
        "test",
        re_metrics(bundle["test"]["label"], pred),
        bootstrap_re_f1(bundle["test"]["label"], pred),
        train_seconds,
        {"embedding_dim": 128, "hidden_dim": 160, "best_dev_positive_macro_f1": best_dev, "epochs": len(curve)},
    )
    return [row], pred, {"curve": curve, "best_dev_positive_macro_f1": best_dev}


def main() -> None:
    seed_everything(42)
    paths = prepare_data()
    ner_bundle, ner_audit = load_ner_bundle(paths)
    re_bundle, re_audit = load_re_bundle(paths)
    write_json(
        OUT / "dataset_audit.json",
        {
            "hashes": paths["hashes"],
            "ner": ner_audit,
            "relation_extraction": re_audit,
            "device": str(DEVICE),
        },
    )

    ner_rows, ner_predictions = fit_token_baselines(ner_bundle)
    lstm_rows, lstm_predictions, ner_curve = fit_ner_bilstm(ner_bundle)
    ner_rows.extend(lstm_rows)
    ner_predictions["BiLSTM"] = lstm_predictions

    re_rows, re_predictions = fit_re_baselines(re_bundle)
    re_lstm_rows, re_lstm_predictions, re_curve = fit_re_bilstm(re_bundle)
    re_rows.extend(re_lstm_rows)
    re_predictions["BiLSTM + Attention"] = re_lstm_predictions

    pd.DataFrame(ner_rows).to_csv(OUT / "ner_metrics.csv", index=False)
    pd.DataFrame(re_rows).to_csv(OUT / "re_metrics.csv", index=False)
    write_json(OUT / "training_curves.json", {"ner_bilstm": ner_curve, "re_bilstm": re_curve})
    with gzip.open(OUT / "predictions.json.gz", "wt", encoding="utf-8") as fp:
        json.dump({"ner": ner_predictions, "re": re_predictions}, fp)
    write_json(OUT / "complete.json", {"status": "success", "ner_models": len(set(x["model"] for x in ner_rows)), "re_models": len(set(x["model"] for x in re_rows))})


if __name__ == "__main__":
    main()
