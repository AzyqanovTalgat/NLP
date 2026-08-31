from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF

from common_benchmark import (
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

PAD = "<PAD>"
UNK = "<UNK>"


def build_vocabulary(token_sequences, minimum_frequency=1, maximum_size=None):
    counts = Counter(token for sequence in token_sequences for token in sequence)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    vocabulary = {PAD: 0, UNK: 1}
    for token, count in ordered:
        if count < minimum_frequency:
            continue
        if maximum_size is not None and len(vocabulary) >= maximum_size:
            break
        vocabulary[token] = len(vocabulary)
    return vocabulary


class NerDataset(Dataset):
    def __init__(self, rows, word_vocab, char_vocab, max_char_length=40):
        self.rows = rows
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_char_length = max_char_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        words = [self.word_vocab.get(token.lower(), 1) for token in row["tokens"]]
        chars = [
            [self.char_vocab.get(character, 1) for character in token[: self.max_char_length]] or [1]
            for token in row["tokens"]
        ]
        labels = [LABEL_TO_ID[label] for label in row["labels"]]
        return words, chars, labels, index


def collate_ner(batch):
    batch_size = len(batch)
    max_tokens = max(len(item[0]) for item in batch)
    max_chars = max(len(characters) for item in batch for characters in item[1])
    words = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    chars = torch.zeros(batch_size, max_tokens, max_chars, dtype=torch.long)
    labels = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    indices = []
    for row_index, (word_ids, char_ids, label_ids, source_index) in enumerate(batch):
        length = len(word_ids)
        words[row_index, :length] = torch.tensor(word_ids)
        labels[row_index, :length] = torch.tensor(label_ids)
        mask[row_index, :length] = True
        for token_index, character_ids in enumerate(char_ids):
            chars[row_index, token_index, : len(character_ids)] = torch.tensor(character_ids)
        indices.append(source_index)
    return words, chars, labels, mask, indices


class CharBiLstmCrf(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, number_of_labels):
        super().__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, 160, padding_idx=0)
        self.char_embedding = nn.Embedding(char_vocab_size, 32, padding_idx=0)
        self.char_convolution = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.encoder = nn.LSTM(
            224,
            192,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.25,
        )
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(384, number_of_labels)
        self.crf = CRF(number_of_labels, batch_first=True)

    def emissions(self, words, chars):
        word_vectors = self.word_embedding(words)
        batch_size, sequence_length, character_length = chars.shape
        character_vectors = self.char_embedding(chars.view(batch_size * sequence_length, character_length))
        character_vectors = character_vectors.transpose(1, 2)
        character_features = torch.relu(self.char_convolution(character_vectors)).max(dim=-1).values
        character_features = character_features.view(batch_size, sequence_length, -1)
        encoded, _ = self.encoder(torch.cat([word_vectors, character_features], dim=-1))
        return self.classifier(self.dropout(encoded))

    def loss(self, words, chars, labels, mask):
        return -self.crf(self.emissions(words, chars), labels, mask=mask, reduction="mean")

    def decode(self, words, chars, mask):
        return self.crf.decode(self.emissions(words, chars), mask=mask)


def predict_ner(model, loader, rows, device):
    predictions = [None] * len(rows)
    model.eval()
    with torch.no_grad():
        for words, chars, _, mask, indices in loader:
            decoded = model.decode(words.to(device), chars.to(device), mask.to(device))
            for source_index, sequence in zip(indices, decoded):
                predictions[source_index] = [NER_LABELS[label] for label in sequence]
    return predictions


def train_ner() -> None:
    seed_everything(42)
    rows = load_task("ner")
    word_vocab = build_vocabulary(
        [[token.lower() for token in row["tokens"]] for row in rows["train"]],
        minimum_frequency=1,
        maximum_size=60000,
    )
    char_vocab = build_vocabulary(
        [[character for token in row["tokens"] for character in token] for row in rows["train"]],
        minimum_frequency=1,
        maximum_size=500,
    )
    datasets = {split: NerDataset(rows[split], word_vocab, char_vocab) for split in rows}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=48 if split == "train" else 96,
            shuffle=split == "train",
            collate_fn=collate_ner,
            num_workers=2,
        )
        for split, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharBiLstmCrf(len(word_vocab), len(char_vocab), len(NER_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    best_score = -1.0
    best_state = None
    patience = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, 16):
        model.train()
        losses = []
        for words, chars, labels, mask, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(words.to(device), chars.to(device), labels.to(device), mask.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_predictions = predict_ner(model, loaders["dev"], rows["dev"], device)
        dev_metrics = ner_metrics(rows["dev"], dev_predictions)
        scheduler.step(dev_metrics["exact_f1"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **dev_metrics})
        print("NER epoch", epoch, history[-1], flush=True)
        if dev_metrics["exact_f1"] > best_score + 1e-4:
            best_score = dev_metrics["exact_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("NER model did not train")
    model.load_state_dict(best_state)
    model.to(device)
    torch.save(
        {"model_state": best_state, "word_vocab": word_vocab, "char_vocab": char_vocab},
        RESULTS / "deep_ner_model.pt",
    )
    result_rows = []
    prediction_payload = {}
    for split in ["dev", "id_test", "temporal_ood_test"]:
        predictions = predict_ner(model, loaders[split], rows[split], device)
        metrics = ner_metrics(rows[split], predictions)
        confidence_interval = (
            bootstrap_ner(rows[split], predictions, iterations=500)
            if split != "dev"
            else (float("nan"), float("nan"))
        )
        result_rows.append(
            {
                "model": "CharCNN-BiLSTM-CRF",
                "family": "deep sequence model",
                "split": split,
                **metrics,
                "ci95_low": confidence_interval[0],
                "ci95_high": confidence_interval[1],
                "train_seconds": train_seconds,
                "best_dev_f1": best_score,
                "hyperparameters": json.dumps(
                    {"word_embedding": 160, "char_embedding": 32, "char_cnn": 64, "hidden": 192, "layers": 2}
                ),
            }
        )
        prediction_payload[split] = predictions
    pd.DataFrame(result_rows).to_csv(RESULTS / "deep_ner_results.csv", index=False)
    save_gzip_json(RESULTS / "deep_ner_predictions.json.gz", {"CharCNN-BiLSTM-CRF": prediction_payload})
    save_json(RESULTS / "deep_ner_history.json", history)
    save_json(RESULTS / "deep_ner_complete.json", {"status": "success", "device": str(device)})


TOKEN_PATTERN = re.compile(r"</?C[12]>|[A-Za-z0-9]+(?:[-/.][A-Za-z0-9]+)*|[^\w\s]")


def relation_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


class RelationDataset(Dataset):
    def __init__(self, rows, vocabulary, max_length=160):
        self.rows = rows
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = relation_tokens(row["marked_sentence"])[: self.max_length]
        ids = [self.vocabulary.get(token.lower(), 1) for token in tokens]
        lower = [token.lower() for token in tokens]
        c1 = lower.index("<c1>") if "<c1>" in lower else 0
        c2 = lower.index("<c2>") if "<c2>" in lower else min(1, len(ids) - 1)
        return ids, int(row["binary_label"]), c1, c2, index


def collate_relation(batch):
    max_length = max(len(item[0]) for item in batch)
    ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.float)
    c1 = torch.zeros(len(batch), dtype=torch.long)
    c2 = torch.zeros(len(batch), dtype=torch.long)
    indices = []
    for row_index, (sequence, label, first, second, source_index) in enumerate(batch):
        ids[row_index, : len(sequence)] = torch.tensor(sequence)
        mask[row_index, : len(sequence)] = True
        labels[row_index] = label
        c1[row_index] = first
        c2[row_index] = second
        indices.append(source_index)
    return ids, mask, labels, c1, c2, indices


class MarkerBiLstmAttention(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 160, padding_idx=0)
        self.encoder = nn.LSTM(
            160,
            192,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.25,
        )
        self.attention = nn.Linear(384, 1)
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Sequential(
            nn.Linear(384 * 5, 384),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(384, 1),
        )

    def forward(self, ids, mask, c1, c2):
        encoded, _ = self.encoder(self.embedding(ids))
        attention_scores = self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        pooled = torch.bmm(attention_weights.unsqueeze(1), encoded).squeeze(1)
        batch_indices = torch.arange(ids.size(0), device=ids.device)
        first = encoded[batch_indices, c1]
        second = encoded[batch_indices, c2]
        representation = torch.cat([pooled, first, second, torch.abs(first - second), first * second], dim=-1)
        return self.classifier(self.dropout(representation)).squeeze(-1)


def predict_relation(model, loader, number_of_rows, device):
    probabilities = np.zeros(number_of_rows, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for ids, mask, _, c1, c2, indices in loader:
            logits = model(ids.to(device), mask.to(device), c1.to(device), c2.to(device))
            values = torch.sigmoid(logits).cpu().numpy()
            for source_index, value in zip(indices, values):
                probabilities[source_index] = value
    return probabilities


def train_relation() -> None:
    seed_everything(42)
    rows = load_task("re")
    train_sequences = [relation_tokens(row["marked_sentence"].lower()) for row in rows["train"]]
    vocabulary = build_vocabulary(train_sequences, minimum_frequency=2, maximum_size=80000)
    datasets = {split: RelationDataset(rows[split], vocabulary) for split in rows}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=64 if split == "train" else 128,
            shuffle=split == "train",
            collate_fn=collate_relation,
            num_workers=2,
        )
        for split, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarkerBiLstmAttention(len(vocabulary)).to(device)
    positive = sum(row["binary_label"] for row in rows["train"])
    negative = len(rows["train"]) - positive
    positive_weight = torch.tensor([negative / positive], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    best_score = -1.0
    best_threshold = 0.5
    best_state = None
    patience = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, 16):
        model.train()
        losses = []
        for ids, mask, labels, c1, c2, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids.to(device), mask.to(device), c1.to(device), c2.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_scores = predict_relation(model, loaders["dev"], len(rows["dev"]), device)
        dev_truth = np.asarray([row["binary_label"] for row in rows["dev"]])
        threshold, dev_f1 = choose_threshold(dev_truth, dev_scores)
        dev_metrics = re_metrics(dev_truth, dev_scores >= threshold, dev_scores)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "threshold": threshold, **dev_metrics})
        print("RE epoch", epoch, history[-1], flush=True)
        if dev_f1 > best_score + 1e-4:
            best_score = dev_f1
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Relation model did not train")
    model.load_state_dict(best_state)
    model.to(device)
    torch.save({"model_state": best_state, "vocabulary": vocabulary, "threshold": best_threshold}, RESULTS / "deep_re_model.pt")
    result_rows = []
    prediction_payload = {}
    for split in ["dev", "id_test", "temporal_ood_test"]:
        scores = predict_relation(model, loaders[split], len(rows[split]), device)
        truth = np.asarray([row["binary_label"] for row in rows[split]])
        predictions = (scores >= best_threshold).astype(int)
        metrics = re_metrics(truth, predictions, scores)
        confidence_interval = (
            bootstrap_re(rows[split], truth, predictions, iterations=1000)
            if split != "dev"
            else (float("nan"), float("nan"))
        )
        result_rows.append(
            {
                "model": "Marker-BiLSTM-Attention",
                "family": "deep relation model",
                "split": split,
                **metrics,
                "ci95_low": confidence_interval[0],
                "ci95_high": confidence_interval[1],
                "threshold": best_threshold,
                "train_seconds": train_seconds,
                "best_dev_f1": best_score,
                "hyperparameters": json.dumps({"embedding": 160, "hidden": 192, "layers": 2}),
            }
        )
        prediction_payload[split] = {"predictions": predictions.tolist(), "scores": scores.tolist()}
    pd.DataFrame(result_rows).to_csv(RESULTS / "deep_re_results.csv", index=False)
    save_gzip_json(RESULTS / "deep_re_predictions.json.gz", {"Marker-BiLSTM-Attention": prediction_payload})
    save_json(RESULTS / "deep_re_history.json", history)
    save_json(RESULTS / "deep_re_complete.json", {"status": "success", "device": str(device)})


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
