from __future__ import annotations

import gzip
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from common import (
    ALL_DDI_LABELS,
    BIOMED_MODEL,
    POSITIVE_DDI_LABELS,
    RESULTS,
    bootstrap_re_f1,
    flatten_metric_row,
    load_re_bundle,
    prepare_data,
    re_metrics,
    seed_everything,
    write_json,
)

OUT = RESULTS / "stage_re_sota"
OUT.mkdir(parents=True, exist_ok=True)
LABEL_TO_ID = {label: i for i, label in enumerate(ALL_DDI_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


class EncodedReDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 256):
        enc = tokenizer(
            frame["marked_sentence"].tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.items = []
        for i, label in enumerate(frame["label"]):
            item = {key: value[i] for key, value in enc.items()}
            item["labels"] = LABEL_TO_ID[label]
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class WeightedSequenceTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, focal_gamma: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction="none")
        if self.focal_gamma > 0:
            pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
            ce = ((1.0 - pt) ** self.focal_gamma) * ce
        loss = ce.mean()
        return (loss, outputs) if return_outputs else loss


def decode(logits: np.ndarray) -> list[str]:
    return [ID_TO_LABEL[int(i)] for i in logits.argmax(axis=-1)]


def transformer_metrics(eval_prediction):
    pred = decode(eval_prediction.predictions)
    truth = [ID_TO_LABEL[int(i)] for i in eval_prediction.label_ids]
    return re_metrics(truth, pred)


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    p, r, f, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ALL_DDI_LABELS,
        average=None,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(ALL_DDI_LABELS)
    }


def main() -> None:
    seed_everything(42)
    paths = prepare_data()
    bundle, audit = load_re_bundle(paths)

    tokenizer = AutoTokenizer.from_pretrained(BIOMED_MODEL, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    train_ds = EncodedReDataset(bundle["train"], tokenizer)
    dev_ds = EncodedReDataset(bundle["dev"], tokenizer)
    test_ds = EncodedReDataset(bundle["test"], tokenizer)

    counts = bundle["train"]["label"].value_counts().to_dict()
    total = len(bundle["train"])
    weights = np.array([np.sqrt(total / max(1, counts.get(label, 0))) for label in ALL_DDI_LABELS], dtype=np.float32)
    weights = np.clip(weights / weights.mean(), 0.45, 2.8)
    class_weights = torch.tensor(weights, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        BIOMED_MODEL,
        num_labels=len(ALL_DDI_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=7,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2.5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="positive_macro_f1",
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

    trainer = WeightedSequenceTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=transformer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-4)],
        class_weights=class_weights,
        focal_gamma=1.25,
    )

    started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started
    output = trainer.predict(test_ds, metric_key_prefix="test")
    pred = decode(output.predictions)
    truth = bundle["test"]["label"].tolist()
    metrics = re_metrics(truth, pred)
    ci = bootstrap_re_f1(truth, pred)
    row = flatten_metric_row(
        "BiomedBERT entity-marker classifier",
        "Domain-pretrained SOTA-class Transformer",
        "test",
        metrics,
        ci,
        train_seconds,
        {
            "pretrained_model": BIOMED_MODEL,
            "epochs_requested": 7,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_dev_metric": trainer.state.best_metric,
            "learning_rate": 2.5e-5,
            "effective_batch_size": 16,
            "max_length": 256,
            "focal_gamma": 1.25,
            "class_weights": {label: float(weight) for label, weight in zip(ALL_DDI_LABELS, weights)},
            "special_tokens": SPECIAL_TOKENS,
        },
    )

    pd.DataFrame([row]).to_csv(OUT / "re_metrics.csv", index=False)
    with gzip.open(OUT / "predictions.json.gz", "wt", encoding="utf-8") as fp:
        json.dump({"re": {"BiomedBERT entity-marker classifier": pred}}, fp)
    write_json(OUT / "per_class_metrics.json", per_class_metrics(truth, pred))
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
            "audit_summary": audit,
            "class_counts": counts,
            "class_weights": {label: float(weight) for label, weight in zip(ALL_DDI_LABELS, weights)},
            "per_class_test": per_class_metrics(truth, pred),
        },
    )
    write_json(OUT / "complete.json", {"status": "success", "model": BIOMED_MODEL})


if __name__ == "__main__":
    main()
