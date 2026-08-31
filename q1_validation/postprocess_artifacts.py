from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

NER_SPLITS = ["test_bc5cdr", "test_ncbi", "test_combined"]
NER_ORDER = ["Logistic Regression", "Linear SVM", "Linear-chain CRF", "BiLSTM", "BiomedBERT"]
RE_ORDER = ["Complement Naive Bayes", "Logistic Regression", "Linear SVM", "BiLSTM + Attention", "BiomedBERT"]
POSITIVE_DDI = ["DDI-advise", "DDI-effect", "DDI-int", "DDI-mechanism"]
ALL_DDI = ["DDI-false", *POSITIVE_DDI]


def locate(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {root}")
    if len(matches) > 1:
        matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        return json.load(fp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_percent(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{100 * float(value):.2f}%"


def seconds_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model"].astype(str)
    out["split"] = out["split"].astype(str)
    out["model"] = out["model"].replace({"BiomedBERT entity-marker classifier": "BiomedBERT"})
    for column in out.columns:
        if column not in {"model", "family", "split", "hyperparameters"}:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def ner_table(df: pd.DataFrame, split: str) -> str:
    subset = df[df["split"] == split].copy()
    subset["_order"] = subset["model"].map({m: i for i, m in enumerate(NER_ORDER)}).fillna(99)
    subset = subset.sort_values(["_order", "model"])
    rows = []
    for _, row in subset.iterrows():
        ci = "—"
        if pd.notna(row.get("f1_ci95_low")) and pd.notna(row.get("f1_ci95_high")):
            ci = f"[{as_percent(row['f1_ci95_low'])}; {as_percent(row['f1_ci95_high'])}]"
        rows.append(
            [
                str(row["model"]),
                as_percent(row.get("entity_precision")),
                as_percent(row.get("entity_recall")),
                as_percent(row.get("entity_f1")),
                ci,
                as_percent(row.get("token_macro_f1")),
                as_percent(row.get("token_accuracy")),
                seconds_text(row.get("train_seconds")),
            ]
        )
    return markdown_table(
        ["Model", "Strict P", "Strict R", "Exact-span F1", "95% bootstrap CI", "Token macro-F1", "Token accuracy", "Training"],
        rows,
    )


def re_table(df: pd.DataFrame) -> str:
    subset = df[df["split"] == "test"].copy()
    subset["_order"] = subset["model"].map({m: i for i, m in enumerate(RE_ORDER)}).fillna(99)
    subset = subset.sort_values(["_order", "model"])
    rows = []
    for _, row in subset.iterrows():
        ci = "—"
        if pd.notna(row.get("f1_ci95_low")) and pd.notna(row.get("f1_ci95_high")):
            ci = f"[{as_percent(row['f1_ci95_low'])}; {as_percent(row['f1_ci95_high'])}]"
        rows.append(
            [
                str(row["model"]),
                as_percent(row.get("positive_macro_precision")),
                as_percent(row.get("positive_macro_recall")),
                as_percent(row.get("positive_macro_f1")),
                ci,
                as_percent(row.get("positive_micro_f1")),
                as_percent(row.get("macro_f1_all")),
                as_percent(row.get("weighted_f1_all")),
                as_percent(row.get("accuracy")),
                f"{float(row.get('mcc', np.nan)):.3f}" if pd.notna(row.get("mcc")) else "—",
                seconds_text(row.get("train_seconds")),
            ]
        )
    return markdown_table(
        ["Model", "Positive macro-P", "Positive macro-R", "Official positive macro-F1", "95% bootstrap CI", "Positive micro-F1", "All-class macro-F1", "Weighted F1", "Accuracy", "MCC", "Training"],
        rows,
    )


def split_count_table(audit: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    modeling = audit["ner"]["modeling"]
    for split in ["train", "dev", *NER_SPLITS]:
        item = modeling[split]
        rows.append(["NER", split, str(item["sequences"]), str(item["tokens"]), str(item["entities"]), "—"])
    re_splits = audit["relation_extraction"]["splits"]
    for split in ["train", "dev", "test"]:
        item = re_splits[split]
        rows.append(["Relation extraction", split, str(item["instances"]), "—", "—", as_percent(item["positive_rate"])])
    return markdown_table(["Task", "Split", "Sequences / instances", "Tokens", "Entities", "Positive rate"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ner-sota", type=Path, required=True)
    parser.add_argument("--re-sota", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=str, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    baseline_ner = normalize_metrics(pd.read_csv(locate(args.baseline, "ner_metrics.csv")))
    baseline_re = normalize_metrics(pd.read_csv(locate(args.baseline, "re_metrics.csv")))
    sota_ner = normalize_metrics(pd.read_csv(locate(args.ner_sota, "ner_metrics.csv")))
    sota_re = normalize_metrics(pd.read_csv(locate(args.re_sota, "re_metrics.csv")))
    ner = pd.concat([baseline_ner, sota_ner], ignore_index=True)
    re_df = pd.concat([baseline_re, sota_re], ignore_index=True)

    ner["model_rank"] = ner["model"].map({m: i for i, m in enumerate(NER_ORDER)}).fillna(99)
    re_df["model_rank"] = re_df["model"].map({m: i for i, m in enumerate(RE_ORDER)}).fillna(99)
    ner = ner.sort_values(["split", "model_rank", "model"]).drop(columns="model_rank")
    re_df = re_df.sort_values(["split", "model_rank", "model"]).drop(columns="model_rank")
    ner.to_csv(out / "ner_model_comparison.csv", index=False)
    re_df.to_csv(out / "re_model_comparison.csv", index=False)

    audit = read_json(locate(args.baseline, "dataset_audit.json"))
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    ner_state = read_json(locate(args.ner_sota, "training_state.json"))
    re_state = read_json(locate(args.re_sota, "training_state.json"))
    (out / "ner_training_state.json").write_text(json.dumps(ner_state, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "re_training_state.json").write_text(json.dumps(re_state, indent=2, ensure_ascii=False), encoding="utf-8")

    re_predictions = read_gzip_json(locate(args.re_sota, "predictions.json.gz"))
    y_true = [str(x).replace("DDI-advice", "DDI-advise") for x in re_predictions["truth"]]
    y_pred = [str(x).replace("DDI-advice", "DDI-advise") for x in re_predictions["prediction"]]
    class_report = classification_report(y_true, y_pred, labels=ALL_DDI, output_dict=True, zero_division=0)
    pd.DataFrame(class_report).T.to_csv(out / "re_biomedbert_per_class.csv")
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=ALL_DDI), index=ALL_DDI, columns=ALL_DDI).to_csv(
        out / "re_biomedbert_confusion_matrix.csv"
    )

    sota_ner_rows = ner[ner["model"] == "BiomedBERT"]
    bc5 = float(sota_ner_rows.loc[sota_ner_rows["split"] == "test_bc5cdr", "entity_f1"].iloc[0])
    ncbi = float(sota_ner_rows.loc[sota_ner_rows["split"] == "test_ncbi", "entity_f1"].iloc[0])
    combined = float(sota_ner_rows.loc[sota_ner_rows["split"] == "test_combined", "entity_f1"].iloc[0])
    re_sota_row = re_df[(re_df["model"] == "BiomedBERT") & (re_df["split"] == "test")].iloc[0]
    re_macro = float(re_sota_row["positive_macro_f1"])
    re_micro = float(re_sota_row["positive_micro_f1"])

    decision = {
        "source_workflow_run": args.source_run,
        "criterion": {
            "minimum": 0.75,
            "strong": 0.85,
            "ner_requires_each_external_test": True,
            "re_primary_metric": "positive_macro_f1 excluding DDI-false",
        },
        "ner": {
            "bc5cdr_exact_f1": bc5,
            "ncbi_exact_f1": ncbi,
            "combined_exact_f1": combined,
            "minimum_external_exact_f1": min(bc5, ncbi),
            "passes_75": min(bc5, ncbi) >= 0.75,
            "passes_85": min(bc5, ncbi) >= 0.85,
        },
        "relation_extraction": {
            "ddi2013_positive_macro_f1": re_macro,
            "ddi2013_positive_micro_f1": re_micro,
            "passes_75": re_macro >= 0.75,
            "passes_85": re_macro >= 0.85,
        },
    }
    decision["overall_passes_75"] = decision["ner"]["passes_75"] and decision["relation_extraction"]["passes_75"]
    decision["overall_passes_85"] = decision["ner"]["passes_85"] and decision["relation_extraction"]["passes_85"]
    (out / "acceptance_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    verdict_ner = "PASS" if decision["ner"]["passes_75"] else "FAIL"
    verdict_re = "PASS" if decision["relation_extraction"]["passes_75"] else "FAIL"
    overall = "CONFIRMED" if decision["overall_passes_75"] else "NOT CONFIRMED"

    report = f"""# Full reproducible validation of alternative biomedical datasets

**Source GitHub Actions run:** `{args.source_run}`  
**Decision threshold:** strict/official F1 ≥ 75%; ≥ 85% is treated as strong confirmation.  
**Overall status:** **{overall}**.

## Experimental protocol

- No pilot subsampling: every model is fitted on the complete cleaned training split.
- Hyperparameters and early stopping use only `dev`; official test labels are read only for final evaluation.
- NER uses the combined official BC5CDR-Disease and NCBI-Disease train/dev data after exact-text deduplication and test-overlap removal. It is evaluated separately on both untouched official test corpora and on their union.
- Relation extraction uses the official BLUE preparation of DDIExtraction 2013 with the five-class schema `DDI-false`, `DDI-advise`, `DDI-effect`, `DDI-int`, `DDI-mechanism`.
- NER primary metric is strict exact-span entity F1. Relation extraction primary metric is macro-F1 over the four positive DDI classes, excluding the dominant negative class.
- Every main F1 is accompanied by a 95% bootstrap confidence interval.

## Dataset integrity and split sizes

{split_count_table(audit)}

NER duplicate removal: train={audit['ner']['deduplication']['train']}; dev={audit['ner']['deduplication']['dev']}.  
DDI leakage removal: {audit['relation_extraction']['removed_for_leakage']}.  
Cross-split DDI index overlap after cleaning: {audit['relation_extraction']['cross_split_index_overlap']}.

## Topic 1 — biomedical scientific entity recognition

### BC5CDR-Disease official test

{ner_table(ner, 'test_bc5cdr')}

### NCBI-Disease independent official test

{ner_table(ner, 'test_ncbi')}

### Combined held-out test

{ner_table(ner, 'test_combined')}

**NER gate:** {verdict_ner}. BiomedBERT exact-span F1 = {as_percent(bc5)} on BC5CDR and {as_percent(ncbi)} on NCBI; the conservative minimum is {as_percent(min(bc5, ncbi))}.

## Topic 2 — drug–drug relation extraction

{re_table(re_df)}

**Relation-extraction gate:** {verdict_re}. BiomedBERT positive macro-F1 = {as_percent(re_macro)} and positive micro-F1 = {as_percent(re_micro)} on the untouched DDI2013 test split.

## Acceptance conclusion

- NER ≥75% on each independent test corpus: **{decision['ner']['passes_75']}**.
- NER ≥85% on each independent test corpus: **{decision['ner']['passes_85']}**.
- DDI positive macro-F1 ≥75%: **{decision['relation_extraction']['passes_75']}**.
- DDI positive macro-F1 ≥85%: **{decision['relation_extraction']['passes_85']}**.
- Both topics jointly satisfy the minimum publication gate: **{decision['overall_passes_75']}**.

This decision is based on predictions produced by the workflow, not on metrics copied from publications.
"""
    (out / "FULL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    # Preserve the raw stage CSVs and logs for auditability.
    raw = out / "raw_stage_evidence"
    for label, source in [("baselines", args.baseline), ("ner_sota", args.ner_sota), ("re_sota", args.re_sota)]:
        target = raw / label
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

    manifest = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            manifest.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(out / "sha256_manifest.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()
