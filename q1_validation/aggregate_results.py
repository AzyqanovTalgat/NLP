from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common import (
    ALL_DDI_LABELS,
    BIOMED_MODEL,
    POSITIVE_DDI_LABELS,
    RESULTS,
    load_ner_bundle,
    load_re_bundle,
    paired_bootstrap_pvalue_ner,
    paired_bootstrap_pvalue_re,
    prepare_data,
    write_json,
)

INCOMING = RESULTS / "incoming"
FINAL = RESULTS / "final"
FINAL.mkdir(parents=True, exist_ok=True)


def find_one(directory: Path, filename: str) -> Path:
    matches = list(directory.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} under {directory}, found {matches}")
    return matches[0]


def load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        return json.load(fp)


def pct(value: Any) -> str:
    try:
        return f"{100 * float(value):.2f}%"
    except Exception:
        return "—"


def dataframe_markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, tablefmt="github")


def copy_stage_diagnostics() -> None:
    diagnostics = FINAL / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for stage in [INCOMING / "baselines", INCOMING / "ner_sota", INCOMING / "re_sota"]:
        if not stage.exists():
            continue
        for path in stage.rglob("*"):
            if path.is_file() and path.name not in {"predictions.json.gz"}:
                target = diagnostics / f"{stage.name}_{path.name}"
                shutil.copy2(path, target)


def main() -> None:
    baseline_dir = INCOMING / "baselines"
    ner_sota_dir = INCOMING / "ner_sota"
    re_sota_dir = INCOMING / "re_sota"

    ner_base = pd.read_csv(find_one(baseline_dir, "ner_metrics.csv"))
    re_base = pd.read_csv(find_one(baseline_dir, "re_metrics.csv"))
    ner_sota = pd.read_csv(find_one(ner_sota_dir, "ner_metrics.csv"))
    re_sota = pd.read_csv(find_one(re_sota_dir, "re_metrics.csv"))
    ner_all = pd.concat([ner_base, ner_sota], ignore_index=True)
    re_all = pd.concat([re_base, re_sota], ignore_index=True)
    ner_all.to_csv(FINAL / "ner_model_comparison.csv", index=False)
    re_all.to_csv(FINAL / "re_model_comparison.csv", index=False)

    audit = json.loads(find_one(baseline_dir, "dataset_audit.json").read_text(encoding="utf-8"))
    write_json(FINAL / "dataset_audit.json", audit)

    base_predictions = load_gzip_json(find_one(baseline_dir, "predictions.json.gz"))
    ner_sota_predictions = load_gzip_json(find_one(ner_sota_dir, "predictions.json.gz"))
    re_sota_predictions = load_gzip_json(find_one(re_sota_dir, "predictions.json.gz"))
    predictions = {
        "ner": {**base_predictions["ner"], **ner_sota_predictions["ner"]},
        "re": {**base_predictions["re"], **re_sota_predictions["re"]},
    }

    paths = prepare_data()
    ner_bundle, _ = load_ner_bundle(paths)
    re_bundle, _ = load_re_bundle(paths)

    # Statistical comparison against the strongest non-transformer model.
    ner_combined = ner_all.loc[ner_all["split"] == "test_combined"].copy()
    best_ner_baseline_row = ner_combined.loc[
        ner_combined["family"] != "Domain-pretrained SOTA-class Transformer"
    ].sort_values("entity_f1", ascending=False).iloc[0]
    best_ner_baseline = best_ner_baseline_row["model"]
    ner_sota_name = ner_sota.iloc[0]["model"]
    ner_truth = [x["labels"] for x in ner_bundle["test_combined"]]
    ner_p_value = paired_bootstrap_pvalue_ner(
        ner_truth,
        predictions["ner"][ner_sota_name]["test_combined"],
        predictions["ner"][best_ner_baseline]["test_combined"],
        n_bootstrap=1000,
    )

    best_re_baseline_row = re_all.loc[
        re_all["family"] != "Domain-pretrained SOTA-class Transformer"
    ].sort_values("positive_macro_f1", ascending=False).iloc[0]
    best_re_baseline = best_re_baseline_row["model"]
    re_sota_name = re_sota.iloc[0]["model"]
    re_truth = re_bundle["test"]["label"].tolist()
    re_p_value = paired_bootstrap_pvalue_re(
        re_truth,
        predictions["re"][re_sota_name],
        predictions["re"][best_re_baseline],
        n_bootstrap=2000,
    )

    # Per-class relation metrics and confusion matrices for every model.
    per_class_rows = []
    for model_name, pred in predictions["re"].items():
        p, r, f, support = precision_recall_fscore_support(
            re_truth,
            pred,
            labels=ALL_DDI_LABELS,
            average=None,
            zero_division=0,
        )
        for i, label in enumerate(ALL_DDI_LABELS):
            per_class_rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "precision": p[i],
                    "recall": r[i],
                    "f1": f[i],
                    "support": int(support[i]),
                }
            )
        cm = confusion_matrix(re_truth, pred, labels=ALL_DDI_LABELS)
        pd.DataFrame(cm, index=ALL_DDI_LABELS, columns=ALL_DDI_LABELS).to_csv(
            FINAL / f"re_confusion_{model_name.lower().replace(' ', '_').replace('+', 'plus')}.csv"
        )
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(FINAL / "re_per_class_metrics.csv", index=False)

    # Acceptance gate requested by the user.
    sota_ner_tests = ner_all.loc[
        (ner_all["model"] == ner_sota_name) & ner_all["split"].isin(["test_bc5cdr", "test_ncbi"])
    ]
    min_ner_f1 = float(sota_ner_tests["entity_f1"].min())
    sota_re_f1 = float(re_sota.iloc[0]["positive_macro_f1"])
    decision = {
        "ner": {
            "model": ner_sota_name,
            "minimum_exact_entity_f1_across_two_held_out_corpora": min_ner_f1,
            "passes_75_percent": min_ner_f1 >= 0.75,
            "passes_85_percent": min_ner_f1 >= 0.85,
        },
        "relation_extraction": {
            "model": re_sota_name,
            "official_positive_macro_f1": sota_re_f1,
            "passes_75_percent": sota_re_f1 >= 0.75,
            "passes_85_percent": sota_re_f1 >= 0.85,
        },
        "overall_passes_75_percent": min_ner_f1 >= 0.75 and sota_re_f1 >= 0.75,
    }
    write_json(FINAL / "acceptance_decision.json", decision)

    significance = {
        "ner": {
            "sota": ner_sota_name,
            "best_non_transformer": best_ner_baseline,
            "paired_bootstrap_p_value": ner_p_value,
            "comparison_split": "test_combined",
        },
        "relation_extraction": {
            "sota": re_sota_name,
            "best_non_transformer": best_re_baseline,
            "paired_bootstrap_p_value": re_p_value,
            "comparison_split": "official test",
        },
    }
    write_json(FINAL / "significance_tests.json", significance)

    # Publication-grade compact tables.
    ner_display = ner_all.loc[ner_all["split"].isin(["test_bc5cdr", "test_ncbi", "test_combined"])].copy()
    ner_display["Precision"] = ner_display["entity_precision"].map(pct)
    ner_display["Recall"] = ner_display["entity_recall"].map(pct)
    ner_display["Exact F1"] = ner_display["entity_f1"].map(pct)
    ner_display["95% CI"] = ner_display.apply(
        lambda row: f"[{pct(row.get('f1_ci95_low'))}; {pct(row.get('f1_ci95_high'))}]", axis=1
    )
    ner_table = ner_display[["model", "family", "split", "Precision", "Recall", "Exact F1", "95% CI"]]
    ner_table.columns = ["Model", "Family", "Held-out test", "Precision", "Recall", "Exact F1", "95% CI"]

    re_display = re_all.copy()
    for source, target in [
        ("accuracy", "Accuracy"),
        ("positive_micro_precision", "Micro-P+"),
        ("positive_micro_recall", "Micro-R+"),
        ("positive_micro_f1", "Micro-F1+"),
        ("positive_macro_f1", "Official Macro-F1+"),
        ("macro_f1_all", "Macro-F1 all"),
        ("weighted_f1_all", "Weighted F1"),
    ]:
        re_display[target] = re_display[source].map(pct)
    re_display["95% CI"] = re_display.apply(
        lambda row: f"[{pct(row.get('f1_ci95_low'))}; {pct(row.get('f1_ci95_high'))}]", axis=1
    )
    re_table = re_display[
        [
            "model",
            "family",
            "Accuracy",
            "Micro-P+",
            "Micro-R+",
            "Micro-F1+",
            "Official Macro-F1+",
            "Macro-F1 all",
            "Weighted F1",
            "mcc",
            "95% CI",
        ]
    ].copy()
    re_table.columns = [
        "Model",
        "Family",
        "Accuracy",
        "Micro-P+",
        "Micro-R+",
        "Micro-F1+",
        "Official Macro-F1+",
        "Macro-F1 all",
        "Weighted F1",
        "MCC",
        "95% CI",
    ]
    re_table["MCC"] = re_table["MCC"].map(lambda x: f"{float(x):.4f}")

    # Figures use default Matplotlib palette to remain accessible in monochrome conversion.
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    pivot = ner_all.pivot_table(index="model", columns="split", values="entity_f1", aggfunc="first")
    pivot = pivot[[c for c in ["test_bc5cdr", "test_ncbi", "test_combined"] if c in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Exact entity F1")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.75, linestyle="--", linewidth=1, label="Acceptance threshold 0.75")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FINAL / "ner_model_comparison.png", dpi=350, bbox_inches="tight")
    plt.close()

    ax = re_all.set_index("model")[["positive_macro_f1", "positive_micro_f1", "macro_f1_all"]].plot(
        kind="bar", figsize=(12, 6)
    )
    ax.set_ylabel("F1")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.75, linestyle="--", linewidth=1, label="Acceptance threshold 0.75")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FINAL / "re_model_comparison.png", dpi=350, bbox_inches="tight")
    plt.close()

    ner_modeling = audit["ner"]["modeling"]
    re_splits = audit["relation_extraction"]["splits"]
    verdict_ner = "ПРОЙДЕН" if decision["ner"]["passes_75_percent"] else "НЕ ПРОЙДЕН"
    verdict_re = "ПРОЙДЕН" if decision["relation_extraction"]["passes_75_percent"] else "НЕ ПРОЙДЕН"
    overall = "ОБА НАБОРА ПОДТВЕРЖДЕНЫ" if decision["overall_passes_75_percent"] else "ТРЕБУЕТСЯ ЗАМЕНА ИЛИ ДОПОЛНИТЕЛЬНАЯ МОДЕЛЬ"

    report = f"""# Полная проверка наборов данных для NER и Relation Extraction

## Итоговый вердикт

**Общий результат: {overall}.** Порог пригодности установлен до запуска эксперимента: строгий F1 не ниже 0.75; уровень 0.85 рассматривается как сильное подтверждение.

- NER: **{verdict_ner}**; минимальный Exact Entity F1 модели {ner_sota_name} на двух нетронутых тестовых корпусах — **{pct(min_ner_f1)}**.
- Relation Extraction: **{verdict_re}**; официальный Macro-F1 по четырём положительным типам DDI — **{pct(sota_re_f1)}**.

## Проверенные данные

### Тема 1 — распознавание биомедицинских научных сущностей

Использована согласованная мультикорпусная постановка **BC5CDR-Disease + NCBI Disease**. Обучение и валидация сформированы только из официальных train/development-разбиений; оба официальных test-разбиения оставлены нетронутыми. Перед обучением удалены точные дубликаты и любые предложения, совпадающие с тестовыми.

| Split | Sequences | Tokens | Entities |
|---|---:|---:|---:|
| Train | {ner_modeling['train']['sequences']} | {ner_modeling['train']['tokens']} | {ner_modeling['train']['entities']} |
| Validation | {ner_modeling['dev']['sequences']} | {ner_modeling['dev']['tokens']} | {ner_modeling['dev']['entities']} |
| BC5CDR held-out test | {ner_modeling['test_bc5cdr']['sequences']} | {ner_modeling['test_bc5cdr']['tokens']} | {ner_modeling['test_bc5cdr']['entities']} |
| NCBI held-out external test | {ner_modeling['test_ncbi']['sequences']} | {ner_modeling['test_ncbi']['tokens']} | {ner_modeling['test_ncbi']['entities']} |

### Тема 2 — извлечение отношений

Использован **DDIExtraction 2013 type classification** из официального релиза BLUE. Метрика принятия — Macro-F1 по классам advice/effect/int/mechanism без доминирующего `DDI-false`.

| Split | Instances | Positive | Negative | Positive rate |
|---|---:|---:|---:|---:|
| Train | {re_splits['train']['instances']} | {re_splits['train']['positive_instances']} | {re_splits['train']['negative_instances']} | {pct(re_splits['train']['positive_rate'])} |
| Validation | {re_splits['dev']['instances']} | {re_splits['dev']['positive_instances']} | {re_splits['dev']['negative_instances']} | {pct(re_splits['dev']['positive_rate'])} |
| Held-out test | {re_splits['test']['instances']} | {re_splits['test']['positive_instances']} | {re_splits['test']['negative_instances']} | {pct(re_splits['test']['positive_rate'])} |

## Сравнение моделей: NER

{dataframe_markdown(ner_table)}

## Сравнение моделей: Relation Extraction

{dataframe_markdown(re_table)}

## Статистическая проверка

- NER: {ner_sota_name} против лучшей нетрансформерной модели ({best_ner_baseline}), paired bootstrap p = **{ner_p_value:.6f}**.
- Relation Extraction: {re_sota_name} против лучшей нетрансформерной модели ({best_re_baseline}), paired bootstrap p = **{re_p_value:.6f}**.
- Интервалы 95% рассчитаны непараметрическим bootstrap на неизменённых test-наборах.

## Воспроизводимость

- Random seed: 42.
- Разделение: официальные train/dev/test; test не использовался для настройки.
- SOTA-class backbone: `{BIOMED_MODEL}`.
- Контроль целостности: SHA-256 всех скачанных архивов и файлов сохранён в `dataset_audit.json`.
- Полные гиперпараметры, learning curves, матрицы ошибок и per-class метрики находятся в каталоге `diagnostics` и CSV-файлах.

## Официальные источники

- BLUE Benchmark: https://github.com/ncbi-nlp/BLUE_Benchmark
- BC5CDR corpus: https://github.com/JHnlp/BioCreative-V-CDR-Corpus
- NCBI Disease corpus: https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/
- DDIExtraction 2013 description: https://github.com/ncbi-nlp/BLUE_Benchmark#relation-extraction
- BiomedBERT model: https://huggingface.co/{BIOMED_MODEL}
"""
    (FINAL / "full_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "status": "success",
        "decision": decision,
        "significance": significance,
        "data_hashes": paths["hashes"],
        "model": BIOMED_MODEL,
        "files": sorted(path.name for path in FINAL.iterdir() if path.is_file()),
    }
    write_json(FINAL / "experiment_manifest.json", manifest)
    copy_stage_diagnostics()


if __name__ == "__main__":
    main()
