from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_benchmark import (
    DATA,
    RESULTS,
    bootstrap_re,
    load_task,
    re_metrics,
    save_json,
    spans_from_labels,
)

ROOT = Path(__file__).resolve().parent
INCOMING = ROOT / "incoming"
FINAL = RESULTS / "final"
FINAL.mkdir(parents=True, exist_ok=True)

THEME_1 = "Автоматическое распознавание химических веществ и ферментов в научных биомедицинских текстах на основе доменно-адаптированных Transformer-моделей"
THEME_2 = "Автоматическое извлечение ферментативных химических превращений между распознанными сущностями и построение графа ферментативных реакций"


def find_one(directory: Path, filename: str) -> Path:
    matches = list(directory.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename} under {directory}, found {matches}")
    return matches[0]


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def merge_prediction_files(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged.update(read_gzip_json(path))
    return merged


def percentage(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{100 * float(value):.2f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, tablefmt="github")


def dataset_summary(audit: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split, values in audit["split_summary"].items():
        rows.append(
            {
                "split": split,
                "documents": values["documents"],
                "year_min": values["year_min"],
                "year_median": values["year_median"],
                "year_max": values["year_max"],
                "chemical_entities": values["entities"].get("Chemical", 0),
                "protein_entities": values["entities"].get("Protein", 0),
                "conversion_relations": values["relations"].get("Conversion", 0),
                "indirect_conversion_relations": values["relations"].get("Indirect_conversion", 0),
                "non_conversion_relations": values["relations"].get("Non_conversion", 0),
                "ner_sequences": values["ner_sequences"],
                "re_candidates": values["re_candidates"],
                "re_positive": values["re_positive"],
                "re_positive_rate": values["re_positive"] / values["re_candidates"] if values["re_candidates"] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_end_to_end(
    ner_rows: dict[str, list[dict[str, Any]]],
    re_rows: dict[str, list[dict[str, Any]]],
    ner_predictions: dict[str, Any],
    re_predictions: dict[str, Any],
    selected_ner: str,
    selected_re: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    results = []
    details: dict[str, Any] = {}
    for split in ["id_test", "temporal_ood_test"]:
        ner_rows_split = ner_rows[split]
        re_rows_split = re_rows[split]
        ner_pred_split = ner_predictions[selected_ner][split]
        re_pred_split = re_predictions[selected_re][split]["predictions"]
        re_score_split = re_predictions[selected_re][split].get("scores", re_pred_split)
        ner_index = {row["uid"]: index for index, row in enumerate(ner_rows_split)}
        predicted_spans = {
            row["uid"]: spans_from_labels(row, ner_pred_split[index])
            for index, row in enumerate(ner_rows_split)
        }
        truth = np.asarray([row["binary_label"] for row in re_rows_split], dtype=int)
        gold_relation_predictions = np.asarray(re_pred_split, dtype=int)
        gold_relation_scores = np.asarray(re_score_split, dtype=float)
        gated_predictions = np.zeros(len(re_rows_split), dtype=int)
        gated_scores = np.zeros(len(re_rows_split), dtype=float)
        both_endpoints = np.zeros(len(re_rows_split), dtype=bool)
        endpoint_1 = np.zeros(len(re_rows_split), dtype=bool)
        endpoint_2 = np.zeros(len(re_rows_split), dtype=bool)
        missing_sentence = 0
        for index, row in enumerate(re_rows_split):
            sentence_uid = ":".join(row["uid"].split(":")[:3])
            if sentence_uid not in ner_index:
                missing_sentence += 1
                continue
            spans = predicted_spans[sentence_uid]
            first = (int(row["entity1"]["start"]), int(row["entity1"]["end"]), "Chemical")
            second = (int(row["entity2"]["start"]), int(row["entity2"]["end"]), "Chemical")
            endpoint_1[index] = first in spans
            endpoint_2[index] = second in spans
            both_endpoints[index] = endpoint_1[index] and endpoint_2[index]
            if both_endpoints[index]:
                gated_predictions[index] = gold_relation_predictions[index]
                gated_scores[index] = gold_relation_scores[index]
        metrics = re_metrics(truth, gated_predictions, gated_scores)
        confidence_interval = bootstrap_re(
            re_rows_split,
            truth,
            gated_predictions,
            iterations=1000,
            seed=2028,
        )
        results.append(
            {
                "ner_model": selected_ner,
                "relation_model": selected_re,
                "split": split,
                "candidate_protocol": "gold candidate universe with predicted-entity exact-span gating",
                **metrics,
                "ci95_low": confidence_interval[0],
                "ci95_high": confidence_interval[1],
                "endpoint1_recall": float(endpoint_1.mean()),
                "endpoint2_recall": float(endpoint_2.mean()),
                "both_endpoint_recall": float(both_endpoints.mean()),
                "positive_pair_both_endpoint_recall": float(both_endpoints[truth == 1].mean()) if (truth == 1).any() else float("nan"),
                "missing_sentence_mappings": missing_sentence,
                "number_of_candidates": len(truth),
                "number_of_positive_relations": int(truth.sum()),
            }
        )
        details[split] = {
            "truth": truth.tolist(),
            "gated_predictions": gated_predictions.tolist(),
            "gated_scores": gated_scores.tolist(),
            "both_endpoints": both_endpoints.astype(int).tolist(),
        }
    return pd.DataFrame(results), details


def display_ner(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source, target in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("exact_f1", "Exact F1"),
        ("macro_type_f1", "Macro type F1"),
        ("chemical_f1", "Chemical F1"),
        ("protein_f1", "Protein F1"),
        ("token_accuracy", "Token accuracy"),
        ("token_macro_f1", "Token Macro-F1"),
    ]:
        result[target] = result[source].map(percentage)
    result["95% CI"] = result.apply(
        lambda row: f"[{percentage(row.get('ci95_low'))}; {percentage(row.get('ci95_high'))}]",
        axis=1,
    )
    return result[
        [
            "model",
            "family",
            "split",
            "Precision",
            "Recall",
            "Exact F1",
            "Macro type F1",
            "Chemical F1",
            "Protein F1",
            "Token accuracy",
            "Token Macro-F1",
            "95% CI",
            "train_seconds",
        ]
    ].rename(
        columns={
            "model": "Модель",
            "family": "Семейство",
            "split": "Разбиение",
            "train_seconds": "Обучение, с",
        }
    )


def display_re(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source, target in [
        ("precision", "Precision+"),
        ("recall", "Recall+"),
        ("positive_f1", "Positive F1"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]:
        result[target] = result[source].map(percentage)
    result["MCC"] = result["mcc"].map(lambda value: f"{float(value):.4f}")
    result["95% CI"] = result.apply(
        lambda row: f"[{percentage(row.get('ci95_low'))}; {percentage(row.get('ci95_high'))}]",
        axis=1,
    )
    columns = [
        "model",
        "family",
        "split",
        "Precision+",
        "Recall+",
        "Positive F1",
        "Accuracy",
        "Balanced accuracy",
        "ROC-AUC",
        "PR-AUC",
        "MCC",
        "95% CI",
        "train_seconds",
    ]
    return result[columns].rename(
        columns={
            "model": "Модель",
            "family": "Семейство",
            "split": "Разбиение",
            "train_seconds": "Обучение, с",
        }
    )


def main() -> None:
    classical = INCOMING / "classical"
    deep_ner = INCOMING / "deep_ner"
    deep_re = INCOMING / "deep_re"
    transformer_ner = INCOMING / "transformer_ner"
    transformer_re = INCOMING / "transformer_re"

    ner_frames = [
        pd.read_csv(find_one(classical, "ner_classical_results.csv")),
        pd.read_csv(find_one(deep_ner, "deep_ner_results.csv")),
        pd.read_csv(find_one(transformer_ner, "transformer_ner_results.csv")),
    ]
    re_frames = [
        pd.read_csv(find_one(classical, "re_classical_results.csv")),
        pd.read_csv(find_one(deep_re, "deep_re_results.csv")),
        pd.read_csv(find_one(transformer_re, "transformer_re_results.csv")),
    ]
    ner_all = pd.concat(ner_frames, ignore_index=True)
    re_all = pd.concat(re_frames, ignore_index=True)
    ner_all.to_csv(FINAL / "ner_comparison.csv", index=False)
    re_all.to_csv(FINAL / "relation_comparison.csv", index=False)

    ner_predictions = merge_prediction_files(
        [
            find_one(classical, "ner_classical_predictions.json.gz"),
            find_one(deep_ner, "deep_ner_predictions.json.gz"),
            find_one(transformer_ner, "transformer_ner_predictions.json.gz"),
        ]
    )
    re_predictions = merge_prediction_files(
        [
            find_one(classical, "re_classical_predictions.json.gz"),
            find_one(deep_re, "deep_re_predictions.json.gz"),
            find_one(transformer_re, "transformer_re_predictions.json.gz"),
        ]
    )

    # Select models once on development data; do not select on either test set.
    selected_ner_row = ner_all.loc[ner_all["split"] == "dev"].sort_values("exact_f1", ascending=False).iloc[0]
    selected_re_row = re_all.loc[re_all["split"] == "dev"].sort_values("positive_f1", ascending=False).iloc[0]
    selected_ner = str(selected_ner_row["model"])
    selected_re = str(selected_re_row["model"])

    ner_rows = load_task("ner")
    re_rows = load_task("re")
    pipeline_frame, pipeline_details = build_end_to_end(
        ner_rows,
        re_rows,
        ner_predictions,
        re_predictions,
        selected_ner,
        selected_re,
    )
    pipeline_frame.to_csv(FINAL / "end_to_end_pipeline.csv", index=False)
    save_json(FINAL / "end_to_end_details.json", pipeline_details)

    audit = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))
    summary_frame = dataset_summary(audit)
    summary_frame.to_csv(FINAL / "dataset_split_summary.csv", index=False)
    shutil.copy2(DATA / "audit.json", FINAL / "dataset_audit.json")
    shutil.copy2(DATA / "splits.json", FINAL / "document_splits.json")

    selected_ner_tests = ner_all.loc[
        (ner_all["model"] == selected_ner) & ner_all["split"].isin(["id_test", "temporal_ood_test"])
    ].set_index("split")
    selected_re_tests = re_all.loc[
        (re_all["model"] == selected_re) & re_all["split"].isin(["id_test", "temporal_ood_test"])
    ].set_index("split")
    pipeline_tests = pipeline_frame.set_index("split")

    decision = {
        "selection_protocol": "best model selected exclusively by development F1",
        "selected_ner_model": selected_ner,
        "selected_relation_model": selected_re,
        "ner": {
            "id_exact_f1": float(selected_ner_tests.loc["id_test", "exact_f1"]),
            "temporal_ood_exact_f1": float(selected_ner_tests.loc["temporal_ood_test", "exact_f1"]),
            "passes_75_both": bool((selected_ner_tests["exact_f1"] >= 0.75).all()),
            "passes_85_both": bool((selected_ner_tests["exact_f1"] >= 0.85).all()),
        },
        "relation_gold_entities": {
            "id_positive_f1": float(selected_re_tests.loc["id_test", "positive_f1"]),
            "temporal_ood_positive_f1": float(selected_re_tests.loc["temporal_ood_test", "positive_f1"]),
            "passes_75_both": bool((selected_re_tests["positive_f1"] >= 0.75).all()),
            "passes_85_both": bool((selected_re_tests["positive_f1"] >= 0.85).all()),
        },
        "sequential_pipeline": {
            "id_positive_f1": float(pipeline_tests.loc["id_test", "positive_f1"]),
            "temporal_ood_positive_f1": float(pipeline_tests.loc["temporal_ood_test", "positive_f1"]),
            "id_positive_pair_endpoint_recall": float(pipeline_tests.loc["id_test", "positive_pair_both_endpoint_recall"]),
            "temporal_ood_positive_pair_endpoint_recall": float(pipeline_tests.loc["temporal_ood_test", "positive_pair_both_endpoint_recall"]),
            "passes_70_both": bool((pipeline_tests["positive_f1"] >= 0.70).all()),
        },
        "same_documents_and_splits_for_both_tasks": True,
        "dataset_structurally_suitable": True,
    }
    decision["overall_empirical_pass_75"] = bool(
        decision["ner"]["passes_75_both"] and decision["relation_gold_entities"]["passes_75_both"]
    )
    save_json(FINAL / "acceptance_decision.json", decision)

    ner_view = display_ner(ner_all)
    re_view = display_re(re_all)
    pipeline_view = pipeline_frame.copy()
    for source, target in [
        ("precision", "Precision+"),
        ("recall", "Recall+"),
        ("positive_f1", "Pipeline F1"),
        ("both_endpoint_recall", "Both endpoints"),
        ("positive_pair_both_endpoint_recall", "Positive-pair endpoints"),
        ("pr_auc", "PR-AUC"),
    ]:
        pipeline_view[target] = pipeline_view[source].map(percentage)
    pipeline_view["95% CI"] = pipeline_view.apply(
        lambda row: f"[{percentage(row['ci95_low'])}; {percentage(row['ci95_high'])}]", axis=1
    )
    pipeline_view = pipeline_view[
        [
            "ner_model",
            "relation_model",
            "split",
            "Precision+",
            "Recall+",
            "Pipeline F1",
            "Both endpoints",
            "Positive-pair endpoints",
            "PR-AUC",
            "95% CI",
        ]
    ].rename(
        columns={
            "ner_model": "NER-модель",
            "relation_model": "RE-модель",
            "split": "Разбиение",
        }
    )

    # Compact publication-style figures.
    for task, frame, value, filename, ylabel in [
        ("NER", ner_all, "exact_f1", "ner_comparison.png", "Exact entity F1"),
        ("RE", re_all, "positive_f1", "relation_comparison.png", "Positive relation F1"),
    ]:
        pivot = frame.pivot_table(index="model", columns="split", values=value, aggfunc="first")
        pivot = pivot[[column for column in ["id_test", "temporal_ood_test"] if column in pivot.columns]]
        axis = pivot.plot(kind="bar", figsize=(12, 6))
        axis.axhline(0.75, linestyle="--", linewidth=1, label="Acceptance threshold 0.75")
        axis.set_ylim(0, 1.0)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("")
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2)
        plt.xticks(rotation=22, ha="right")
        plt.tight_layout()
        plt.savefig(FINAL / filename, dpi=350, bbox_inches="tight")
        plt.close()

    verdict = (
        "ПРИГОДНОСТЬ ПОДТВЕРЖДЕНА ПО ПОРОГУ 0,75"
        if decision["overall_empirical_pass_75"]
        else "ПРИГОДНОСТЬ ПОЛНОСТЬЮ НЕ ПОДТВЕРЖДЕНА ПО ПОРОГУ 0,75"
    )
    split_display = summary_frame.copy()
    split_display["re_positive_rate"] = split_display["re_positive_rate"].map(percentage)

    report = f"""# Полная экспериментальная проверка EnzChemRED для двух последовательных научных статей

## Итоговый вердикт

**{verdict}.** Модели выбирались только по development-разбиению; оба test-набора не использовались для выбора архитектуры или порога.

- Выбранная NER-модель: **{selected_ner}**.
- Выбранная Relation Extraction-модель: **{selected_re}**.
- NER Exact F1: **{percentage(decision['ner']['id_exact_f1'])}** на internal ID test и **{percentage(decision['ner']['temporal_ood_exact_f1'])}** на temporal OOD.
- Relation Positive F1 при gold entities: **{percentage(decision['relation_gold_entities']['id_positive_f1'])}** на ID test и **{percentage(decision['relation_gold_entities']['temporal_ood_positive_f1'])}** на temporal OOD.
- Последовательный pipeline F1: **{percentage(decision['sequential_pipeline']['id_positive_f1'])}** на ID test и **{percentage(decision['sequential_pipeline']['temporal_ood_positive_f1'])}** на temporal OOD.

## Две взаимосвязанные темы

### Статья 1

**{THEME_1}.**

Задача: извлечь из научного текста сущности `Chemical` и `Protein`, их точные символьные границы и тип. Результат первой модели является обязательным входом второй статьи.

### Статья 2

**{THEME_2}.**

Задача: для пар химических сущностей, распознанных первой моделью, определить наличие отношения `Conversion` или `Indirect_conversion`; далее связать реакцию с сущностью `Protein/Converter` и построить граф «фермент — субстрат — продукт».

## Почему задачи действительно последовательны

Обе статьи используют один и тот же EnzChemRED, одинаковые 847/121/121/121 документы и одни и те же document-level разбиения. Полный процесс:

`научный текст → Chemical/Protein NER → пары Chemical → conversion relation → Protein/Converter → reaction knowledge graph`.

## Данные

{markdown_table(split_display)}

- Полный корпус: 1 210 PubMed abstracts.
- Основные сущности: 18 887 Chemical и 13 028 Protein в исходных XML.
- Отношения: 4 512 Conversion, 427 Indirect_conversion и 20 Non_conversion в исходной экспертной разметке.
- Все 31 899 Chemical/Protein-упоминаний, использованные в моделировании, восстановимы по точным символьным границам.
- Temporal OOD выделен по публикационному времени и содержит 121 наиболее новый документ; его публикации относятся преимущественно к 2014–2020 годам.
- SHA-256 исходного архива: `{audit['archive_sha256']}`.

## Результаты NER — три baseline, глубокая модель и Transformer

{markdown_table(ner_view)}

## Результаты Relation Extraction — три baseline, глубокая модель и SOTA-class Transformer

{markdown_table(re_view)}

## Проверка реальной связи между статьями

Relation classifier сначала оценивался с правильными gold-сущностями, а затем через predicted-entity gating: отношение засчитывалось только при точном обнаружении обеих Chemical-сущностей моделью первой статьи.

{markdown_table(pipeline_view)}

Этот показатель является консервативной оценкой распространения NER-ошибок на Relation Extraction. Он использует полный gold candidate universe и penalizes пропущенные сущности, но не создаёт дополнительные пары из ложноположительных NER-сущностей; поэтому в окончательной статье его следует называть `predicted-entity-gated RE`, а не полностью unconstrained end-to-end F1.

## Критерии принятия

- `Exact Entity F1 ≥ 0.75` одновременно на ID и temporal OOD.
- `Positive Relation F1 ≥ 0.75` одновременно на ID и temporal OOD.
- Для последовательного pipeline отдельно контролируется F1 ≥ 0.70.
- Уровень 0.85 рассматривается как сильное, но не обязательное подтверждение.

## Научно-прикладная новизна

Первая статья может развивать гибрид `PubMedBERT + CharCNN + CRF + confidence calibration + temporal-OOD uncertainty`. Вторая — `BioREx/PubMedBERT + entity confidence fusion + reaction-aware graph attention + enzyme–chemical structural priors`. Практический результат — confidence-weighted граф ферментативных реакций, пригодный для поиска биохимических знаний и поддержки curated databases.

## Ограничения

- Temporal OOD является out-of-time holdout внутри одного корпуса, а не независимым внешним корпусом с другой группой аннотаторов.
- Один запуск не заменяет публикационную оценку на 3–5 seeds с mean ± SD.
- Высокая метрика и качественный набор данных не гарантируют принятие статьи журналом Q1; необходимы новая архитектура, ablation, статистическое сравнение и содержательный error analysis.

## Источники данных

- Official EnzChemRED archive: https://ftp.expasy.org/databases/rhea/nlp/EnzChemRED.tar.gz
- Zenodo record: https://zenodo.org/records/11067998
- Corpus repository: https://github.com/ncbi-nlp/EnzChemRED
- Paper: https://www.nature.com/articles/s41597-024-03263-x
- Official pipeline: https://github.com/ncbi/enzchemred
- PubMedBERT: https://huggingface.co/{'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'}
"""
    (FINAL / "FULL_REPORT_RU.md").write_text(report, encoding="utf-8")

    manifest = {
        "status": "success",
        "dataset": "EnzChemRED",
        "themes": [THEME_1, THEME_2],
        "decision": decision,
        "audit_sha256": audit["archive_sha256"],
        "files": sorted(path.name for path in FINAL.iterdir() if path.is_file()),
    }
    save_json(FINAL / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
