from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from common_benchmark import (
    RESULTS,
    SPLITS,
    bootstrap_ner,
    bootstrap_re,
    choose_threshold,
    load_task,
    ner_metrics,
    re_metrics,
    repair_bio,
    save_gzip_json,
    save_json,
    seed_everything,
    split_flat_predictions,
)

OUT = RESULTS / "classical"
OUT.mkdir(parents=True, exist_ok=True)


def shape(token: str) -> str:
    result = []
    for character in token:
        if character.isupper():
            value = "X"
        elif character.islower():
            value = "x"
        elif character.isdigit():
            value = "d"
        else:
            value = character
        if not result or result[-1] != value:
            result.append(value)
    return "".join(result)[:20]


def token_features(tokens: Sequence[str], index: int) -> dict[str, Any]:
    token = tokens[index]
    lower = token.lower()
    features: dict[str, Any] = {
        "bias": 1.0,
        "word.lower": lower,
        "word.shape": shape(token),
        "word.isupper": token.isupper(),
        "word.istitle": token.istitle(),
        "word.isdigit": token.isdigit(),
        "word.hasdigit": any(character.isdigit() for character in token),
        "word.hyphen": "-" in token,
        "word.slash": "/" in token,
        "word.parenthesis": "(" in token or ")" in token,
        "word.length": min(len(token), 24),
        "prefix1": lower[:1],
        "prefix2": lower[:2],
        "prefix3": lower[:3],
        "prefix4": lower[:4],
        "suffix1": lower[-1:],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],
        "suffix4": lower[-4:],
        "suffix5": lower[-5:],
    }
    for offset, name in [(-2, "-2"), (-1, "-1"), (1, "+1"), (2, "+2")]:
        other = index + offset
        if 0 <= other < len(tokens):
            value = tokens[other]
            features[f"{name}.lower"] = value.lower()
            features[f"{name}.shape"] = shape(value)
            features[f"{name}.istitle"] = value.istitle()
            features[f"{name}.isupper"] = value.isupper()
        else:
            features["BOS" if other < 0 else "EOS"] = abs(offset)
    if index > 0:
        features["prev.bigram"] = tokens[index - 1].lower() + "|" + lower
    if index + 1 < len(tokens):
        features["next.bigram"] = lower + "|" + tokens[index + 1].lower()
    return features


def flatten_ner(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    features = []
    labels = []
    for row in rows:
        for index in range(len(row["tokens"])):
            features.append(token_features(row["tokens"], index))
            labels.append(row["labels"][index])
    return features, labels


def build_lexicon(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, ...], str], int]:
    counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for row in rows:
        labels = repair_bio(row["labels"])
        tokens = row["tokens"]
        index = 0
        while index < len(tokens):
            if labels[index].startswith("B-"):
                entity_type = labels[index][2:]
                end = index + 1
                while end < len(tokens) and labels[end] == f"I-{entity_type}":
                    end += 1
                key = tuple(token.lower() for token in tokens[index:end])
                counts[key][entity_type] += 1
                index = end
            else:
                index += 1
    lexicon = {key: value.most_common(1)[0][0] for key, value in counts.items()}
    return lexicon, max(len(key) for key in lexicon)


def lexicon_predict(rows, lexicon, max_length):
    predictions = []
    for row in rows:
        tokens = row["tokens"]
        normalized = [token.lower() for token in tokens]
        labels = ["O"] * len(tokens)
        index = 0
        while index < len(tokens):
            match = None
            for length in range(min(max_length, len(tokens) - index), 0, -1):
                key = tuple(normalized[index : index + length])
                if key in lexicon:
                    match = length, lexicon[key]
                    break
            if match is None:
                index += 1
                continue
            length, entity_type = match
            labels[index] = f"B-{entity_type}"
            for inner in range(index + 1, index + length):
                labels[inner] = f"I-{entity_type}"
            index += length
        predictions.append(labels)
    return predictions


def evaluate_ner_model(name, family, rows_by_split, predictor, training_seconds, hyperparameters):
    output_rows = []
    predictions = {}
    for split in ["dev", "id_test", "temporal_ood_test"]:
        predicted = predictor(split)
        metrics = ner_metrics(rows_by_split[split], predicted)
        confidence_interval = (
            bootstrap_ner(rows_by_split[split], predicted, iterations=500)
            if split != "dev"
            else (float("nan"), float("nan"))
        )
        output_rows.append(
            {
                "model": name,
                "family": family,
                "split": split,
                **metrics,
                "ci95_low": confidence_interval[0],
                "ci95_high": confidence_interval[1],
                "train_seconds": training_seconds,
                "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
            }
        )
        predictions[split] = predicted
    return output_rows, predictions


def transform_relation(word_vectorizer, char_vectorizer, texts):
    return hstack(
        [word_vectorizer.transform(texts), char_vectorizer.transform(texts)],
        format="csr",
    )


def main() -> None:
    seed_everything(42)
    ner = load_task("ner")
    relation = load_task("re")
    all_ner_results = []
    all_ner_predictions = {}

    print("Building NER lexicon", flush=True)
    lexicon, max_length = build_lexicon(ner["train"])
    rows, predictions = evaluate_ner_model(
        "Train-lexicon longest match",
        "dictionary baseline",
        ner,
        lambda split: lexicon_predict(ner[split], lexicon, max_length),
        0.0,
        {"entries": len(lexicon), "maximum_phrase_tokens": max_length},
    )
    all_ner_results.extend(rows)
    all_ner_predictions["Train-lexicon longest match"] = predictions
    joblib.dump(lexicon, OUT / "ner_lexicon.joblib")

    print("Vectorizing all NER tokens", flush=True)
    train_feature_dicts, train_labels = flatten_ner(ner["train"])
    dev_feature_dicts, _ = flatten_ner(ner["dev"])
    vectorizer = DictVectorizer(sparse=True)
    train_features = vectorizer.fit_transform(train_feature_dicts)
    dev_features = vectorizer.transform(dev_feature_dicts)
    joblib.dump(vectorizer, OUT / "ner_vectorizer.joblib")
    del train_feature_dicts, dev_feature_dicts

    configurations = [
        (
            "Logistic Regression (SGD)",
            "classical token classifier",
            "log_loss",
            [3e-6, 1e-5, 3e-5, 1e-4],
        ),
        (
            "Linear SVM (SGD)",
            "classical token classifier",
            "hinge",
            [3e-6, 1e-5, 3e-5, 1e-4],
        ),
    ]
    for name, family, loss, alphas in configurations:
        candidates = []
        for alpha in alphas:
            classifier = SGDClassifier(
                loss=loss,
                alpha=alpha,
                class_weight="balanced",
                max_iter=30,
                tol=1e-3,
                average=True,
                random_state=42,
                n_jobs=-1,
            )
            started = time.perf_counter()
            classifier.fit(train_features, train_labels)
            seconds = time.perf_counter() - started
            dev_flat = classifier.predict(dev_features)
            dev_predictions = split_flat_predictions(ner["dev"], dev_flat)
            score = ner_metrics(ner["dev"], dev_predictions)["exact_f1"]
            print(name, alpha, score, flush=True)
            candidates.append((score, alpha, classifier, seconds))
        _, best_alpha, best_model, seconds = max(candidates, key=lambda item: item[0])
        joblib.dump(best_model, OUT / ("ner_logistic.joblib" if loss == "log_loss" else "ner_svm.joblib"))

        transformed = {"dev": dev_features}
        for split in ["id_test", "temporal_ood_test"]:
            feature_dicts, _ = flatten_ner(ner[split])
            transformed[split] = vectorizer.transform(feature_dicts)
        result_rows, result_predictions = evaluate_ner_model(
            name,
            family,
            ner,
            lambda split, model=best_model: split_flat_predictions(
                ner[split], model.predict(transformed[split])
            ),
            seconds,
            {"loss": loss, "alpha": best_alpha, "epochs_max": 30},
        )
        all_ner_results.extend(result_rows)
        all_ner_predictions[name] = result_predictions

    print("Vectorizing relation instances", flush=True)
    train_texts = [row["marked_sentence"] for row in relation["train"]]
    dev_texts = [row["marked_sentence"] for row in relation["dev"]]
    train_labels_relation = np.asarray([row["binary_label"] for row in relation["train"]])
    dev_labels_relation = np.asarray([row["binary_label"] for row in relation["dev"]])
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=120000,
        token_pattern=r"(?u)\b\w[\w\-/.]*\b|</?C[12]>",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=80000,
        sublinear_tf=True,
    )
    train_word = word_vectorizer.fit_transform(train_texts)
    train_char = char_vectorizer.fit_transform(train_texts)
    train_features_relation = hstack([train_word, train_char], format="csr")
    dev_features_relation = transform_relation(word_vectorizer, char_vectorizer, dev_texts)
    joblib.dump(word_vectorizer, OUT / "re_word_tfidf.joblib")
    joblib.dump(char_vectorizer, OUT / "re_char_tfidf.joblib")

    relation_specs = [
        ("Complement Naive Bayes", [("alpha", value) for value in [0.05, 0.1, 0.25, 0.5, 1.0]]),
        ("Logistic Regression", [("C", value) for value in [0.5, 1.0, 2.0, 4.0, 8.0]]),
        ("Linear SVM", [("C", value) for value in [0.1, 0.25, 0.5, 1.0, 2.0]]),
    ]
    selected_relation_models = []
    for name, grid in relation_specs:
        candidates = []
        for parameter, value in grid:
            if name == "Complement Naive Bayes":
                classifier = ComplementNB(alpha=value)
            elif name == "Logistic Regression":
                classifier = LogisticRegression(
                    C=value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1500,
                    random_state=42,
                )
            else:
                classifier = LinearSVC(
                    C=value,
                    class_weight="balanced",
                    max_iter=15000,
                    random_state=42,
                )
            started = time.perf_counter()
            classifier.fit(train_features_relation, train_labels_relation)
            seconds = time.perf_counter() - started
            if hasattr(classifier, "predict_proba"):
                dev_scores = classifier.predict_proba(dev_features_relation)[:, 1]
            else:
                raw = np.clip(classifier.decision_function(dev_features_relation), -30, 30)
                dev_scores = 1.0 / (1.0 + np.exp(-raw))
            threshold, score = choose_threshold(dev_labels_relation, dev_scores)
            print(name, parameter, value, threshold, score, flush=True)
            candidates.append((score, parameter, value, threshold, classifier, seconds))
        selected_relation_models.append(max(candidates, key=lambda item: item[0]))

    all_relation_results = []
    all_relation_predictions = {}
    for spec, selected in zip(relation_specs, selected_relation_models):
        name = spec[0]
        _, parameter, value, threshold, classifier, seconds = selected
        joblib.dump(classifier, OUT / f"re_{name.lower().replace(' ', '_')}.joblib")
        predictions_by_split = {}
        for split in ["dev", "id_test", "temporal_ood_test"]:
            rows_split = relation[split]
            texts = [row["marked_sentence"] for row in rows_split]
            features = dev_features_relation if split == "dev" else transform_relation(
                word_vectorizer, char_vectorizer, texts
            )
            if hasattr(classifier, "predict_proba"):
                scores = classifier.predict_proba(features)[:, 1]
            else:
                raw = np.clip(classifier.decision_function(features), -30, 30)
                scores = 1.0 / (1.0 + np.exp(-raw))
            predicted = (scores >= threshold).astype(int)
            truth = np.asarray([row["binary_label"] for row in rows_split])
            metrics = re_metrics(truth, predicted, scores)
            confidence_interval = (
                bootstrap_re(rows_split, truth, predicted, iterations=1000)
                if split != "dev"
                else (float("nan"), float("nan"))
            )
            all_relation_results.append(
                {
                    "model": name,
                    "family": "classical relation classifier",
                    "split": split,
                    **metrics,
                    "ci95_low": confidence_interval[0],
                    "ci95_high": confidence_interval[1],
                    "threshold": threshold,
                    "train_seconds": seconds,
                    "hyperparameters": json.dumps({parameter: value}, sort_keys=True),
                }
            )
            predictions_by_split[split] = {
                "predictions": predicted.tolist(),
                "scores": scores.tolist(),
            }
        all_relation_predictions[name] = predictions_by_split

    ner_frame = pd.DataFrame(all_ner_results)
    relation_frame = pd.DataFrame(all_relation_results)
    ner_frame.to_csv(OUT / "ner_classical_results.csv", index=False)
    relation_frame.to_csv(OUT / "re_classical_results.csv", index=False)
    save_gzip_json(OUT / "ner_classical_predictions.json.gz", all_ner_predictions)
    save_gzip_json(OUT / "re_classical_predictions.json.gz", all_relation_predictions)
    summary = {
        "ner_best_id": ner_frame.loc[ner_frame["split"] == "id_test"].sort_values("exact_f1", ascending=False).iloc[0].to_dict(),
        "ner_best_ood": ner_frame.loc[ner_frame["split"] == "temporal_ood_test"].sort_values("exact_f1", ascending=False).iloc[0].to_dict(),
        "re_best_id": relation_frame.loc[relation_frame["split"] == "id_test"].sort_values("positive_f1", ascending=False).iloc[0].to_dict(),
        "re_best_ood": relation_frame.loc[relation_frame["split"] == "temporal_ood_test"].sort_values("positive_f1", ascending=False).iloc[0].to_dict(),
    }
    save_json(OUT / "summary.json", summary)
    save_json(OUT / "complete.json", {"status": "success"})
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
