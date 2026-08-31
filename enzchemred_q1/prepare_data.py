from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import spacy
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUT = ROOT / "prepared"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

DATA_URLS = [
    "https://ftp.expasy.org/databases/rhea/nlp/EnzChemRED.tar.gz",
    "https://zenodo.org/records/11067998/files/EnzChemRED.tar.gz?download=1",
]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EVAL_ENTITY_TYPES = {"Chemical", "Protein"}
REL_TYPES = {"Conversion", "Indirect_conversion", "Non_conversion"}
REL_PRIORITY = {"None": 0, "Non_conversion": 1, "Indirect_conversion": 2, "Conversion": 3}


def download_archive() -> Path:
    path = CACHE / "EnzChemRED.tar.gz"
    if path.exists() and path.stat().st_size > 100_000:
        return path
    last: Exception | None = None
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 EnzChemRED-Q1/1.0"})
            with urllib.request.urlopen(req, timeout=240) as response, path.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            if path.stat().st_size > 100_000:
                return path
        except Exception as exc:
            last = exc
            if path.exists():
                path.unlink()
    raise RuntimeError(f"Could not download EnzChemRED: {last}")


def extract_archive(path: Path) -> Path:
    target = CACHE / "EnzChemRED"
    marker = target / ".complete"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as tf:
        tf.extractall(target)
    marker.write_text("ok\n", encoding="utf-8")
    return target


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def infon_map(element: ET.Element) -> dict[str, str]:
    return {node.attrib.get("key", ""): (node.text or "") for node in element.findall("infon")}


def parse_document(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    document = root.find("document")
    if document is None:
        raise ValueError(f"No document node in {path}")
    pmid = (document.findtext("id") or path.stem).strip()
    passages: list[dict[str, Any]] = []
    annotation_index: dict[str, dict[str, Any]] = {}
    for p_idx, passage in enumerate(document.findall("passage")):
        p_info = infon_map(passage)
        p_offset = int(passage.findtext("offset") or 0)
        text = passage.findtext("text") or ""
        annotations = []
        for ann in passage.findall("annotation"):
            a_info = infon_map(ann)
            location = ann.find("location")
            if location is None:
                continue
            start = int(location.attrib.get("offset", 0))
            length = int(location.attrib.get("length", 0))
            item = {
                "id": ann.attrib.get("id", ""),
                "type": a_info.get("type", "UNKNOWN"),
                "identifier": a_info.get("identifier", ""),
                "start": start,
                "end": start + length,
                "text": ann.findtext("text") or "",
                "passage_index": p_idx,
            }
            annotations.append(item)
            annotation_index[item["id"]] = item
        relations = []
        for rel in passage.findall("relation"):
            r_info = infon_map(rel)
            nodes = [
                {"refid": node.attrib.get("refid", ""), "role": node.attrib.get("role", "")}
                for node in rel.findall("node")
            ]
            relations.append(
                {
                    "id": rel.attrib.get("id", ""),
                    "type": r_info.get("type", "UNKNOWN"),
                    "nodes": nodes,
                    "passage_index": p_idx,
                }
            )
        passages.append(
            {
                "type": p_info.get("type", "UNKNOWN"),
                "offset": p_offset,
                "text": text,
                "annotations": annotations,
                "relations": relations,
            }
        )
    full_text = "\n".join(p["text"] for p in passages)
    rel_counter = Counter(
        rel["type"] for passage in passages for rel in passage["relations"] if rel["type"] in REL_TYPES
    )
    entity_counter = Counter(
        ann["type"] for passage in passages for ann in passage["annotations"] if ann["type"] in EVAL_ENTITY_TYPES
    )
    return {
        "pmid": pmid,
        "path": path.name,
        "passages": passages,
        "annotation_index": annotation_index,
        "full_text": full_text,
        "normalized_hash": hashlib.sha256(normalize_text(full_text).encode("utf-8")).hexdigest(),
        "relation_counts": dict(rel_counter),
        "entity_counts": dict(entity_counter),
    }


def fetch_years(pmids: list[str]) -> tuple[dict[str, int], dict[str, Any]]:
    cache_path = CACHE / "pubmed_years.json"
    cached: dict[str, int] = {}
    if cache_path.exists():
        cached = {str(k): int(v) for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
    missing = [pmid for pmid in pmids if pmid not in cached]
    errors: list[str] = []
    for start in range(0, len(missing), 100):
        batch = missing[start : start + 100]
        payload = urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "json"}
        ).encode("utf-8")
        request = urllib.request.Request(
            EUTILS,
            data=payload,
            headers={"User-Agent": "EnzChemRED-Q1/1.0 (research@example.org)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            result = data.get("result", {})
            for pmid in batch:
                record = result.get(pmid, {})
                values = [record.get("pubdate", ""), record.get("sortpubdate", "")]
                year = 0
                for value in values:
                    match = re.search(r"\b(19|20)\d{2}\b", str(value))
                    if match:
                        year = int(match.group(0))
                        break
                cached[pmid] = year
        except Exception as exc:
            errors.append(f"batch {start}: {exc}")
            for pmid in batch:
                cached.setdefault(pmid, 0)
        time.sleep(0.4)
    cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True), encoding="utf-8")
    return cached, {
        "requested": len(pmids),
        "missing_before_fetch": len(missing),
        "resolved_years": sum(1 for p in pmids if cached.get(p, 0) > 0),
        "errors": errors,
    }


def safe_strata(documents: list[dict[str, Any]]) -> list[str] | None:
    raw = []
    for doc in documents:
        rel_total = sum(doc["relation_counts"].values())
        indirect = doc["relation_counts"].get("Indirect_conversion", 0) > 0
        nonconv = doc["relation_counts"].get("Non_conversion", 0) > 0
        chem = doc["entity_counts"].get("Chemical", 0)
        raw.append(f"r{min(rel_total, 2)}_i{int(indirect)}_n{int(nonconv)}_c{min(chem // 10, 2)}")
    counts = Counter(raw)
    collapsed = [label if counts[label] >= 4 else "RARE" for label in raw]
    if min(Counter(collapsed).values(), default=0) < 2:
        return None
    return collapsed


def split_documents(documents: list[dict[str, Any]], years: dict[str, int]) -> dict[str, list[str]]:
    rng = random.Random(SEED)
    docs = list(documents)
    for doc in docs:
        doc["year"] = int(years.get(doc["pmid"], 0))
    n_ood = max(110, round(0.10 * len(docs)))
    ranked = sorted(
        docs,
        key=lambda d: (d["year"] if d["year"] > 0 else -1, int(d["pmid"]) if d["pmid"].isdigit() else 0),
        reverse=True,
    )
    ood = ranked[:n_ood]
    remaining = ranked[n_ood:]
    strata = safe_strata(remaining)
    train_part, holdout = train_test_split(
        remaining,
        test_size=0.2222222222,
        random_state=SEED,
        shuffle=True,
        stratify=strata,
    )
    holdout_strata = safe_strata(holdout)
    dev, id_test = train_test_split(
        holdout,
        test_size=0.5,
        random_state=SEED + 1,
        shuffle=True,
        stratify=holdout_strata,
    )
    rng.shuffle(train_part)
    return {
        "train": [d["pmid"] for d in train_part],
        "dev": [d["pmid"] for d in dev],
        "id_test": [d["pmid"] for d in id_test],
        "temporal_ood_test": [d["pmid"] for d in ood],
    }


def add_entity_boundaries(tokens: list[Any], sentence_text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boundaries = {0, len(sentence_text)}
    for tok in tokens:
        boundaries.add(tok.idx)
        boundaries.add(tok.idx + len(tok.text))
    for entity in entities:
        boundaries.add(entity["start"])
        boundaries.add(entity["end"])
    ordered = sorted(x for x in boundaries if 0 <= x <= len(sentence_text))
    pieces = []
    for left, right in zip(ordered, ordered[1:]):
        text = sentence_text[left:right]
        if text and not text.isspace():
            pieces.append({"text": text, "start": left, "end": right})
    return pieces


def label_tokens(tokens: list[dict[str, Any]], entities: list[dict[str, Any]]) -> tuple[list[str], int]:
    labels = ["O"] * len(tokens)
    conflicts = 0
    for entity in sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        indices = [
            i
            for i, token in enumerate(tokens)
            if token["start"] >= entity["start"] and token["end"] <= entity["end"]
        ]
        if not indices:
            continue
        for j, idx in enumerate(indices):
            new_label = ("B-" if j == 0 else "I-") + entity["type"]
            if labels[idx] != "O" and labels[idx] != new_label:
                conflicts += 1
                continue
            labels[idx] = new_label
    return labels, conflicts


def mark_pair(text: str, first: dict[str, Any], second: dict[str, Any]) -> str:
    ordered = sorted([(first, "C1"), (second, "C2")], key=lambda x: x[0]["start"], reverse=True)
    result = text
    for entity, tag in ordered:
        result = (
            result[: entity["start"]]
            + f" <{tag}> "
            + result[entity["start"] : entity["end"]]
            + f" </{tag}> "
            + result[entity["end"] :]
        )
    return re.sub(r"\s+", " ", result).strip()


def build_examples(
    documents: list[dict[str, Any]],
    split_map: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ner_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    re_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit = Counter()
    relation_type_counts = Counter()
    relation_pair_counts = Counter()

    for document in documents:
        pmid = document["pmid"]
        split = split_map[pmid]
        ann_index = document["annotation_index"]
        for p_idx, passage in enumerate(document["passages"]):
            p_text = passage["text"]
            p_offset = passage["offset"]
            spacy_doc = nlp(p_text)
            relation_map: dict[frozenset[str], dict[str, Any]] = {}
            for relation in passage["relations"]:
                r_type = relation["type"]
                if r_type not in REL_TYPES:
                    continue
                chemical_ids = [
                    node["refid"]
                    for node in relation["nodes"]
                    if node["refid"] in ann_index and ann_index[node["refid"]]["type"] == "Chemical"
                ]
                converter_ids = [
                    node["refid"]
                    for node in relation["nodes"]
                    if node.get("role") == "Converter"
                ]
                if len(chemical_ids) < 2:
                    audit["relations_with_fewer_than_two_chemicals"] += 1
                    continue
                for left_id, right_id in itertools.combinations(sorted(set(chemical_ids)), 2):
                    key = frozenset((left_id, right_id))
                    previous = relation_map.get(key)
                    if previous is None or REL_PRIORITY[r_type] > REL_PRIORITY[previous["type"]]:
                        relation_map[key] = {
                            "type": r_type,
                            "converter_ids": converter_ids,
                            "relation_id": relation["id"],
                        }
                relation_type_counts[r_type] += 1

            for s_idx, sentence in enumerate(spacy_doc.sents):
                sent_start = sentence.start_char
                sent_end = sentence.end_char
                sent_text = p_text[sent_start:sent_end]
                entities = []
                for ann in passage["annotations"]:
                    if ann["type"] not in EVAL_ENTITY_TYPES:
                        continue
                    local_start = ann["start"] - p_offset
                    local_end = ann["end"] - p_offset
                    if local_start >= sent_start and local_end <= sent_end:
                        entities.append(
                            {
                                **ann,
                                "start": local_start - sent_start,
                                "end": local_end - sent_start,
                            }
                        )
                    elif local_start < sent_end and local_end > sent_start:
                        audit["entities_crossing_sentence_boundary"] += 1
                spacy_tokens = [tok for tok in sentence if not tok.is_space]
                # Convert token indices from passage-relative to sentence-relative.
                proxy_tokens = []
                for token in spacy_tokens:
                    proxy = type("TokenProxy", (), {})()
                    proxy.idx = token.idx - sent_start
                    proxy.text = token.text
                    proxy_tokens.append(proxy)
                tokens = add_entity_boundaries(proxy_tokens, sent_text, entities)
                labels, conflicts = label_tokens(tokens, entities)
                audit["token_label_conflicts"] += conflicts
                recovered = sum(1 for entity in entities if any(
                    token["start"] == entity["start"] for token in tokens
                ) and any(token["end"] == entity["end"] for token in tokens))
                audit["ner_entities_total"] += len(entities)
                audit["ner_entities_boundary_recoverable"] += recovered
                ner_by_split[split].append(
                    {
                        "uid": f"{pmid}:{p_idx}:{s_idx}",
                        "pmid": pmid,
                        "year": document.get("year", 0),
                        "split": split,
                        "passage_type": passage["type"],
                        "text": sent_text,
                        "tokens": [token["text"] for token in tokens],
                        "token_offsets": [[token["start"], token["end"]] for token in tokens],
                        "labels": labels,
                        "entities": [
                            {
                                "id": entity["id"],
                                "type": entity["type"],
                                "identifier": entity["identifier"],
                                "start": entity["start"],
                                "end": entity["end"],
                                "text": entity["text"],
                            }
                            for entity in entities
                        ],
                    }
                )

                chemicals = sorted((e for e in entities if e["type"] == "Chemical"), key=lambda x: x["start"])
                for first, second in itertools.combinations(chemicals, 2):
                    if first["end"] > second["start"]:
                        audit["overlapping_chemical_pairs_skipped"] += 1
                        continue
                    relation = relation_map.get(frozenset((first["id"], second["id"])))
                    relation_type = relation["type"] if relation else "None"
                    binary_label = int(relation_type in {"Conversion", "Indirect_conversion"})
                    relation_pair_counts[relation_type] += 1
                    re_by_split[split].append(
                        {
                            "uid": f"{pmid}:{p_idx}:{s_idx}:{first['id']}:{second['id']}",
                            "pmid": pmid,
                            "year": document.get("year", 0),
                            "split": split,
                            "sentence": sent_text,
                            "marked_sentence": mark_pair(sent_text, first, second),
                            "entity1": {k: first[k] for k in ["id", "identifier", "start", "end", "text"]},
                            "entity2": {k: second[k] for k in ["id", "identifier", "start", "end", "text"]},
                            "relation_type": relation_type,
                            "binary_label": binary_label,
                            "converter_ids": relation["converter_ids"] if relation else [],
                        }
                    )
    audit["relation_annotations_by_type"] = dict(relation_type_counts)
    audit["candidate_pairs_by_type"] = dict(relation_pair_counts)
    return dict(ner_by_split), dict(re_by_split), dict(audit)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    archive = download_archive()
    extracted = extract_archive(archive)
    xml_files = sorted(extracted.rglob("*.xml"))
    documents = [parse_document(path) for path in xml_files]

    # Remove exact duplicate abstracts before any split.
    unique: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    duplicate_pmids: list[str] = []
    for document in documents:
        if document["normalized_hash"] in seen_hashes:
            duplicate_pmids.append(document["pmid"])
            continue
        seen_hashes.add(document["normalized_hash"])
        unique.append(document)
    documents = unique

    years, year_audit = fetch_years([document["pmid"] for document in documents])
    splits = split_documents(documents, years)
    split_map = {pmid: split for split, pmids in splits.items() for pmid in pmids}
    for document in documents:
        document["year"] = years.get(document["pmid"], 0)

    ner_examples, re_examples, example_audit = build_examples(documents, split_map)

    file_manifest = {}
    for split in ["train", "dev", "id_test", "temporal_ood_test"]:
        file_manifest[f"ner_{split}"] = write_jsonl(OUT / f"ner_{split}.jsonl", ner_examples.get(split, []))
        file_manifest[f"re_{split}"] = write_jsonl(OUT / f"re_{split}.jsonl", re_examples.get(split, []))

    doc_summary = {}
    for split, pmids in splits.items():
        split_docs = [document for document in documents if document["pmid"] in set(pmids)]
        years_split = [document["year"] for document in split_docs if document["year"] > 0]
        doc_summary[split] = {
            "documents": len(split_docs),
            "year_min": min(years_split) if years_split else None,
            "year_median": float(np.median(years_split)) if years_split else None,
            "year_max": max(years_split) if years_split else None,
            "entities": dict(sum((Counter(d["entity_counts"]) for d in split_docs), Counter())),
            "relations": dict(sum((Counter(d["relation_counts"]) for d in split_docs), Counter())),
            "ner_sequences": len(ner_examples.get(split, [])),
            "re_candidates": len(re_examples.get(split, [])),
            "re_positive": sum(x["binary_label"] for x in re_examples.get(split, [])),
        }

    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    audit = {
        "dataset": "EnzChemRED",
        "source_urls": DATA_URLS,
        "license": "CC BY 4.0",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "xml_files_raw": len(xml_files),
        "documents_after_exact_deduplication": len(documents),
        "duplicate_pmids_removed": duplicate_pmids,
        "publication_year_fetch": year_audit,
        "splits": splits,
        "split_summary": doc_summary,
        "example_audit": example_audit,
        "file_manifest": file_manifest,
        "seed": SEED,
        "split_policy": {
            "temporal_ood_test": "latest 10% of documents by PubMed publication year; PMID used as tie-breaker",
            "remaining": "document-level stratified random split into 70% train, 10% dev, 10% ID test",
        },
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(json.dumps({
        "documents": len(documents),
        "archive_sha256": archive_sha,
        "split_summary": doc_summary,
        "example_audit": example_audit,
        "files": file_manifest,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
