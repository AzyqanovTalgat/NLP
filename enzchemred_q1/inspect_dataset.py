from __future__ import annotations

import hashlib
import json
import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUT = ROOT / "inspection"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

URLS = [
    "https://ftp.expasy.org/databases/rhea/nlp/EnzChemRED.tar.gz",
    "https://zenodo.org/records/11067998/files/EnzChemRED.tar.gz?download=1",
]

archive = CACHE / "EnzChemRED.tar.gz"
if not archive.exists():
    last = None
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, archive.open("wb") as w:
                w.write(r.read())
            if archive.stat().st_size > 100_000:
                break
        except Exception as exc:
            last = exc
            if archive.exists():
                archive.unlink()
    else:
        raise RuntimeError(f"Could not download EnzChemRED: {last}")

extract_dir = CACHE / "EnzChemRED"
if not extract_dir.exists():
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_dir)

xml_files = sorted(extract_dir.rglob("*.xml"))
if not xml_files:
    raise RuntimeError("No XML files found")

ann_types = Counter()
rel_types = Counter()
node_roles = Counter()
passage_types = Counter()
infons = Counter()
ann_count = rel_count = text_chars = 0
sample = None

for path in xml_files:
    root = ET.parse(path).getroot()
    if sample is None:
        sample = {
            "file": path.name,
            "root_tag": root.tag,
            "xml_excerpt": ET.tostring(root, encoding="unicode")[:12000],
        }
    for passage in root.iter("passage"):
        ptype = None
        for infon in passage.findall("infon"):
            infons[f"passage:{infon.attrib.get('key')}={infon.text}"] += 1
            if infon.attrib.get("key") == "type":
                ptype = infon.text
        passage_types[ptype or "UNKNOWN"] += 1
        text_chars += len(passage.findtext("text") or "")
    for ann in root.iter("annotation"):
        ann_count += 1
        atype = None
        for infon in ann.findall("infon"):
            key = infon.attrib.get("key")
            infons[f"annotation:{key}"] += 1
            if key == "type":
                atype = infon.text
        ann_types[atype or "UNKNOWN"] += 1
    for rel in root.iter("relation"):
        rel_count += 1
        rtype = None
        for infon in rel.findall("infon"):
            key = infon.attrib.get("key")
            infons[f"relation:{key}"] += 1
            if key == "type":
                rtype = infon.text
        rel_types[rtype or "UNKNOWN"] += 1
        for node in rel.findall("node"):
            node_roles[node.attrib.get("role", "UNKNOWN")] += 1

sha = hashlib.sha256(archive.read_bytes()).hexdigest()
report = {
    "archive_url_candidates": URLS,
    "archive_size": archive.stat().st_size,
    "archive_sha256": sha,
    "xml_files": len(xml_files),
    "annotation_count": ann_count,
    "relation_count": rel_count,
    "text_characters": text_chars,
    "annotation_types": dict(ann_types),
    "relation_types": dict(rel_types),
    "node_roles": dict(node_roles),
    "passage_types": dict(passage_types),
    "infon_keys": dict(infons),
    "sample": sample,
    "python": os.sys.version,
}
(OUT / "dataset_inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "sample"}, indent=2, ensure_ascii=False))
