from __future__ import annotations

import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "_inspect_data"
DATA.mkdir(parents=True, exist_ok=True)

urls = {
    "blue_bert": "https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/bert_data.zip",
    "blue_data": "https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/data_v0.2.zip",
    "bc5cdr": "https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip",
}

report: dict[str, object] = {"python": os.sys.version, "archives": {}}
for name, url in urls.items():
    dest = DATA / f"{name}.zip"
    urllib.request.urlretrieve(url, dest)
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
    report["archives"][name] = {
        "url": url,
        "size": dest.stat().st_size,
        "members": names,
    }

out = ROOT / "archive_inventory.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
