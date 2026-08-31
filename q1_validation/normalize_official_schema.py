from __future__ import annotations

from pathlib import Path

path = Path(__file__).resolve().parent / "common.py"
source = path.read_text(encoding="utf-8")
updated = source.replace('"DDI-advice"', '"DDI-advise"')
if updated == source and '"DDI-advise"' not in source:
    raise RuntimeError("Expected DDI label constant was not found")
path.write_text(updated, encoding="utf-8")
print("Normalized DDIExtraction label schema to the official `DDI-advise` spelling.")
