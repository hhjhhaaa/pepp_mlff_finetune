#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_mlff.data.split_dataset import DEFAULT_SPLIT_PROFILES, split_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Create manifest split records.")
    parser.add_argument("--manifest", default="manifests/cp2k_patches.csv")
    parser.add_argument("--profile", default="interpolation", choices=sorted(DEFAULT_SPLIT_PROFILES))
    parser.add_argument("--output", default="data/splits/split_manifest.json")
    args = parser.parse_args()
    with (ROOT / args.manifest).open("r", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    splits = split_records(records, DEFAULT_SPLIT_PROFILES[args.profile])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
