#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_mlff.data.build_dataset import build_dataset_from_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse CP2K manifest to extxyz.")
    parser.add_argument("--manifest", default="manifests/cp2k_patches.csv")
    parser.add_argument("--output", default="data/processed/pepp_cp2k_patches.extxyz")
    args = parser.parse_args()
    build_dataset_from_manifest(ROOT / args.manifest, ROOT / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
