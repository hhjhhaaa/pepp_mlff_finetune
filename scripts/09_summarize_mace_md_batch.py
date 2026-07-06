#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_mlff.analysis.mace_md_summary import summarize_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MACE-MD batch outputs and post-processing targets.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    summary = summarize_batch(batch_dir)
    output = Path(args.output).expanduser().resolve() if args.output else batch_dir / "batch_postprocess_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "n_runs": summary["n_runs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
