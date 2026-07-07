#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_mlff.analysis.trajectory_validation import analyze_batch, write_csv, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and compare completed MACE-MD validation trajectories.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--frame-stride", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    summary = analyze_batch(batch_dir, frame_stride=args.frame_stride)
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else batch_dir / "trajectory_validation_summary.json"
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else batch_dir / "trajectory_validation_table.csv"
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else batch_dir / "trajectory_validation_report.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(summary, output_csv)
    write_markdown(summary, output_md)
    print(
        json.dumps(
            {
                "n_runs": summary["n_runs"],
                "all_runs_ok": summary["comparison"]["all_runs_ok"],
                "all_validation_gates_ok": summary["comparison"]["all_validation_gates_ok"],
                "ready_for_binary_mixtures": summary["comparison"]["ready_for_binary_mixtures"],
                "output_json": str(output_json),
                "output_csv": str(output_csv),
                "output_md": str(output_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

