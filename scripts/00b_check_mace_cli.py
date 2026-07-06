#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_mlff.utils.logging import write_json

REQUIRED_FLAGS = [
    "--foundation_model",
    "--foundation_head",
    "--train_file",
    "--valid_file",
    "--test_file",
    "--device",
    "--default_dtype",
    "--energy_weight",
    "--forces_weight",
    "--E0s",
    "--multiheads_finetuning",
]


def main() -> int:
    exe = shutil.which("mace_run_train")
    report = {"ok": False, "executable": exe, "missing_flags": REQUIRED_FLAGS, "returncode": None}
    if exe is None:
        report["error"] = "mace_run_train not found on PATH"
        write_json(ROOT / "logs/mace_cli_check.json", report)
        print(json.dumps(report, indent=2))
        return 1

    completed = subprocess.run([exe, "--help"], text=True, capture_output=True, check=False)
    help_text = completed.stdout + completed.stderr
    missing = [flag for flag in REQUIRED_FLAGS if flag not in help_text]
    report.update(
        {
            "ok": completed.returncode == 0 and not missing,
            "returncode": completed.returncode,
            "missing_flags": missing,
            "help_excerpt": help_text[:4000],
        }
    )
    write_json(ROOT / "logs/mace_cli_check.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
