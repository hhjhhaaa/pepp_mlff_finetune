#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_mlff.config.load_config import load_yaml_config
from pepp_mlff.models.pretrained_mace import environment_provenance, load_foundation_calculator, smoke_test_calculator
from pepp_mlff.utils.logging import write_json


def main() -> int:
    config = load_yaml_config(ROOT / "configs/model/mace_mh.yaml")
    allow_fallback = bool(config.get("allow_api_fallback_for_smoke_test", True))
    if config.get("local_checkpoint"):
        config["local_checkpoint"] = str(ROOT / config["local_checkpoint"])
    if config.get("fallback_local_checkpoint"):
        config["fallback_local_checkpoint"] = str(ROOT / config["fallback_local_checkpoint"])
    try:
        calc = load_foundation_calculator(config, allow_api_fallback=allow_fallback)
        result = smoke_test_calculator(calc)
        result["ok"] = True
        result["provenance"] = environment_provenance(config.get("local_checkpoint"), config.get("device"))
        print(result)
        write_json(ROOT / "logs/pretrained_mace_off_check.json", result)
        return 0
    except Exception as exc:
        result = {
            "ok": False,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "suggestion": "Install PyTorch for this machine, then pip install -e '.[mace]'. For formal PE/PP-silica work, place local MACE-MH checkpoints under models/pretrained/.",
        }
        print(result, file=sys.stderr)
        write_json(ROOT / "logs/pretrained_mace_off_check.json", result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
