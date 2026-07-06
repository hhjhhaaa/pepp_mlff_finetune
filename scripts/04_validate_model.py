#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_mlff.config.load_config import load_yaml_config


def main() -> int:
    config = load_yaml_config(ROOT / "configs/validate/validation.yaml")
    print("Validation interface is reserved. Configured gates:")
    print(config.get("validation_gates", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
