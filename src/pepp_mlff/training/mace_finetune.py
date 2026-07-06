from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from pepp_mlff.config.load_config import load_yaml_config
from pepp_mlff.models.pretrained_mace import auto_device, environment_provenance
from pepp_mlff.utils.logging import write_json


BOOL_FIELDS = {"ema", "amsgrad", "multiheads_finetuning"}
DIRECT_FIELDS = [
    "name",
    "train_file",
    "valid_file",
    "valid_fraction",
    "test_file",
    "foundation_model",
    "foundation_head",
    "work_dir",
    "energy_key",
    "forces_key",
    "energy_weight",
    "forces_weight",
    "batch_size",
    "max_num_epochs",
    "lr",
    "device",
    "seed",
    "default_dtype",
    "ema_decay",
    "scaling",
]


def _resolve_e0s(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if value.get("mode") == "average_debug":
            return "average"
        e0s_file = value.get("e0s_file")
        return str(e0s_file) if e0s_file else None
    return str(value)


def build_mace_finetune_command(config: dict[str, Any]) -> list[str]:
    """Build a `mace_run_train` command without running training."""
    command = ["mace_run_train"]
    for field in DIRECT_FIELDS:
        value = config.get(field)
        if value is not None:
            if field == "device" and value == "auto":
                value = auto_device()
            command.extend([f"--{field}", str(value)])

    e0s = _resolve_e0s(config.get("E0s"))
    if e0s is not None:
        command.extend(["--E0s", e0s])

    for field in BOOL_FIELDS:
        if field in config:
            command.extend([f"--{field}", str(bool(config[field]))])
    return command


def write_training_provenance(config: dict[str, Any]) -> dict[str, Any]:
    provenance = {
        "foundation_model": config.get("foundation_model"),
        "environment": environment_provenance(
            config.get("foundation_model"),
            device=config.get("device", "auto"),
        ),
        "e0s": config.get("E0s"),
        "energy_weight": config.get("energy_weight"),
        "forces_weight": config.get("forces_weight"),
    }
    output = config.get("provenance_output")
    if output:
        write_json(output, provenance)
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or run a MACE fine-tuning command.")
    parser.add_argument("--config", default="configs/train/mace_finetune.yaml")
    parser.add_argument("--run", action="store_true", help="Run the command instead of dry-run printing.")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    command = build_mace_finetune_command(config)
    write_training_provenance(config)
    print(" ".join(command))
    if args.run:
        if config.get("require_local_foundation_model", True) and not Path(
            config.get("foundation_model", "")
        ).is_file():
            raise FileNotFoundError(
                "Formal fine-tuning requires a local foundation_model checkpoint."
            )
        return subprocess.call(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
