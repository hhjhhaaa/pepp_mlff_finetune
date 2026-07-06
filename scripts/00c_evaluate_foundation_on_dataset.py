#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_mlff.config.load_config import load_yaml_config
from pepp_mlff.models.pretrained_mace import environment_provenance, load_foundation_calculator
from pepp_mlff.utils.logging import write_json


def _evaluate(calc, dataset_path: Path) -> dict:
    import numpy as np
    from ase.io import read

    images = read(dataset_path, index=":")
    energy_errors = []
    force_errors = []
    for atoms in images:
        ref_energy = atoms.info.get("energy")
        ref_forces = atoms.arrays.get("forces")
        atoms.calc = calc
        pred_energy = atoms.get_potential_energy()
        pred_forces = atoms.get_forces()
        if ref_energy is not None:
            energy_errors.append((pred_energy - float(ref_energy)) / len(atoms))
        if ref_forces is not None:
            force_errors.extend((pred_forces - ref_forces).reshape(-1).tolist())

    energy_errors = np.asarray(energy_errors, dtype=float)
    force_errors = np.asarray(force_errors, dtype=float)
    return {
        "n_frames": len(images),
        "energy_mae_per_atom_eV": float(np.mean(np.abs(energy_errors))) if energy_errors.size else math.nan,
        "energy_rmse_per_atom_eV": float(np.sqrt(np.mean(energy_errors**2))) if energy_errors.size else math.nan,
        "force_mae_eV_A": float(np.mean(np.abs(force_errors))) if force_errors.size else math.nan,
        "force_rmse_eV_A": float(np.sqrt(np.mean(force_errors**2))) if force_errors.size else math.nan,
    }


def main() -> int:
    model_config = load_yaml_config(ROOT / "configs/model/mace_mh.yaml")
    train_config = load_yaml_config(ROOT / "configs/train/mace_finetune.yaml")
    dataset_paths = [train_config.get("valid_file"), train_config.get("test_file")]
    dataset_paths = [ROOT / path for path in dataset_paths if path]

    try:
        checkpoint = ROOT / str(model_config.get("local_checkpoint"))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local foundation checkpoint not found: {checkpoint}")
        model_config["local_checkpoint"] = str(checkpoint)
        if model_config.get("fallback_local_checkpoint"):
            model_config["fallback_local_checkpoint"] = str(ROOT / model_config["fallback_local_checkpoint"])
        calc = load_foundation_calculator(model_config, allow_api_fallback=False)
        results = {}
        for path in dataset_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            results[str(path.relative_to(ROOT))] = _evaluate(calc, path)
        report = {
            "ok": True,
            "results": results,
            "provenance": environment_provenance(checkpoint, model_config.get("device")),
        }
        write_json(ROOT / "logs/foundation_baseline_eval.json", report)
        print(report)
        return 0
    except Exception as exc:
        report = {"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)}
        write_json(ROOT / "logs/foundation_baseline_eval.json", report)
        print(report, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
