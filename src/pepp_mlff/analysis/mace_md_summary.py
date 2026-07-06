from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def production_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("phase") == "production"]


def temperature_stats(rows: list[dict[str, Any]], target_temperature_K: float | None) -> dict[str, Any]:
    values = np.array([float(row["temperature_K"]) for row in production_rows(rows) if "temperature_K" in row], dtype=float)
    if not len(values):
        return {"available": False}
    mean = float(values.mean())
    result = {
        "available": True,
        "mean_K": mean,
        "min_K": float(values.min()),
        "max_K": float(values.max()),
        "n_samples": int(len(values)),
    }
    if target_temperature_K:
        rel = abs(mean - float(target_temperature_K)) / float(target_temperature_K)
        result["relative_error"] = float(rel)
        result["within_15_percent"] = bool(rel <= 0.15)
    return result


def msd_summary(unwrapped_npz: Path) -> dict[str, Any]:
    if not unwrapped_npz.exists():
        return {"available": False}
    data = np.load(unwrapped_npz)
    positions = data["positions_A"]
    times_fs = data["times_fs"]
    if len(positions) < 2 or len(times_fs) < 2:
        return {"available": False, "reason": "fewer_than_two_frames"}
    displacement = positions - positions[0][None, :, :]
    msd = np.mean(np.sum(displacement * displacement, axis=2), axis=1)
    dt_ps = (float(times_fs[-1]) - float(times_fs[0])) / 1000.0
    apparent_D_A2_ps = None
    if dt_ps > 0:
        apparent_D_A2_ps = float((msd[-1] - msd[0]) / (6.0 * dt_ps))
    return {
        "available": True,
        "n_frames": int(len(positions)),
        "time_window_ps": float(dt_ps),
        "msd_final_A2": float(msd[-1]),
        "apparent_D_A2_ps": apparent_D_A2_ps,
        "interpretation": "short-window apparent diffusion proxy; not a final reference transport value",
    }


def run_output_dirs(batch_dir: Path) -> list[Path]:
    return sorted(path.parent for path in batch_dir.glob("*/*/*/summary.json"))


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    quality = load_json(run_dir / "quality_report.json") if (run_dir / "quality_report.json").exists() else {}
    analysis_preview = load_json(run_dir / "analysis_preview.json") if (run_dir / "analysis_preview.json").exists() else {}
    manifest = load_manifest(run_dir / "system_manifest.yaml")
    log_rows = read_jsonl(run_dir / "md_log.jsonl")
    target_temperature = summary.get("temperature_K") or manifest.get("target_temperature_K")
    return {
        "system_id": summary.get("system_id") or manifest.get("system_id") or run_dir.parents[2].name,
        "components": manifest.get("components", []),
        "run_dir": str(run_dir),
        "ok": bool(summary.get("ok")) and bool(quality.get("ok", True)),
        "status": summary.get("status") or quality.get("status"),
        "n_atoms": summary.get("n_atoms") or analysis_preview.get("n_atoms"),
        "temperature_K": target_temperature,
        "model": {
            "model_id": summary.get("model", {}).get("model_id"),
            "selected_head": summary.get("model", {}).get("selected_head"),
            "local_checkpoint": summary.get("model", {}).get("local_checkpoint"),
        },
        "trajectory": {
            "timestep_fs": summary.get("timestep_fs"),
            "production_ps": _phase_ps(log_rows, "production"),
            "trajectory_interval_steps": summary.get("strides", {}).get("trajectory_interval_steps"),
            "short_highres_interval_steps": summary.get("strides", {}).get("short_highres_interval_steps"),
            "short_highres_window_ps": summary.get("strides", {}).get("short_highres_window_ps"),
        },
        "quality": {
            "ok": quality.get("ok"),
            "failures": quality.get("failures", []),
            "temperature": temperature_stats(log_rows, target_temperature),
        },
        "post_processing_targets": {
            "msd": msd_summary(run_dir / "trajectory_unwrapped.npz"),
            "rdf": {"available": False, "reason": "not_implemented_in_summary_scaffold"},
            "dihedral_distribution": {"available": False, "reason": "requires topology-specific postprocessor"},
            "contact_lifetime": {"available": False, "reason": "requires component contact postprocessor"},
            "phenyl_orientation": {"available": False, "reason": "requires PS phenyl-ring trajectory postprocessor"},
        },
        "label_status": "provisional MLFF labels",
    }


def _phase_ps(rows: list[dict[str, Any]], phase: str) -> float | None:
    phase_times = [float(row["time_fs"]) for row in rows if row.get("phase") == phase and "time_fs" in row]
    if len(phase_times) < 2:
        return None
    return float((max(phase_times) - min(phase_times)) / 1000.0)


def summarize_batch(batch_dir: Path) -> dict[str, Any]:
    runs = [summarize_run(run_dir) for run_dir in run_output_dirs(batch_dir)]
    pure_order = ["PE100", "PP100", "PS100"]
    completed_pure = {run["system_id"][:5].rstrip("_") for run in runs if run.get("ok")}
    return {
        "batch_dir": str(batch_dir),
        "n_runs": len(runs),
        "runs": runs,
        "pure_melt_validation": {
            "completed_system_prefixes": sorted(completed_pure),
            "ready_for_binary_mixtures": all(prefix in completed_pure for prefix in pure_order),
            "note": "Mixtures must wait until PE100, PP100, and PS100 pass structural and dynamical gates.",
        },
        "label_policy": "zero-shot outputs are provisional MLFF labels, not reference labels",
    }
