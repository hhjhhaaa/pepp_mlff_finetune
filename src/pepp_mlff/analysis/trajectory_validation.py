from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PURE_SYSTEM_ORDER = ["PE100", "PP100", "PS100"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run_dirs(batch_dir: Path) -> list[Path]:
    return sorted(path.parent for path in batch_dir.glob("*/*/*/summary.json"))


def scalar_stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"available": False}
    return {
        "available": True,
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def phase_metric_stats(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    phase_rows = [row for row in rows if row.get("phase") == phase]
    metrics = {}
    for key in [
        "temperature_K",
        "energy_potential_eV_per_atom",
        "energy_total_eV",
        "max_force_eV_A",
        "mean_force_eV_A",
        "bond_min_A",
        "bond_max_A",
    ]:
        values = [float(row[key]) for row in phase_rows if key in row and row[key] is not None]
        metrics[key] = scalar_stats(values)
    times = [float(row["time_fs"]) for row in phase_rows if "time_fs" in row]
    metrics["time_window_ps"] = float((max(times) - min(times)) / 1000.0) if len(times) >= 2 else None
    metrics["n_samples"] = len(phase_rows)
    return metrics


def phase_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {phase: phase_metric_stats(rows, phase) for phase in ["stabilization", "production", "nve_audit"]}


def read_first_frame_species(extxyz_path: Path) -> list[str]:
    with extxyz_path.open("r", encoding="utf-8") as handle:
        natoms = int(handle.readline().strip())
        handle.readline()
        return [handle.readline().split()[0] for _ in range(natoms)]


def msd_curve(unwrapped_npz: Path) -> dict[str, Any]:
    data = np.load(unwrapped_npz)
    positions = np.asarray(data["positions_A"], dtype=float)
    times_fs = np.asarray(data["times_fs"], dtype=float)
    if len(positions) < 2:
        return {"available": False, "reason": "fewer_than_two_frames"}
    displacement = positions - positions[0][None, :, :]
    msd = np.mean(np.sum(displacement * displacement, axis=2), axis=1)
    dt_ps = (times_fs - times_fs[0]) / 1000.0
    positive = dt_ps > 0
    apparent_D = None
    fitted_D = None
    if positive.any():
        apparent_D = float((msd[positive][-1] - msd[0]) / (6.0 * dt_ps[positive][-1]))
    if positive.sum() >= 4:
        fit_idx = np.where(positive)[0][max(0, positive.sum() // 2) :]
        slope, _ = np.polyfit(dt_ps[fit_idx], msd[fit_idx], 1)
        fitted_D = float(slope / 6.0)
    return {
        "available": True,
        "n_frames": int(len(positions)),
        "time_window_ps": float(dt_ps[positive][-1]) if positive.any() else 0.0,
        "msd_final_A2": float(msd[positive][-1]) if positive.any() else float(msd[-1]),
        "apparent_D_A2_ps": apparent_D,
        "linear_fit_D_A2_ps": fitted_D,
        "msd_start_A2": float(msd[0]),
    }


def _sample_indices(n: int, max_items: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_items:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(n, size=max_items, replace=False))


def _minimum_image(delta: np.ndarray, cell_lengths: np.ndarray) -> np.ndarray:
    return delta - cell_lengths * np.round(delta / cell_lengths)


def pair_distance_summary(
    unwrapped_npz: Path,
    species: list[str],
    *,
    frame_stride: int = 10,
    max_pairs_per_type_per_frame: int = 250_000,
    seed: int = 20260707,
) -> dict[str, Any]:
    data = np.load(unwrapped_npz)
    positions = np.asarray(data["positions_A"], dtype=float)
    cell = np.asarray(data["cell_A"], dtype=float)
    if cell.ndim == 3:
        cell = cell[0]
    cell_lengths = np.diag(cell).astype(float)
    if np.any(cell_lengths <= 0):
        return {"available": False, "reason": "non_orthorhombic_or_missing_cell"}
    frame_ids = list(range(0, len(positions), max(1, frame_stride)))
    if (len(positions) - 1) not in frame_ids:
        frame_ids.append(len(positions) - 1)
    symbols = np.asarray(species)
    by_symbol = {symbol: np.where(symbols == symbol)[0] for symbol in sorted(set(species))}
    pairs = [("C", "C"), ("C", "H"), ("H", "H")]
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "available": True,
        "frame_count": len(frame_ids),
        "frame_stride": frame_stride,
        "method": "sampled minimum-image pair distance distribution",
    }
    for a, b in pairs:
        ia = by_symbol.get(a, np.array([], dtype=int))
        ib = by_symbol.get(b, np.array([], dtype=int))
        if len(ia) == 0 or len(ib) == 0:
            result[f"{a}-{b}"] = {"available": False}
            continue
        distances: list[np.ndarray] = []
        for frame_id in frame_ids:
            frame = positions[frame_id]
            if a == b:
                tri_i, tri_j = np.triu_indices(len(ia), k=1)
                pair_count = len(tri_i)
                chosen = _sample_indices(pair_count, max_pairs_per_type_per_frame, rng)
                left = ia[tri_i[chosen]]
                right = ia[tri_j[chosen]]
            else:
                pair_count = len(ia) * len(ib)
                chosen = _sample_indices(pair_count, max_pairs_per_type_per_frame, rng)
                left = ia[chosen // len(ib)]
                right = ib[chosen % len(ib)]
            delta = _minimum_image(frame[left] - frame[right], cell_lengths)
            distances.append(np.linalg.norm(delta, axis=1))
        all_distances = np.concatenate(distances) if distances else np.array([], dtype=float)
        stats = scalar_stats(all_distances)
        if stats.get("available"):
            hist, edges = np.histogram(all_distances, bins=200, range=(0.0, float(min(cell_lengths) / 2.0)))
            nonzero = np.where(hist > 0)[0]
            peak = int(nonzero[np.argmax(hist[nonzero])]) if len(nonzero) else None
            shell_volume = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
            rdf_like = np.divide(hist, shell_volume, out=np.zeros_like(shell_volume), where=shell_volume > 0)
            centers = 0.5 * (edges[:-1] + edges[1:])
            first_shell = np.where((centers >= 0.8) & (centers <= 6.0) & (hist > 0))[0]
            rdf_peak = int(first_shell[np.argmax(rdf_like[first_shell])]) if len(first_shell) else None
            stats.update(
                {
                    "q05_A": float(np.quantile(all_distances, 0.05)),
                    "q50_A": float(np.quantile(all_distances, 0.50)),
                    "q95_A": float(np.quantile(all_distances, 0.95)),
                    "raw_histogram_peak_distance_A": float(centers[peak]) if peak is not None else None,
                    "rdf_like_first_peak_distance_A": float(centers[rdf_peak]) if rdf_peak is not None else None,
                    "sampled_pairs": int(len(all_distances)),
                }
            )
        result[f"{a}-{b}"] = stats
    return result


def _dihedral_degrees(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=1)
    good = b1_norm > 0
    b1_unit = np.zeros_like(b1)
    b1_unit[good] = b1[good] / b1_norm[good, None]
    v = b0 - np.sum(b0 * b1_unit, axis=1)[:, None] * b1_unit
    w = b2 - np.sum(b2 * b1_unit, axis=1)[:, None] * b1_unit
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1_unit, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))


def dihedral_summary(
    unwrapped_npz: Path,
    topology: dict[str, Any],
    *,
    frame_stride: int = 10,
    max_dihedrals: int = 2500,
    seed: int = 20260707,
) -> dict[str, Any]:
    dihedrals = np.asarray(topology.get("dihedrals", []), dtype=int)
    if dihedrals.size == 0:
        return {"available": False, "reason": "no_dihedrals"}
    rng = np.random.default_rng(seed)
    chosen = _sample_indices(len(dihedrals), max_dihedrals, rng)
    dihedrals = dihedrals[chosen]
    data = np.load(unwrapped_npz)
    positions = np.asarray(data["positions_A"], dtype=float)
    frame_ids = list(range(0, len(positions), max(1, frame_stride)))
    if (len(positions) - 1) not in frame_ids:
        frame_ids.append(len(positions) - 1)
    angles = []
    for frame_id in frame_ids:
        frame = positions[frame_id]
        idx = dihedrals
        angles.append(_dihedral_degrees(frame[idx[:, 0]], frame[idx[:, 1]], frame[idx[:, 2]], frame[idx[:, 3]]))
    values = np.concatenate(angles) if angles else np.array([], dtype=float)
    abs_values = np.abs(values)
    return {
        "available": True,
        "frame_count": len(frame_ids),
        "sampled_dihedrals": int(len(dihedrals)),
        "sampled_angles": int(len(values)),
        "mean_abs_deg": float(abs_values.mean()),
        "std_deg": float(values.std(ddof=0)),
        "trans_fraction_abs_gt_120": float(np.mean(abs_values > 120.0)),
        "gauche_fraction_30_to_90": float(np.mean((abs_values >= 30.0) & (abs_values <= 90.0))),
        "near_cis_fraction_abs_lt_30": float(np.mean(abs_values < 30.0)),
    }


def phenyl_orientation_summary(
    unwrapped_npz: Path,
    topology: dict[str, Any],
    *,
    frame_stride: int = 10,
) -> dict[str, Any]:
    rings = np.asarray(topology.get("phenyl_rings", []), dtype=int)
    if rings.size == 0:
        return {"available": False, "reason": "no_phenyl_rings"}
    data = np.load(unwrapped_npz)
    positions = np.asarray(data["positions_A"], dtype=float)
    frame_ids = list(range(0, len(positions), max(1, frame_stride)))
    if (len(positions) - 1) not in frame_ids:
        frame_ids.append(len(positions) - 1)
    cosines = []
    z_axis = np.array([0.0, 0.0, 1.0])
    for frame_id in frame_ids:
        frame = positions[frame_id]
        for ring in rings:
            coords = frame[ring]
            centered = coords - coords.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            norm = np.linalg.norm(normal)
            if norm > 0:
                cosines.append(abs(float(np.dot(normal / norm, z_axis))))
    arr = np.asarray(cosines, dtype=float)
    if len(arr) == 0:
        return {"available": False, "reason": "could_not_fit_ring_planes"}
    return {
        "available": True,
        "frame_count": len(frame_ids),
        "n_rings": int(len(rings)),
        "samples": int(len(arr)),
        "mean_abs_cos_to_z": float(arr.mean()),
        "std_abs_cos_to_z": float(arr.std(ddof=0)),
        "nematic_Szz_from_abs_cos": float(np.mean((3.0 * arr * arr - 1.0) / 2.0)),
        "isotropic_reference_mean_abs_cos": 0.5,
    }


def analyze_run(run_dir: Path, *, frame_stride: int = 10) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    quality = load_json(run_dir / "quality_report.json")
    manifest = load_yaml(run_dir / "system_manifest.yaml")
    topology = load_json(run_dir / "topology.json")
    rows = read_jsonl(run_dir / "md_log.jsonl")
    species = read_first_frame_species(run_dir / "input.extxyz")
    target_temperature = float(summary.get("temperature_K") or manifest.get("target_temperature_K") or 0.0)
    prod_temp = phase_metric_stats(rows, "production")["temperature_K"]
    temp_rel_err = None
    if target_temperature and prod_temp.get("available"):
        temp_rel_err = abs(prod_temp["mean"] - target_temperature) / target_temperature
    nve_drift = summary.get("ensemble", {}).get("nve_audit", {}).get("energy_drift_eV_per_atom")
    gates = {
        "quality_ok": bool(quality.get("ok")),
        "production_temperature_within_15_percent": bool(temp_rel_err is not None and temp_rel_err <= 0.15),
        "nve_energy_drift_below_1e-4_eV_per_atom": bool(nve_drift is not None and abs(float(nve_drift)) <= 1e-4),
        "no_quality_failures": not bool(quality.get("failures")),
    }
    return {
        "system_id": summary.get("system_id") or manifest.get("system_id") or run_dir.parents[2].name,
        "components": manifest.get("components", []),
        "run_dir": str(run_dir),
        "n_atoms": int(summary.get("n_atoms", len(species))),
        "species_counts": {symbol: int(species.count(symbol)) for symbol in sorted(set(species))},
        "status": summary.get("status") or quality.get("status"),
        "ok": bool(summary.get("ok")) and bool(quality.get("ok")),
        "target_temperature_K": target_temperature,
        "phase_stats": phase_summary(rows),
        "nve_energy_drift_eV_per_atom": nve_drift,
        "temperature_relative_error": temp_rel_err,
        "gates": gates,
        "msd": msd_curve(run_dir / "trajectory_unwrapped.npz"),
        "pair_distance_distribution": pair_distance_summary(run_dir / "trajectory_unwrapped.npz", species, frame_stride=frame_stride),
        "dihedrals": dihedral_summary(run_dir / "trajectory_unwrapped.npz", topology, frame_stride=frame_stride),
        "phenyl_orientation": phenyl_orientation_summary(run_dir / "trajectory_unwrapped.npz", topology, frame_stride=frame_stride),
    }


def compare_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = {run["system_id"][:5].rstrip("_") for run in runs if run.get("ok")}
    mobility = sorted(
        [
            (run["system_id"], run.get("msd", {}).get("apparent_D_A2_ps"))
            for run in runs
            if run.get("msd", {}).get("apparent_D_A2_ps") is not None
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "all_runs_ok": all(run.get("ok") for run in runs),
        "all_validation_gates_ok": all(all(run.get("gates", {}).values()) for run in runs),
        "completed_pure_system_prefixes": sorted(completed),
        "ready_for_binary_mixtures": all(prefix in completed for prefix in PURE_SYSTEM_ORDER),
        "apparent_mobility_rank_A2_ps": [
            {"system_id": system_id, "apparent_D_A2_ps": value} for system_id, value in mobility
        ],
    }


def analyze_batch(batch_dir: Path, *, frame_stride: int = 10) -> dict[str, Any]:
    runs = [analyze_run(path, frame_stride=frame_stride) for path in run_dirs(batch_dir)]
    return {
        "batch_dir": str(batch_dir),
        "n_runs": len(runs),
        "runs": runs,
        "comparison": compare_runs(runs),
        "interpretation_notes": [
            "Zero-shot MACE trajectories are validation/provisional-label outputs, not AIMD reference labels.",
            "MSD and diffusion values are short-window apparent proxies; they should not be used as final transport properties.",
            "Pair-distance distributions are sampled minimum-image diagnostics, not fully normalized production RDFs.",
        ],
    }


def write_csv(summary: dict[str, Any], output: Path) -> None:
    rows = []
    for run in summary["runs"]:
        prod = run["phase_stats"]["production"]
        nve = run["phase_stats"]["nve_audit"]
        pair_cc = run["pair_distance_distribution"].get("C-C", {})
        rows.append(
            {
                "system_id": run["system_id"],
                "status": run["status"],
                "ok": run["ok"],
                "n_atoms": run["n_atoms"],
                "target_temperature_K": run["target_temperature_K"],
                "production_temperature_mean_K": prod["temperature_K"].get("mean"),
                "production_temperature_std_K": prod["temperature_K"].get("std"),
                "temperature_relative_error": run["temperature_relative_error"],
                "nve_temperature_mean_K": nve["temperature_K"].get("mean"),
                "nve_energy_drift_eV_per_atom": run["nve_energy_drift_eV_per_atom"],
                "production_bond_min_A": prod["bond_min_A"].get("min"),
                "production_bond_max_A": prod["bond_max_A"].get("max"),
                "msd_final_A2": run["msd"].get("msd_final_A2"),
                "apparent_D_A2_ps": run["msd"].get("apparent_D_A2_ps"),
                "linear_fit_D_A2_ps": run["msd"].get("linear_fit_D_A2_ps"),
                "cc_rdf_like_first_peak_A": pair_cc.get("rdf_like_first_peak_distance_A"),
                "dihedral_trans_fraction": run["dihedrals"].get("trans_fraction_abs_gt_120"),
                "dihedral_gauche_fraction": run["dihedrals"].get("gauche_fraction_30_to_90"),
                "phenyl_mean_abs_cos_to_z": run["phenyl_orientation"].get("mean_abs_cos_to_z"),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    def number(value: Any, default: float = float("nan")) -> float:
        return default if value is None else float(value)

    lines = [
        "# MACE-MD Trajectory Validation",
        "",
        f"Batch: `{summary['batch_dir']}`",
        "",
        "## Verdict",
        "",
        f"- Runs analyzed: {summary['n_runs']}",
        f"- All runs ok: `{summary['comparison']['all_runs_ok']}`",
        f"- All validation gates ok: `{summary['comparison']['all_validation_gates_ok']}`",
        f"- Ready for binary mixtures: `{summary['comparison']['ready_for_binary_mixtures']}`",
        "",
        "## Run Comparison",
        "",
        "| system | ok | Tprod mean K | T rel err | NVE drift eV/atom | MSD final A^2 | D apparent A^2/ps | C-C first peak A | dihedral trans | phenyl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        prod = run["phase_stats"]["production"]
        pair_cc = run["pair_distance_distribution"].get("C-C", {})
        phenyl = run["phenyl_orientation"]
        lines.append(
            "| {system} | {ok} | {tmean:.3f} | {terr:.5f} | {drift:.3e} | {msd:.3f} | {diff:.4f} | {ccpeak:.3f} | {trans:.3f} | {phenyl} |".format(
                system=run["system_id"],
                ok=str(run["ok"]),
                tmean=prod["temperature_K"].get("mean", float("nan")),
                terr=run["temperature_relative_error"] if run["temperature_relative_error"] is not None else float("nan"),
                drift=float(run["nve_energy_drift_eV_per_atom"]),
                msd=run["msd"].get("msd_final_A2", float("nan")),
                diff=run["msd"].get("apparent_D_A2_ps", float("nan")),
                ccpeak=number(pair_cc.get("rdf_like_first_peak_distance_A")),
                trans=run["dihedrals"].get("trans_fraction_abs_gt_120", float("nan")),
                phenyl=(
                    f"{phenyl.get('mean_abs_cos_to_z'):.3f}"
                    if phenyl.get("available")
                    else phenyl.get("reason", "n/a")
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Mobility Rank",
            "",
        ]
    )
    for item in summary["comparison"]["apparent_mobility_rank_A2_ps"]:
        lines.append(f"- `{item['system_id']}`: {item['apparent_D_A2_ps']:.4f} A^2/ps")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary["interpretation_notes"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
