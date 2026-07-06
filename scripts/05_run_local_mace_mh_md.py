#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ase import units
from ase.data import covalent_radii
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS, FIRE

from pepp_mlff.config.load_config import load_yaml_config
from pepp_mlff.models.pretrained_mace import environment_provenance, load_foundation_calculator


class QualityGateError(RuntimeError):
    """Raised when a trajectory should stop immediately."""


@dataclass
class QualityState:
    ok: bool = True
    failures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    high_temp_steps: int = 0
    last_metrics: dict[str, Any] = field(default_factory=dict)

    def fail(self, gate: str, message: str, metrics: dict[str, Any]) -> None:
        self.ok = False
        failure = {"gate": gate, "message": message, "metrics": metrics}
        self.failures.append(failure)
        raise QualityGateError(message)


def resolve_root_path(path: str | Path | None) -> Path | None:
    if path in (None, "", "null"):
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_model_config(path: Path, default_dtype: str | None, fullgraph: bool) -> dict[str, Any]:
    config = load_yaml_config(path)
    for key in ("local_checkpoint", "fallback_local_checkpoint"):
        resolved = resolve_root_path(config.get(key))
        config[key] = str(resolved) if resolved else None
    if default_dtype:
        config["default_dtype"] = default_dtype
    if fullgraph:
        config["fullgraph"] = True
    return config


def model_id_from_config(config: dict[str, Any]) -> str:
    return str(config.get("model_id") or config.get("preferred_version") or "mace_mh").lower().replace("-", "_")


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        "input.extxyz",
        "relaxed.extxyz",
        "short_highres.extxyz",
        "trajectory.extxyz",
        "trajectory_unwrapped.npz",
        "md_log.jsonl",
        "summary.json",
        "quality_report.json",
        "analysis_preview.json",
    ]
    existing = [name for name in generated if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory already contains generated files: {existing}. "
            "Choose a new replica_id/output_dir or pass --overwrite."
        )
    if overwrite:
        for name in generated:
            target = output_dir / name
            if target.exists():
                target.unlink()


def atom_counts(atoms) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


def guess_bonds(atoms, scale: float = 1.25, max_cutoff: float = 2.2) -> list[list[int]]:
    numbers = atoms.get_atomic_numbers()
    radii = np.array([covalent_radii[number] for number in numbers])
    i_idx, j_idx, distances = neighbor_list("ijd", atoms, max_cutoff)
    bonds: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for i, j, distance in zip(i_idx, j_idx, distances):
        if i == j:
            continue
        a, b = sorted((int(i), int(j)))
        if (a, b) in seen:
            continue
        threshold = scale * float(radii[a] + radii[b])
        if distance <= threshold:
            bonds.append([a, b])
            seen.add((a, b))
    return bonds


def write_topology(path: Path, atoms, source: Path | None) -> dict[str, Any]:
    if source:
        shutil.copy2(source, path)
        return json.loads(path.read_text(encoding="utf-8"))

    symbols = atoms.get_chemical_symbols()
    carbon_atoms = [idx for idx, symbol in enumerate(symbols) if symbol == "C"]
    non_carbon_atoms = [idx for idx, symbol in enumerate(symbols) if symbol != "C"]
    topology = {
        "atom_indexing": "0-based",
        "component_id": ["unknown"] * len(atoms),
        "chain_id": [-1] * len(atoms),
        "monomer_id": [-1] * len(atoms),
        "segment_id": [-1] * len(atoms),
        "bonds": guess_bonds(atoms),
        "dihedrals": [],
        "backbone_atoms": carbon_atoms,
        "sidegroup_atoms": non_carbon_atoms,
        "phenyl_rings": [],
        "metadata": {
            "topology_source": "generated_from_geometry_for_smoke_quality_gates",
            "complete_polymer_topology": False,
            "note": (
                "Generated topology is acceptable only for smoke-test gates. "
                "Do not use it as PS production topology or phenyl-ring evidence."
            ),
        },
    }
    path.write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return topology


def write_manifest(
    path: Path,
    atoms,
    source: Path | None,
    args: argparse.Namespace,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    if source:
        shutil.copy2(source, path)
        try:
            import yaml

            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {"source_manifest": str(source)}

    cell_lengths = [float(x) for x in atoms.cell.lengths()]
    manifest = {
        "system_id": args.system_id,
        "components": args.components.split(",") if args.components else ["unknown"],
        "n_chains": None,
        "repeat_units": None,
        "target_density_g_cm3": args.target_density_g_cm3,
        "target_temperature_K": args.temperature_K,
        "box": {
            "type": "periodic" if any(atoms.pbc) else "nonperiodic",
            "cell_lengths_A": cell_lengths,
            "pbc": [bool(x) for x in atoms.pbc],
        },
        "builder": args.builder,
        "builder_seed": args.builder_seed,
        "preprocess_history": [
            {
                "operation": "mace_md_input_copy",
                "source": str(Path(args.input).resolve()),
                "generated_by": Path(__file__).name,
            }
        ],
        "density_source": args.density_source,
        "atom_counts": atom_counts(atoms),
        "model": {
            "model_id": model_config.get("model_id"),
            "selected_head": model_config.get("selected_head") or model_config.get("foundation_head"),
            "local_checkpoint": model_config.get("local_checkpoint"),
        },
    }
    try:
        import yaml

        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    except ImportError:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _nested_get(mapping: dict[str, Any], path: list[str], default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def enforce_builder_policy(manifest: dict[str, Any], topology: dict[str, Any], args: argparse.Namespace) -> None:
    components = [str(component).upper() for component in (args.components.split(",") if args.components else manifest.get("components", []))]
    is_bulk_polymer = bool(components) and all(component in {"PE", "PP", "PS"} for component in components)
    if not is_bulk_polymer or args.allow_non_emc_smoke:
        return
    builder = manifest.get("builder", {})
    builder_used = builder.get("builder_used") if isinstance(builder, dict) else manifest.get("builder")
    emc_success = builder.get("emc_success") if isinstance(builder, dict) else manifest.get("emc_success")
    coordinate_source = builder.get("coordinate_source") if isinstance(builder, dict) else manifest.get("coordinate_source")
    topology_source = builder.get("topology_source") if isinstance(builder, dict) else topology.get("metadata", {}).get("topology_source")
    if not (
        str(builder_used).lower() == "emc"
        and emc_success is True
        and str(coordinate_source).lower() == "emc"
        and str(topology_source).lower() == "emc"
    ):
        raise ValueError(
            "Bulk PE/PP/PS MACE-MD requires EMC-generated coordinates and topology "
            "(builder_used=emc, emc_success=true, coordinate_source=emc, topology_source=emc)."
        )


def bond_lengths(atoms, bonds: list[list[int]]) -> np.ndarray:
    if not bonds:
        return np.array([])
    return np.array([atoms.get_distance(i, j, mic=True) for i, j in bonds], dtype=float)


def min_nonbonded_distance(atoms, bonded_pairs: set[tuple[int, int]], cutoff: float) -> float | None:
    i_idx, j_idx, distances = neighbor_list("ijd", atoms, cutoff)
    best: float | None = None
    for i, j, distance in zip(i_idx, j_idx, distances):
        if i == j:
            continue
        a, b = sorted((int(i), int(j)))
        if (a, b) in bonded_pairs:
            continue
        value = float(distance)
        best = value if best is None else min(best, value)
    return best


def gpu_memory() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda_available": True,
            "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "memory_free_bytes": int(free),
            "memory_total_bytes": int(total),
            "memory_used_fraction": float((total - free) / total),
        }
    except Exception as exc:
        return {"cuda_available": False, "error": str(exc)}


def scalar_metrics(
    atoms,
    phase: str,
    step: int,
    time_fs: float,
    bonds: list[list[int]],
    bonded_pairs: set[tuple[int, int]],
    min_distance_cutoff: float,
) -> dict[str, Any]:
    epot = float(atoms.get_potential_energy())
    ekin = float(atoms.get_kinetic_energy())
    forces = atoms.get_forces()
    force_norms = np.linalg.norm(forces, axis=1)
    metrics = {
        "phase": phase,
        "step": int(step),
        "time_fs": float(time_fs),
        "energy_potential_eV": epot,
        "energy_kinetic_eV": ekin,
        "energy_total_eV": epot + ekin,
        "energy_potential_eV_per_atom": epot / len(atoms),
        "temperature_K": float(atoms.get_temperature()),
        "max_force_eV_A": float(force_norms.max()),
        "mean_force_eV_A": float(force_norms.mean()),
        "min_nonbonded_distance_A": min_nonbonded_distance(atoms, bonded_pairs, min_distance_cutoff),
        "gpu": gpu_memory(),
    }
    if bonds:
        current = bond_lengths(atoms, bonds)
        metrics["bond_min_A"] = float(current.min())
        metrics["bond_max_A"] = float(current.max())
    return metrics


def check_quality(
    state: QualityState,
    atoms,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    reference_bond_lengths: np.ndarray,
    bonds: list[list[int]],
    interval_steps: int,
) -> None:
    values = [
        metrics["energy_potential_eV"],
        metrics["energy_kinetic_eV"],
        metrics["energy_total_eV"],
        metrics["temperature_K"],
        metrics["max_force_eV_A"],
    ]
    if not np.isfinite(values).all() or not np.isfinite(atoms.positions).all():
        state.fail("finite_values", "NaN or inf detected in energy, force, temperature, or coordinates.", metrics)
    if metrics["max_force_eV_A"] > args.abort_max_force_eV_A:
        state.fail("force", f"max_force exceeded {args.abort_max_force_eV_A} eV/A.", metrics)
    min_distance = metrics.get("min_nonbonded_distance_A")
    if min_distance is not None and min_distance < args.min_nonbonded_distance_A:
        state.fail("geometry", f"nonbonded distance below {args.min_nonbonded_distance_A} A.", metrics)
    if metrics["temperature_K"] > 2.0 * args.temperature_K:
        state.high_temp_steps += interval_steps
    else:
        state.high_temp_steps = 0
    if state.high_temp_steps > args.high_temp_abort_steps:
        state.fail("temperature", "instantaneous temperature stayed above 2x target too long.", metrics)
    if bonds and len(reference_bond_lengths):
        current = bond_lengths(atoms, bonds)
        lower = args.bond_lower_ratio * reference_bond_lengths
        upper = args.bond_upper_ratio * reference_bond_lengths
        if np.any(current < lower) or np.any(current > upper):
            state.fail("bond", "topology-bonded distance left allowed ratio window.", metrics)
    if not atoms.cell.rank and any(atoms.pbc):
        state.fail("cell", "periodic structure has invalid cell.", metrics)


class UnwrappedRecorder:
    def __init__(self, atoms):
        scaled = atoms.cell.scaled_positions(atoms.positions)
        self.previous_scaled = np.asarray(scaled, dtype=float)
        self.unwrapped_scaled = np.asarray(scaled, dtype=float)
        self.frames: list[np.ndarray] = []
        self.steps: list[int] = []
        self.times_fs: list[float] = []

    def record(self, atoms, step: int, time_fs: float) -> None:
        scaled = np.asarray(atoms.cell.scaled_positions(atoms.positions), dtype=float)
        delta = scaled - self.previous_scaled
        for axis, periodic in enumerate(atoms.pbc):
            if periodic:
                delta[:, axis] -= np.round(delta[:, axis])
        self.unwrapped_scaled = self.unwrapped_scaled + delta
        self.previous_scaled = scaled
        positions = self.unwrapped_scaled @ np.asarray(atoms.cell)
        self.frames.append(positions.astype(np.float32))
        self.steps.append(int(step))
        self.times_fs.append(float(time_fs))


def append_extxyz(path: Path, atoms) -> None:
    write(path, atoms, format="extxyz", append=path.exists())


def run_langevin_phase(
    atoms,
    phase: str,
    steps: int,
    timestep_fs: float,
    temperature_K: float,
    friction_per_fs: float,
    log_interval: int,
    frame_interval: int,
    log_path: Path,
    quality: QualityState,
    args: argparse.Namespace,
    bonds: list[list[int]],
    bonded_pairs: set[tuple[int, int]],
    reference_bond_lengths: np.ndarray,
    short_highres_path: Path,
    trajectory_path: Path,
    unwrap: UnwrappedRecorder,
    write_long: bool,
    highres_until_fs: float,
    time_offset_fs: float,
) -> float:
    if steps <= 0:
        return time_offset_fs
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction_per_fs / units.fs,
        logfile=None,
    )

    def record() -> None:
        global_step = int(dyn.nsteps)
        phase_time_fs = global_step * timestep_fs
        total_time_fs = time_offset_fs + phase_time_fs
        metrics = scalar_metrics(
            atoms,
            phase,
            global_step,
            total_time_fs,
            bonds,
            bonded_pairs,
            args.min_distance_search_cutoff_A,
        )
        quality.last_metrics = metrics
        check_quality(quality, atoms, metrics, args, reference_bond_lengths, bonds, log_interval)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        if global_step % frame_interval == 0:
            if phase == "stabilization" or total_time_fs <= highres_until_fs:
                append_extxyz(short_highres_path, atoms)
            if write_long:
                append_extxyz(trajectory_path, atoms)
                unwrap.record(atoms, global_step, total_time_fs)
        print(json.dumps(metrics, sort_keys=True), flush=True)

    record()
    dyn.attach(record, interval=log_interval)
    dyn.run(steps)
    return time_offset_fs + steps * timestep_fs


def run_nve_audit(
    atoms,
    steps: int,
    timestep_fs: float,
    log_interval: int,
    log_path: Path,
    quality: QualityState,
    args: argparse.Namespace,
    bonds: list[list[int]],
    bonded_pairs: set[tuple[int, int]],
    reference_bond_lengths: np.ndarray,
    time_offset_fs: float,
) -> dict[str, Any]:
    if steps <= 0:
        return {"enabled": False}
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs, logfile=None)
    first_energy: float | None = None
    last_energy: float | None = None

    def record() -> None:
        nonlocal first_energy, last_energy
        step = int(dyn.nsteps)
        total_time_fs = time_offset_fs + step * timestep_fs
        metrics = scalar_metrics(
            atoms,
            "nve_audit",
            step,
            total_time_fs,
            bonds,
            bonded_pairs,
            args.min_distance_search_cutoff_A,
        )
        if first_energy is None:
            first_energy = metrics["energy_total_eV"]
        last_energy = metrics["energy_total_eV"]
        quality.last_metrics = metrics
        check_quality(quality, atoms, metrics, args, reference_bond_lengths, bonds, log_interval)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True), flush=True)

    record()
    dyn.attach(record, interval=log_interval)
    dyn.run(steps)
    drift = None if first_energy is None or last_energy is None else (last_energy - first_energy) / len(atoms)
    return {"enabled": True, "steps": steps, "energy_drift_eV_per_atom": drift}


def ps_topology_is_complete(topology: dict[str, Any], components: list[str]) -> bool:
    has_ps = any(component.strip().upper() == "PS" for component in components)
    if not has_ps:
        return True
    return bool(topology.get("phenyl_rings")) and bool(topology.get("dihedrals"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local MACE-MH relax/NVT/NVE trajectories with gates.")
    parser.add_argument("--input", required=True, help="Input structure readable by ASE.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-id", default="batch_0a_driver_smoke")
    parser.add_argument("--system-id", default="unknown_system")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--replica-id", default="replica_0001")
    parser.add_argument("--model-config", default="configs/model/mace_mh0.yaml")
    parser.add_argument("--topology", default=None)
    parser.add_argument("--system-manifest", default=None)
    parser.add_argument("--components", default=None)
    parser.add_argument("--builder", default="unknown")
    parser.add_argument("--builder-seed", type=int, default=None)
    parser.add_argument("--density-source", default="unknown")
    parser.add_argument("--target-density-g-cm3", type=float, default=None)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-dtype", default=None, choices=["float32", "float64", None])
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--relax-fmax", type=float, default=0.5)
    parser.add_argument("--production-relax-fmax", type=float, default=0.2)
    parser.add_argument("--fire-steps", type=int, default=200)
    parser.add_argument("--bfgs-steps", type=int, default=0)
    parser.add_argument("--stabilization-ps", type=float, default=1.0)
    parser.add_argument("--production-ps", type=float, default=2.0)
    parser.add_argument("--nve-audit-ps", type=float, default=0.0)
    parser.add_argument("--friction-per-fs", type=float, default=0.02)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--highres-interval", type=int, default=20)
    parser.add_argument("--trajectory-interval", type=int, default=100)
    parser.add_argument("--short-highres-window-ps", type=float, default=5.0)
    parser.add_argument("--abort-max-force-eV-A", type=float, default=200.0)
    parser.add_argument("--min-nonbonded-distance-A", type=float, default=0.45)
    parser.add_argument("--min-distance-search-cutoff-A", type=float, default=1.2)
    parser.add_argument("--high-temp-abort-steps", type=int, default=100)
    parser.add_argument("--bond-lower-ratio", type=float, default=0.65)
    parser.add_argument("--bond-upper-ratio", type=float, default=1.45)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-non-emc-smoke",
        action="store_true",
        help="Allow non-EMC inputs only for explicit interface/driver smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_config_path = resolve_root_path(args.model_config)
    if model_config_path is None:
        raise ValueError("--model-config is required")
    model_config = load_model_config(model_config_path, args.default_dtype, args.fullgraph)
    model_id = args.model_id or model_id_from_config(model_config)
    output_dir = (
        resolve_root_path(args.output_dir)
        if args.output_dir
        else ROOT / "runs" / "mace_md" / args.batch_id / args.system_id / model_id / args.replica_id
    )
    if output_dir is None:
        raise ValueError("Could not resolve output directory")
    ensure_output_dir(output_dir, args.overwrite)

    input_path = resolve_root_path(args.input)
    if input_path is None or not input_path.is_file():
        raise FileNotFoundError(f"Input structure not found: {args.input}")

    atoms = read(input_path)
    write(output_dir / "input.extxyz", atoms, format="extxyz")

    topology = write_topology(output_dir / "topology.json", atoms, resolve_root_path(args.topology))
    manifest = write_manifest(
        output_dir / "system_manifest.yaml",
        atoms,
        resolve_root_path(args.system_manifest),
        args,
        model_config,
    )

    enforce_builder_policy(manifest, topology, args)
    components = args.components.split(",") if args.components else manifest.get("components", [])
    if not ps_topology_is_complete(topology, components):
        raise ValueError("PS production/audit run requires topology.json with phenyl_rings and dihedrals.")

    selected_head = model_config.get("selected_head") or model_config.get("foundation_head")
    calc = load_foundation_calculator(model_config, allow_api_fallback=False)
    atoms.calc = calc

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    bonds = [[int(i), int(j)] for i, j in topology.get("bonds", [])]
    bonded_pairs = {tuple(sorted((i, j))) for i, j in bonds}
    reference_bond_lengths = bond_lengths(atoms, bonds)
    quality = QualityState()
    start = time.time()

    log_path = output_dir / "md_log.jsonl"
    short_highres_path = output_dir / "short_highres.extxyz"
    trajectory_path = output_dir / "trajectory.extxyz"
    unwrap = UnwrappedRecorder(atoms)
    nve_audit: dict[str, Any] = {"enabled": False}
    status = "ok"

    try:
        fire_log = output_dir / "relax_fire.log"
        fire = FIRE(atoms, logfile=str(fire_log))
        fire.run(fmax=args.relax_fmax, steps=args.fire_steps)
        relax_forces = atoms.get_forces()
        relax_fmax = float(np.linalg.norm(relax_forces, axis=1).max())
        if args.bfgs_steps > 0 and relax_fmax > args.relax_fmax:
            bfgs = BFGS(atoms, logfile=str(output_dir / "relax_bfgs.log"))
            bfgs.run(fmax=args.relax_fmax, steps=args.bfgs_steps)
            relax_forces = atoms.get_forces()
            relax_fmax = float(np.linalg.norm(relax_forces, axis=1).max())
        write(output_dir / "relaxed.extxyz", atoms, format="extxyz")
        if not math.isfinite(relax_fmax) or relax_fmax > args.relax_fmax:
            quality.fail(
                "relax",
                f"relax final max_force {relax_fmax} eV/A exceeds {args.relax_fmax} eV/A.",
                {"relax_final_max_force_eV_A": relax_fmax},
            )

        stabilization_steps = int(round(args.stabilization_ps * 1000.0 / args.timestep_fs))
        production_steps = int(round(args.production_ps * 1000.0 / args.timestep_fs))
        nve_steps = int(round(args.nve_audit_ps * 1000.0 / args.timestep_fs))
        time_fs = 0.0
        highres_until_fs = args.short_highres_window_ps * 1000.0
        time_fs = run_langevin_phase(
            atoms,
            "stabilization",
            stabilization_steps,
            args.timestep_fs,
            args.temperature_K,
            args.friction_per_fs,
            args.log_interval,
            args.highres_interval,
            log_path,
            quality,
            args,
            bonds,
            bonded_pairs,
            reference_bond_lengths,
            short_highres_path,
            trajectory_path,
            unwrap,
            write_long=False,
            highres_until_fs=highres_until_fs,
            time_offset_fs=time_fs,
        )
        time_fs = run_langevin_phase(
            atoms,
            "production",
            production_steps,
            args.timestep_fs,
            args.temperature_K,
            args.friction_per_fs,
            args.log_interval,
            args.trajectory_interval,
            log_path,
            quality,
            args,
            bonds,
            bonded_pairs,
            reference_bond_lengths,
            short_highres_path,
            trajectory_path,
            unwrap,
            write_long=True,
            highres_until_fs=highres_until_fs,
            time_offset_fs=time_fs,
        )
        nve_audit = run_nve_audit(
            atoms,
            nve_steps,
            args.timestep_fs,
            args.log_interval,
            log_path,
            quality,
            args,
            bonds,
            bonded_pairs,
            reference_bond_lengths,
            time_offset_fs=time_fs,
        )
    except QualityGateError:
        status = "failed_quality_gate"

    if unwrap.frames:
        np.savez_compressed(
            output_dir / "trajectory_unwrapped.npz",
            positions_A=np.stack(unwrap.frames),
            steps=np.array(unwrap.steps, dtype=np.int64),
            times_fs=np.array(unwrap.times_fs, dtype=np.float64),
            cell_A=np.asarray(atoms.cell, dtype=np.float64),
        )
    else:
        np.savez_compressed(
            output_dir / "trajectory_unwrapped.npz",
            positions_A=np.empty((0, len(atoms), 3), dtype=np.float32),
            steps=np.empty((0,), dtype=np.int64),
            times_fs=np.empty((0,), dtype=np.float64),
            cell_A=np.asarray(atoms.cell, dtype=np.float64),
        )

    elapsed = time.time() - start
    quality_report = {
        "ok": quality.ok,
        "status": status,
        "failures": quality.failures,
        "warnings": quality.warnings,
        "last_metrics": quality.last_metrics,
        "gates": {
            "relax_fmax_eV_A": args.relax_fmax,
            "production_preferred_relax_fmax_eV_A": args.production_relax_fmax,
            "temperature_mean_window": "+/-10-15% target, assessed in post-processing",
            "temperature_instant_abort": f">2x target for >{args.high_temp_abort_steps} steps",
            "bond_ratio_window": [args.bond_lower_ratio, args.bond_upper_ratio],
            "min_nonbonded_distance_A": args.min_nonbonded_distance_A,
        },
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(quality_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    analysis_preview = {
        "atom_counts": atom_counts(atoms),
        "n_atoms": len(atoms),
        "trajectory_frames": len(unwrap.frames),
        "last_metrics": quality.last_metrics,
        "post_processing_status": "preview_only",
        "label_status": "provisional MLFF labels",
    }
    (output_dir / "analysis_preview.json").write_text(
        json.dumps(analysis_preview, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "ok": quality.ok,
        "status": status,
        "input": str(input_path.resolve()),
        "output_dir": str(output_dir),
        "n_atoms": len(atoms),
        "seed": args.seed,
        "timestep_fs": args.timestep_fs,
        "temperature_K": args.temperature_K,
        "elapsed_s": elapsed,
        "ensemble": {
            "stabilization": "NVT",
            "production": "NVT",
            "thermostat": "Langevin",
            "friction_per_fs": args.friction_per_fs,
            "nve_audit": nve_audit,
        },
        "strides": {
            "short_highres_interval_steps": args.highres_interval,
            "trajectory_interval_steps": args.trajectory_interval,
            "md_log_interval_steps": args.log_interval,
            "short_highres_window_ps": args.short_highres_window_ps,
        },
        "files": {
            "input": str(output_dir / "input.extxyz"),
            "topology": str(output_dir / "topology.json"),
            "system_manifest": str(output_dir / "system_manifest.yaml"),
            "relaxed": str(output_dir / "relaxed.extxyz"),
            "short_highres": str(short_highres_path),
            "trajectory": str(trajectory_path),
            "trajectory_unwrapped": str(output_dir / "trajectory_unwrapped.npz"),
            "md_log": str(log_path),
            "summary": str(output_dir / "summary.json"),
            "quality_report": str(output_dir / "quality_report.json"),
            "analysis_preview": str(output_dir / "analysis_preview.json"),
        },
        "model": {
            "config": str(model_config_path),
            "model_id": model_config.get("model_id"),
            "local_checkpoint": model_config.get("local_checkpoint"),
            "fallback_local_checkpoint": model_config.get("fallback_local_checkpoint"),
            "selected_head": selected_head,
            "candidate_heads": model_config.get("candidate_heads"),
            "default_dtype": model_config.get("default_dtype"),
            "fullgraph": model_config.get("fullgraph", False),
        },
        "provenance": environment_provenance(
            model_config.get("local_checkpoint"),
            device=model_config.get("device", "auto"),
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if quality.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
