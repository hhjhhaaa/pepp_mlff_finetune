#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ase.io import read
from pepp_mlff.io.lammps_topology import (
    build_topology_from_lammps,
    parse_component_chain_counts,
    read_lammps_box,
)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def first_existing(candidates: list[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate and candidate.expanduser().exists():
            return candidate.expanduser().resolve()
    return None


def find_obabel(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        shutil.which("obabel"),
        "/home/jinhao/miniforge3/envs/pepp-graph-spib/bin/obabel",
        "/home/jinhao/miniconda3/envs/SurfDock/bin/obabel",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return Path(candidate).expanduser()
    raise FileNotFoundError("Open Babel executable not found; pass --obabel.")


def patch_extxyz_cell(path: Path, box: tuple[float, float, float]) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Open Babel output is not a valid XYZ/exyz file: {path}")
    lattice = f'{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}'
    lines[1] = f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"'
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_with_obabel(obabel: Path, src: Path, dst: Path, box: tuple[float, float, float]) -> None:
    input_format = "pdb" if src.suffix == ".gz" or src.suffix.lower() == ".pdb" else src.suffix.lstrip(".")
    completed = subprocess.run(
        [str(obabel), f"-i{input_format}", str(src), "-oexyz", "-O", str(dst)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Open Babel conversion failed:\n{completed.stdout}\n{completed.stderr}")
    patch_extxyz_cell(dst, box)


def copy_extxyz(src: Path, dst: Path, box: tuple[float, float, float]) -> None:
    shutil.copy2(src, dst)
    patch_extxyz_cell(dst, box)


def atom_counts(atoms) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


def import_emc(args: argparse.Namespace) -> Path:
    builder_metadata_path = Path(args.metadata_yaml).expanduser().resolve() if args.metadata_yaml else None
    if builder_metadata_path and not builder_metadata_path.exists():
        raise FileNotFoundError(builder_metadata_path)
    builder_metadata = load_yaml(builder_metadata_path) if builder_metadata_path else {}

    lane = builder_metadata.get("structure_task", {}).get("lane")
    if lane and lane != args.expected_lane:
        raise ValueError(f"Expected initial_builder lane {args.expected_lane!r}, got {lane!r}.")

    source_dir = Path(args.source_dir).expanduser().resolve() if args.source_dir else None
    if source_dir is None and builder_metadata_path:
        source_dir = builder_metadata_path.parent
    if source_dir is None:
        raise ValueError("Pass --source-dir or --metadata-yaml from initial_builder.")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    metadata_paths = builder_metadata.get("paths", {})
    system_id = args.system_id or builder_metadata.get("system_id") or source_dir.name
    components_value = args.components or ",".join(builder_metadata.get("components", []))
    if not components_value:
        raise ValueError("Pass --components or use an initial_builder metadata.yaml with components.")
    structure = Path(args.structure).expanduser().resolve() if args.structure else None
    if structure is None:
        metadata_structure = metadata_paths.get("mlff_start_extxyz") or metadata_paths.get("extxyz") or metadata_paths.get("pdb")
        candidates = [
            Path(metadata_structure).expanduser() if metadata_structure else None,
            source_dir / "polymer.extxyz",
            source_dir / "relaxed.extxyz",
            source_dir / "polyethylene.extxyz",
            source_dir / "polymer.pdb",
            source_dir / "polyethylene.pdb.gz",
            source_dir / "polyethylene.pdb",
        ]
        structure = first_existing(candidates)
    if structure is None or not structure.exists():
        raise FileNotFoundError("No EMC structure file found; pass --structure explicitly.")
    data_path = Path(args.lammps_data).expanduser().resolve() if args.lammps_data else None
    if data_path is None:
        metadata_data = metadata_paths.get("mlff_start_lammps_data") or metadata_paths.get("lammps_data")
        candidates = [
            Path(metadata_data).expanduser() if metadata_data else None,
            source_dir / "relaxed.data",
            source_dir / "polymer.data",
            source_dir / "polyethylene.data",
        ]
        data_path = first_existing(candidates)
    if data_path is None or not data_path.exists():
        raise FileNotFoundError("No EMC LAMMPS data file found; pass --lammps-data explicitly.")
    emc_script = Path(args.emc_script).expanduser().resolve() if args.emc_script else None
    if emc_script is None:
        metadata_script = metadata_paths.get("emc_recipe")
        candidates = [
            Path(metadata_script).expanduser() if metadata_script else None,
            source_dir / "build.emc",
            source_dir / "polymer.esh",
            source_dir / "polyethylene.esh",
        ]
        emc_script = first_existing(candidates)
    if emc_script is None or not emc_script.exists():
        raise FileNotFoundError("No EMC recipe/script found; pass --emc-script explicitly.")

    obabel = find_obabel(args.obabel)
    box = read_lammps_box(data_path)
    out_dir = ROOT / "data" / "emc_systems" / system_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if structure.suffix.lower() == ".extxyz":
        copy_extxyz(structure, out_dir / "input.extxyz", box)
    else:
        convert_with_obabel(obabel, structure, out_dir / "input.extxyz", box)
    atoms = read(out_dir / "input.extxyz")
    shutil.copy2(data_path, out_dir / data_path.name)
    shutil.copy2(emc_script, out_dir / emc_script.name)
    params = Path(args.params).expanduser().resolve() if args.params else None
    if params is None:
        metadata_params = metadata_paths.get("params")
        candidates = [Path(metadata_params).expanduser()] if metadata_params else []
        candidates.extend(sorted(source_dir.glob("*.params")))
        params = first_existing(candidates)
    if params and params.exists():
        shutil.copy2(params, out_dir / params.name)

    components = [item.strip().upper() for item in components_value.split(",")]
    component_chain_counts_raw = args.component_chain_counts or builder_metadata.get("component_chain_counts_arg")
    if not component_chain_counts_raw and builder_metadata.get("component_chain_counts"):
        counts = builder_metadata.get("component_chain_counts", {})
        component_chain_counts_raw = ",".join(f"{component}:{int(counts[component])}" for component in components if component in counts)
    topology = build_topology_from_lammps(
        data_path,
        atoms.get_chemical_symbols(),
        components,
        parse_component_chain_counts(component_chain_counts_raw),
    )
    topology["metadata"]["lammps_data_source"] = str(data_path)
    if "PS" in set(components) and not topology.get("phenyl_rings"):
        raise ValueError("PS EMC import requires detectable phenyl_rings before production use.")
    (out_dir / "topology.json").write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "system_id": system_id,
        "components": components,
        "n_chains": args.n_chains or builder_metadata.get("n_chains") or topology["metadata"]["n_molecules"],
        "component_chain_counts": parse_component_chain_counts(component_chain_counts_raw),
        "repeat_units": args.repeat_units or builder_metadata.get("repeat_units"),
        "target_density_g_cm3": args.target_density_g_cm3 or builder_metadata.get("target_density_g_cm3"),
        "target_temperature_K": args.target_temperature_K or builder_metadata.get("target_temperature_K"),
        "box": {
            "type": "periodic" if any(atoms.pbc) else "unknown",
            "cell_lengths_A": [float(x) for x in atoms.cell.lengths()],
            "pbc": [bool(x) for x in atoms.pbc],
        },
        "builder": {
            "builder_used": "emc",
            "emc_success": True,
            "coordinate_source": "emc",
            "topology_source": "emc",
            "force_field": args.force_field or builder_metadata.get("builder", {}).get("force_field"),
            "source_dir": str(source_dir),
            "emc_script": str(emc_script),
            "lammps_data": str(data_path),
            "params": str(params) if params else None,
            "initial_builder_metadata": str(builder_metadata_path) if builder_metadata_path else None,
            "format_converter": "openbabel",
            "obabel_executable": str(obabel),
        },
        "structure_task": builder_metadata.get("structure_task", {"lane": args.expected_lane}),
        "relaxation": builder_metadata.get("relaxation", {}),
        "builder_seed": args.builder_seed,
        "preprocess_history": [
            {
                "operation": "import_emc_system",
                "source_structure": str(structure),
                "source_lammps_data": str(data_path),
            }
        ],
        "density_source": args.density_source,
        "atom_counts": atom_counts(atoms),
        "label_status": "provisional MLFF labels after zero-shot MACE-MD",
    }
    dump_yaml(out_dir / "system_manifest.yaml", manifest)
    summary = {
        "ok": True,
        "system_id": system_id,
        "n_atoms": len(atoms),
        "atom_counts": atom_counts(atoms),
        "output_dir": str(out_dir),
        "input": str(out_dir / "input.extxyz"),
        "topology": str(out_dir / "topology.json"),
        "system_manifest": str(out_dir / "system_manifest.yaml"),
        "n_bonds": len(topology["bonds"]),
        "n_angles": len(topology["angles"]),
        "n_dihedrals": len(topology["dihedrals"]),
        "n_phenyl_rings": len(topology["phenyl_rings"]),
    }
    (out_dir / "import_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a verified EMC polymer system for MACE-MD.")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--metadata-yaml", default=None, help="initial_builder structure-library metadata.yaml.")
    parser.add_argument("--expected-lane", default="mlff_direct")
    parser.add_argument("--system-id", default=None)
    parser.add_argument("--components", default=None, help="Comma-separated PE,PP,PS components.")
    parser.add_argument(
        "--component-chain-counts",
        default=None,
        help="Required for mixtures, for example PE:8,PP:8. Chains are assigned by EMC molecule id order.",
    )
    parser.add_argument("--structure", default=None)
    parser.add_argument("--lammps-data", default=None)
    parser.add_argument("--emc-script", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--obabel", default=None)
    parser.add_argument("--force-field", default=None)
    parser.add_argument("--builder-seed", type=int, default=None)
    parser.add_argument("--n-chains", type=int, default=None)
    parser.add_argument("--repeat-units", type=int, default=None)
    parser.add_argument("--target-density-g-cm3", type=float, default=None)
    parser.add_argument("--target-temperature-K", type=float, default=523.0)
    parser.add_argument("--density-source", default="emc_recipe")
    return parser.parse_args()


def main() -> int:
    import_emc(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
