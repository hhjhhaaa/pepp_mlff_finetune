#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ase.io import read


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_lammps_bonds(data_path: Path) -> list[list[int]]:
    bonds: list[list[int]] = []
    in_bonds = False
    for raw_line in data_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "Bonds":
            in_bonds = True
            continue
        if in_bonds and line and not line[0].isdigit():
            break
        if in_bonds:
            parts = line.split()
            if len(parts) >= 4:
                bonds.append([int(parts[2]) - 1, int(parts[3]) - 1])
    return bonds


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


def lammps_box(data_path: Path) -> tuple[float, float, float]:
    bounds: dict[str, float] = {}
    for raw_line in data_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.split()
        if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
            bounds["x"] = float(parts[1]) - float(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
            bounds["y"] = float(parts[1]) - float(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ["zlo", "zhi"]:
            bounds["z"] = float(parts[1]) - float(parts[0])
    if set(bounds) != {"x", "y", "z"}:
        raise ValueError(f"Could not parse orthorhombic box from {data_path}")
    return bounds["x"], bounds["y"], bounds["z"]


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


def atom_counts(atoms) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


def import_emc(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    structure = Path(args.structure).expanduser().resolve() if args.structure else None
    if structure is None:
        candidates = [
            source_dir / "polymer.extxyz",
            source_dir / "polyethylene.extxyz",
            source_dir / "polymer.pdb",
            source_dir / "polyethylene.pdb.gz",
            source_dir / "polyethylene.pdb",
        ]
        structure = next((candidate for candidate in candidates if candidate.exists()), None)
    if structure is None or not structure.exists():
        raise FileNotFoundError("No EMC structure file found; pass --structure explicitly.")
    data_path = Path(args.lammps_data).expanduser().resolve() if args.lammps_data else None
    if data_path is None:
        candidates = [source_dir / "polymer.data", source_dir / "polyethylene.data"]
        data_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if data_path is None or not data_path.exists():
        raise FileNotFoundError("No EMC LAMMPS data file found; pass --lammps-data explicitly.")
    emc_script = Path(args.emc_script).expanduser().resolve() if args.emc_script else None
    if emc_script is None:
        candidates = [source_dir / "build.emc", source_dir / "polymer.esh", source_dir / "polyethylene.esh"]
        emc_script = next((candidate for candidate in candidates if candidate.exists()), None)
    if emc_script is None or not emc_script.exists():
        raise FileNotFoundError("No EMC recipe/script found; pass --emc-script explicitly.")

    obabel = find_obabel(args.obabel)
    box = lammps_box(data_path)
    out_dir = ROOT / "data" / "emc_systems" / args.system_id
    out_dir.mkdir(parents=True, exist_ok=True)
    convert_with_obabel(obabel, structure, out_dir / "input.extxyz", box)
    atoms = read(out_dir / "input.extxyz")
    shutil.copy2(data_path, out_dir / data_path.name)
    shutil.copy2(emc_script, out_dir / emc_script.name)
    params = Path(args.params).expanduser().resolve() if args.params else None
    if params is None:
        candidates = sorted(source_dir.glob("*.params"))
        params = candidates[0] if candidates else None
    if params and params.exists():
        shutil.copy2(params, out_dir / params.name)

    bonds = read_lammps_bonds(data_path)
    topology = {
        "atom_indexing": "0-based",
        "component_id": [args.components] * len(atoms),
        "chain_id": [-1] * len(atoms),
        "monomer_id": [-1] * len(atoms),
        "segment_id": [-1] * len(atoms),
        "bonds": bonds,
        "dihedrals": [],
        "backbone_atoms": [idx for idx, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "C"],
        "sidegroup_atoms": [],
        "phenyl_rings": [],
        "metadata": {
            "topology_source": "emc",
            "complete_polymer_topology": False,
            "lammps_data_source": str(data_path),
            "note": "Imported EMC bond topology. Full chain/monomer/dihedral annotation should be added before production analysis.",
        },
    }
    if "PS" in {item.strip().upper() for item in args.components.split(",")}:
        raise ValueError("PS EMC import requires phenyl ring extraction before production use.")
    (out_dir / "topology.json").write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "system_id": args.system_id,
        "components": [item.strip().upper() for item in args.components.split(",")],
        "n_chains": args.n_chains,
        "repeat_units": args.repeat_units,
        "target_density_g_cm3": args.target_density_g_cm3,
        "target_temperature_K": args.target_temperature_K,
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
            "force_field": args.force_field,
            "source_dir": str(source_dir),
            "emc_script": str(emc_script),
            "lammps_data": str(data_path),
            "params": str(params) if params else None,
            "format_converter": "openbabel",
            "obabel_executable": str(obabel),
        },
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
        "system_id": args.system_id,
        "n_atoms": len(atoms),
        "atom_counts": atom_counts(atoms),
        "output_dir": str(out_dir),
        "input": str(out_dir / "input.extxyz"),
        "topology": str(out_dir / "topology.json"),
        "system_manifest": str(out_dir / "system_manifest.yaml"),
        "n_bonds": len(bonds),
    }
    (out_dir / "import_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a verified EMC polymer system for MACE-MD.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--components", required=True, help="Comma-separated PE,PP,PS components.")
    parser.add_argument("--structure", default=None)
    parser.add_argument("--lammps-data", default=None)
    parser.add_argument("--emc-script", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--obabel", default=None)
    parser.add_argument("--force-field", default="opls-aa")
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
