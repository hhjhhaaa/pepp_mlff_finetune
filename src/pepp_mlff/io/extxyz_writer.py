from __future__ import annotations

from pathlib import Path
from typing import Any


GLOBAL_METADATA_KEYS = [
    "config_type",
    "patch_id",
    "parent_frame_id",
    "system_id",
    "system_type",
    "composition",
    "pe_fraction_global",
    "pp_fraction_global",
    "density_g_cm3",
    "local_pe_fraction",
    "local_pp_fraction",
    "patch_region",
    "sample_type",
    "pbc",
    "cell_source",
    "cap_type",
    "fixed_atoms_present",
    "force_label_valid",
    "force_label_scope",
    "stress_available",
    "stress_key",
    "stress_unit",
    "train_stress",
    "dataset_version",
    "label_level_id",
    "source_builder_commit",
    "cp2k_input_sha256",
    "cp2k_output_sha256",
    "cp2k_version",
    "parser_version",
    "cp2k_xc_functional",
    "dispersion",
    "basis_set",
    "potential",
    "cutoff",
    "rel_cutoff",
    "eps_scf",
    "scf_converged",
    "calculation_completed",
    "calculation_type",
    "reject_reason",
]


def atoms_from_sample(sample: dict[str, Any]):
    """Build an ASE Atoms object from a normalized CP2K sample dictionary."""
    try:
        import numpy as np
        from ase import Atoms
    except ImportError as exc:
        raise RuntimeError("Missing dependency ase/numpy. Install project dependencies first.") from exc

    atoms = Atoms(symbols=sample["symbols"], positions=sample["positions_A"])
    cell = sample.get("cell_A")
    pbc = sample.get("pbc", False)
    if cell is not None:
        atoms.cell = cell
        atoms.pbc = pbc
    else:
        atoms.pbc = False

    atoms.info["energy"] = sample["energy_eV"]
    atoms.arrays["forces"] = np.asarray(sample["forces_eV_per_A"], dtype=float)

    atom_role = sample.get("atom_role")
    if atom_role is not None:
        atoms.arrays["atom_role"] = np.asarray(atom_role, dtype=str)

    for key in GLOBAL_METADATA_KEYS:
        if key in sample and sample[key] is not None:
            atoms.info[key] = sample[key]

    if sample.get("stress_available") and sample.get("stress") is not None:
        atoms.info[sample.get("stress_key", "stress")] = sample["stress"]

    return atoms


def write_extxyz(samples: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write normalized samples to extxyz using ASE.

    Stress/virial fields are reserved and only written when explicitly supplied;
    this function does not fabricate stress labels.
    """
    try:
        from ase.io import write
    except ImportError as exc:
        raise RuntimeError("Missing dependency ase. Install project dependencies first.") from exc

    atoms_list = [atoms_from_sample(sample) for sample in samples]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, atoms_list, format="extxyz")
