from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


REQUIRED_METADATA = {
    "dataset_version",
    "label_level_id",
    "system_id",
    "system_type",
    "composition",
    "density_g_cm3",
    "sample_type",
    "pbc",
    "cell_source",
    "fixed_atoms_present",
    "force_label_valid",
    "force_label_scope",
    "cp2k_xc_functional",
    "basis_set",
    "potential",
    "scf_converged",
    "calculation_completed",
}

PERIODIC_SAMPLE_TYPES = {"periodic_bulk", "periodic_subcell"}
ALLOWED_FORCE_SCOPES = {"all_atoms", "mobile_atoms_only", "unknown"}


@dataclass
class DatasetCheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def check_atoms(atoms, require_standard_training: bool = True) -> DatasetCheckResult:
    """Check one ASE Atoms object for PE/PP MACE fine-tuning eligibility."""
    errors: list[str] = []
    warnings: list[str] = []

    symbols = set(atoms.get_chemical_symbols())
    allowed_symbols = {"C", "H", "O", "Si"}
    if not symbols <= allowed_symbols:
        errors.append(f"Unsupported elements: {sorted(symbols - allowed_symbols)}")

    for key in REQUIRED_METADATA:
        if key not in atoms.info or atoms.info.get(key) in (None, ""):
            errors.append(f"Missing required metadata: {key}")

    if "energy" not in atoms.info:
        errors.append("Missing energy label in atoms.info['energy']")
    if "forces" not in atoms.arrays:
        errors.append("Missing forces label in atoms.arrays['forces']")
    else:
        forces = atoms.arrays["forces"]
        if getattr(forces, "shape", None) != (len(atoms), 3):
            errors.append(f"Invalid forces shape: {getattr(forces, 'shape', None)}")
        try:
            import numpy as np

            if not np.isfinite(forces).all():
                errors.append("Forces contain NaN or Inf")
        except ImportError:
            pass

    if "atom_role" not in atoms.arrays:
        errors.append("Missing per-atom atom_role array")
    elif len(atoms.arrays["atom_role"]) != len(atoms):
        errors.append("atom_role length must match number of atoms")

    energy = atoms.info.get("energy")
    if isinstance(energy, (int, float)) and not math.isfinite(energy):
        errors.append("Energy contains NaN or Inf")

    sample_type = atoms.info.get("sample_type")
    if sample_type in PERIODIC_SAMPLE_TYPES and not atoms.pbc.any():
        errors.append(f"{sample_type} must have periodic boundary conditions")
    if sample_type in PERIODIC_SAMPLE_TYPES and atoms.cell.volume <= 0:
        errors.append(f"{sample_type} must have a valid cell")

    if atoms.info.get("force_label_scope") not in ALLOWED_FORCE_SCOPES:
        errors.append("force_label_scope must be all_atoms, mobile_atoms_only, or unknown")

    if require_standard_training:
        if not _truthy(atoms.info.get("force_label_valid")):
            errors.append("force_label_valid must be true for standard MACE fine-tuning")
        if atoms.info.get("force_label_scope") != "all_atoms":
            errors.append(
                "Standard MACE fine-tuning requires all_atoms force labels unless atom-wise "
                "force masks are supported by a custom training path"
            )
        if not _truthy(atoms.info.get("scf_converged")):
            errors.append("SCF did not converge")
        if not _truthy(atoms.info.get("calculation_completed")):
            errors.append("CP2K calculation did not complete")

    return DatasetCheckResult(ok=not errors, errors=errors, warnings=warnings)


def check_extxyz(path: str, require_standard_training: bool = True) -> DatasetCheckResult:
    """Check an extxyz file and aggregate quality-gate failures."""
    try:
        from ase.io import read
    except ImportError as exc:
        raise RuntimeError("Missing dependency ase. Install project dependencies first.") from exc

    all_errors: list[str] = []
    all_warnings: list[str] = []
    images = read(path, index=":")
    for index, atoms in enumerate(images):
        result = check_atoms(atoms, require_standard_training=require_standard_training)
        all_errors.extend(f"frame {index}: {error}" for error in result.errors)
        all_warnings.extend(f"frame {index}: {warning}" for warning in result.warnings)
    return DatasetCheckResult(ok=not all_errors, errors=all_errors, warnings=all_warnings)
