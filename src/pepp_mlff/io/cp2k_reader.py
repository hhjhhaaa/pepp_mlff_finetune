from __future__ import annotations

from pathlib import Path
from typing import Any


def read_cp2k_patch_output(path: Path) -> dict[str, Any]:
    """Read one CP2K-labeled PE/PP sample.

    The eventual parser must extract atom elements, coordinates preferably in
    Angstrom, total energy, atomic forces, cell for periodic samples, `patch_id`,
    `parent_frame_id`, system/composition metadata, sample type
    (`capped_patch`, `periodic_bulk`, `periodic_subcell`), pbc/cell source,
    cap information, force-label validity/scope, per-atom roles
    (`polymer_core`, `cap`, `frozen_boundary`), stress/virial availability, CP2K
    quality fields, and dataset provenance fields.

    Unit conversion must call `pepp_mlff.utils.units` rather than hard-coding
    constants inside parser branches.
    """
    raise NotImplementedError(
        f"CP2K parsing is intentionally not implemented yet for {path}. "
        "Define the exact CP2K output conventions before enabling dataset builds."
    )


def parse_energy(*args, **kwargs):
    """Parse total energy and convert through `utils.units` when needed."""
    raise NotImplementedError("Energy parser is not implemented yet.")


def parse_forces(*args, **kwargs):
    """Parse atomic forces and convert through `utils.units` when needed."""
    raise NotImplementedError("Forces parser is not implemented yet.")


def parse_cell(*args, **kwargs):
    """Parse periodic cell vectors for periodic_bulk or periodic_subcell samples."""
    raise NotImplementedError("Cell parser is not implemented yet.")


def parse_atomic_structure(*args, **kwargs):
    """Parse atom symbols, positions, and per-atom roles."""
    raise NotImplementedError("Atomic structure parser is not implemented yet.")
