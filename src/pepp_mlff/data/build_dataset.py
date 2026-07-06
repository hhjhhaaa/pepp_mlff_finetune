from __future__ import annotations

from pathlib import Path
from typing import Any

from pepp_mlff.io.cp2k_reader import read_cp2k_patch_output
from pepp_mlff.io.extxyz_writer import write_extxyz


def build_dataset_from_manifest(manifest_path: str | Path, output_extxyz: str | Path) -> None:
    """Build an extxyz dataset from a CP2K manifest.

    This function is wired but currently blocked by the intentionally unimplemented
    CP2K parser. The manifest must include all quality/provenance metadata needed
    by the dataset gate.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Missing dependency pandas. Install project dependencies first.") from exc

    manifest = pd.read_csv(manifest_path)
    samples: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        sample = read_cp2k_patch_output(Path(row["path"]))
        for key, value in row.items():
            sample.setdefault(key, value)
        samples.append(sample)
    write_extxyz(samples, output_extxyz)
