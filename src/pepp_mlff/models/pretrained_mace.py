from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
import subprocess
from pathlib import Path
from typing import Any


def auto_device() -> str:
    """Return cuda when torch and CUDA are available, otherwise cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path: str | Path) -> str:
    """Compute a file SHA256 hash in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_provenance(checkpoint: str | Path | None = None, device: str | None = None) -> dict[str, Any]:
    """Collect model/environment provenance for reproducible fine-tuning."""
    resolved_device = auto_device() if device in (None, "auto") else device
    provenance: dict[str, Any] = {
        "device": resolved_device,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": None,
    }
    if checkpoint and Path(checkpoint).is_file():
        provenance["checkpoint_sha256"] = sha256_file(checkpoint)

    try:
        import torch

        provenance["torch_version"] = torch.__version__
        provenance["cuda_available"] = torch.cuda.is_available()
        provenance["torch_cuda_version"] = torch.version.cuda
        provenance["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        provenance["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        provenance.update(
            {
                "torch_version": None,
                "cuda_available": False,
                "torch_cuda_version": None,
                "cuda_device_count": 0,
                "cuda_device_name": None,
            }
        )

    try:
        provenance["mace_torch_version"] = importlib.metadata.version("mace-torch")
    except importlib.metadata.PackageNotFoundError:
        provenance["mace_torch_version"] = None
    return provenance


def list_checkpoint_heads(checkpoint: str | Path, target_device: str = "cpu") -> list[str]:
    """Return heads advertised by a local MACE-MH checkpoint."""
    exe = shutil.which("mace_select_head")
    if exe is None:
        raise RuntimeError("mace_select_head not found on PATH; cannot validate checkpoint heads.")
    completed = subprocess.run(
        [exe, "--list_heads", str(checkpoint), "--target_device", target_device],
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"mace_select_head failed for {checkpoint}:\n{output}")
    heads: list[str] = []
    in_head_block = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Available heads:":
            in_head_block = True
            continue
        if in_head_block and stripped and not stripped.startswith(("/", "torch.load", "_Jd,")):
            heads.append(stripped)
    return heads


def validate_selected_head(config: dict[str, Any], checkpoint: str | Path) -> None:
    """Validate selected_head/foundation_head against the checkpoint when requested."""
    if not config.get("head_validation_required"):
        return
    selected_head = config.get("selected_head") or config.get("foundation_head") or config.get("head")
    if not selected_head:
        raise ValueError("head_validation_required is true, but no selected_head was configured.")
    heads = list_checkpoint_heads(checkpoint)
    if selected_head not in heads:
        raise ValueError(
            f"Selected head {selected_head!r} is absent from {checkpoint}. Available heads: {heads}"
        )
    candidate_heads = config.get("candidate_heads") or []
    missing_candidates = [head for head in candidate_heads if head not in heads]
    if missing_candidates:
        raise ValueError(
            f"Candidate heads absent from {checkpoint}: {missing_candidates}. Available heads: {heads}"
        )


def load_foundation_calculator(config: dict[str, Any], allow_api_fallback: bool = False):
    """Load a MACE foundation calculator from local checkpoint.

    Production PE/PP-silica work should use local MACE-MH-1 or MACE-MH-0
    checkpoints. API fallback is retained only for legacy explicit callers and
    is disabled by the production config.
    """
    device = config.get("device", "auto")
    if device == "auto":
        device = auto_device()
    local_checkpoint = config.get("local_checkpoint")
    fallback_local_checkpoint = config.get("fallback_local_checkpoint")
    default_dtype = config.get("default_dtype")
    foundation_head = config.get("selected_head") or config.get("foundation_head") or config.get("head")
    fullgraph = config.get("fullgraph")

    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise RuntimeError(
            "mace-torch is not installed. Install PyTorch for this machine first, then run "
            "pip install -e '.[mace]' or pip install mace-torch."
        ) from exc

    if local_checkpoint and Path(local_checkpoint).is_file():
        validate_selected_head(config, local_checkpoint)
        kwargs = {"model_paths": str(local_checkpoint), "device": device}
        if default_dtype:
            kwargs["default_dtype"] = default_dtype
        if foundation_head:
            kwargs["head"] = foundation_head
        if fullgraph is not None:
            kwargs["fullgraph"] = bool(fullgraph)
        return MACECalculator(**kwargs)

    if fallback_local_checkpoint and Path(fallback_local_checkpoint).is_file():
        validate_selected_head(config, fallback_local_checkpoint)
        kwargs = {"model_paths": str(fallback_local_checkpoint), "device": device}
        if default_dtype:
            kwargs["default_dtype"] = default_dtype
        if foundation_head:
            kwargs["head"] = foundation_head
        if fullgraph is not None:
            kwargs["fullgraph"] = bool(fullgraph)
        return MACECalculator(**kwargs)

    if not allow_api_fallback:
        raise FileNotFoundError(
            "Local MACE foundation checkpoint is required for formal fine-tuning/evaluation. "
            f"Missing checkpoint: {local_checkpoint}; fallback: {fallback_local_checkpoint}"
        )

    try:
        from mace.calculators import mace_off
    except ImportError as exc:
        raise RuntimeError("Installed mace-torch does not expose mace.calculators.mace_off.") from exc
    return mace_off(model=config.get("api_model", "medium"), device=device)


def load_mace_off_calculator(config: dict[str, Any], allow_api_fallback: bool = False):
    """Backward-compatible alias for older scripts."""
    return load_foundation_calculator(config, allow_api_fallback=allow_api_fallback)


def smoke_test_calculator(calc) -> dict[str, Any]:
    """Run a methane CH4 energy/force smoke test."""
    try:
        import numpy as np
        from ase import Atoms
    except ImportError as exc:
        raise RuntimeError("Missing dependency ase/numpy. Install project dependencies first.") from exc

    positions = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.6291, 0.6291, 0.6291],
            [-0.6291, -0.6291, 0.6291],
            [-0.6291, 0.6291, -0.6291],
            [0.6291, -0.6291, -0.6291],
        ]
    )
    atoms = Atoms("CH4", positions=positions)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    return {
        "energy_eV": energy,
        "forces_shape": list(forces.shape),
        "device": str(getattr(calc, "device", None)),
        "backend": calc.__class__.__name__,
    }
