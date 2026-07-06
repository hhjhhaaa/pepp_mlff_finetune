from __future__ import annotations


def run_short_rollout(*args, **kwargs):
    """Reserve a short ASE MD rollout interface for NVE/NVT stability checks."""
    raise NotImplementedError(
        "ASE rollout validation is reserved. Future checks should include NVE drift, "
        "NVT stability, bond sanity, RDF, Rg, dihedrals, PE-PP contacts, and MSD."
    )
