from __future__ import annotations

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


def hartree_to_ev(x):
    """Convert Hartree to eV for scalars or array-like objects."""
    return x * HARTREE_TO_EV


def hartree_per_bohr_to_ev_per_angstrom(x):
    """Convert Hartree/Bohr to eV/Angstrom for scalars or array-like objects."""
    return x * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
