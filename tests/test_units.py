from pepp_mlff.utils.units import (
    BOHR_TO_ANGSTROM,
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    hartree_per_bohr_to_ev_per_angstrom,
    hartree_to_ev,
)


def test_hartree_to_ev():
    assert hartree_to_ev(1.0) == HARTREE_TO_EV


def test_hartree_per_bohr_to_ev_per_angstrom():
    assert hartree_per_bohr_to_ev_per_angstrom(1.0) == HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    assert HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM == HARTREE_TO_EV / BOHR_TO_ANGSTROM
