from pathlib import Path

from pepp_mlff.io.lammps_topology import (
    build_topology_from_lammps,
    parse_component_chain_counts,
    read_lammps_box,
)


def test_build_topology_from_lammps_sections(tmp_path: Path):
    data = tmp_path / "toy.data"
    data.write_text(
        """toy

4 atoms
3 bonds
2 angles
1 dihedrals

2 atom types
1 bond types
1 angle types
1 dihedral types

0 10 xlo xhi
0 11 ylo yhi
0 12 zlo zhi

Masses

1 12.011
2 1.008

Atoms

1 1 1 -0.1 0 0 0
2 1 1 -0.1 1 0 0
3 1 1 -0.1 2 0 0
4 1 2 0.1 3 0 0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4

Angles

1 1 1 2 3
2 1 2 3 4

Dihedrals

1 1 1 2 3 4
""",
        encoding="utf-8",
    )

    topology = build_topology_from_lammps(data, ["C", "C", "C", "H"], ["PE"])

    assert read_lammps_box(data) == (10.0, 11.0, 12.0)
    assert topology["component_id"] == ["PE", "PE", "PE", "PE"]
    assert topology["chain_id"] == [0, 0, 0, 0]
    assert topology["bonds"] == [[0, 1], [1, 2], [2, 3]]
    assert topology["angles"] == [[0, 1, 2], [1, 2, 3]]
    assert topology["dihedrals"] == [[0, 1, 2, 3]]
    assert topology["metadata"]["complete_polymer_topology"] is True


def test_component_chain_counts_for_mixture(tmp_path: Path):
    data = tmp_path / "mix.data"
    data.write_text(
        """mix

2 atoms
0 bonds
0 angles
0 dihedrals

1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Atoms

1 10 1 0.0 0 0 0
2 20 1 0.0 1 0 0
""",
        encoding="utf-8",
    )

    topology = build_topology_from_lammps(
        data,
        ["C", "C"],
        ["PE", "PP"],
        parse_component_chain_counts("PE:1,PP:1"),
    )

    assert topology["component_id"] == ["PE", "PP"]
    assert topology["chain_id"] == [0, 1]


def test_component_chain_counts_can_follow_emc_molecule_size_runs(tmp_path: Path):
    data = tmp_path / "emc_adjusted_mix.data"
    data.write_text(
        """emc adjusted mix

6 atoms
0 bonds
0 angles
0 dihedrals

1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Atoms

1 10 1 0.0 0 0 0
2 10 1 0.0 1 0 0
3 20 1 0.0 2 0 0
4 20 1 0.0 3 0 0
5 30 1 0.0 4 0 0
6 40 1 0.0 5 0 0
""",
        encoding="utf-8",
    )

    topology = build_topology_from_lammps(
        data,
        ["C"] * 6,
        ["PE", "PS"],
        parse_component_chain_counts("PE:1,PS:1"),
    )

    assert topology["component_id"] == ["PE", "PE", "PE", "PE", "PS", "PS"]
    assert topology["metadata"]["component_chain_counts"] == {"PE": 2, "PS": 2}
    assert topology["metadata"]["requested_component_chain_counts"] == {"PE": 1, "PS": 1}
    assert topology["metadata"]["component_chain_count_source"] == "inferred_from_lammps_molecule_size_runs"
