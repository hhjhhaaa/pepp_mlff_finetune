from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def section_rows(path: str | Path, section: str) -> list[list[str]]:
    """Return tokenized numeric rows from a LAMMPS data section."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        clean = line.split("#", 1)[0].strip()
        if clean == section:
            start = idx + 1
            break
    if start is None:
        return []

    rows: list[list[str]] = []
    for line in lines[start:]:
        clean = line.split("#", 1)[0].strip()
        if not clean:
            if rows:
                break
            continue
        first = clean.split()[0]
        if not first.lstrip("-").isdigit():
            if rows:
                break
            continue
        rows.append(clean.split())
    return rows


def read_lammps_box(path: str | Path) -> tuple[float, float, float]:
    """Parse an orthorhombic LAMMPS data box as cell lengths in Angstrom."""
    bounds: dict[str, float] = {}
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.split()
        if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
            bounds["x"] = float(parts[1]) - float(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
            bounds["y"] = float(parts[1]) - float(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ["zlo", "zhi"]:
            bounds["z"] = float(parts[1]) - float(parts[0])
    if set(bounds) != {"x", "y", "z"}:
        raise ValueError(f"Could not parse orthorhombic box from {path}")
    return bounds["x"], bounds["y"], bounds["z"]


def parse_component_chain_counts(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    counts: dict[str, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        name, _, value = item.partition(":")
        if not name or not value:
            raise ValueError(f"Expected component chain counts like PE:8,PP:8; got {raw!r}")
        counts[name.strip().upper()] = int(value)
    return counts


def parse_lammps_data(path: str | Path) -> dict[str, Any]:
    """Parse topology-bearing sections from an EMC/LAMMPS full-atom data file."""
    atom_rows = []
    for row in section_rows(path, "Atoms"):
        if len(row) < 7:
            raise ValueError(f"Expected full atom style row with >=7 fields in {path}: {row}")
        atom_rows.append(
            {
                "atom_id": int(row[0]),
                "molecule_id": int(row[1]),
                "atom_type": int(row[2]),
                "charge": float(row[3]),
                "x": float(row[4]),
                "y": float(row[5]),
                "z": float(row[6]),
            }
        )
    atom_rows.sort(key=lambda item: item["atom_id"])
    id_to_index = {row["atom_id"]: idx for idx, row in enumerate(atom_rows)}

    def convert_rows(section: str, width: int) -> list[list[int]]:
        converted: list[list[int]] = []
        for row in section_rows(path, section):
            if len(row) < 2 + width:
                raise ValueError(f"Malformed {section} row in {path}: {row}")
            converted.append([id_to_index[int(atom_id)] for atom_id in row[2 : 2 + width]])
        return converted

    return {
        "atoms": atom_rows,
        "bonds": convert_rows("Bonds", 2),
        "angles": convert_rows("Angles", 3),
        "dihedrals": convert_rows("Dihedrals", 4),
        "impropers": convert_rows("Impropers", 4),
    }


def _component_ids_for_chains(
    ordered_chain_ids: list[int],
    components: list[str],
    component_chain_counts: dict[str, int],
    molecule_sizes: dict[int, int] | None = None,
) -> tuple[dict[int, str], dict[str, int], dict[str, int]]:
    requested_component_chain_counts = dict(component_chain_counts)
    if len(components) == 1 and not component_chain_counts:
        return {chain_id: components[0] for chain_id in ordered_chain_ids}, {}, {}
    if not component_chain_counts:
        raise ValueError(
            "Multiple components require --component-chain-counts, for example PE:8,PP:8."
        )
    missing = [component for component in components if component not in component_chain_counts]
    if missing:
        raise ValueError(f"Missing component counts for: {missing}")
    if sum(component_chain_counts.values()) != len(ordered_chain_ids):
        inferred = _infer_component_counts_from_molecule_size_runs(
            ordered_chain_ids,
            components,
            molecule_sizes or {},
        )
        if not inferred:
            raise ValueError(
                "Component chain counts do not match molecule count: "
                f"{component_chain_counts} vs {len(ordered_chain_ids)} molecules."
            )
        component_chain_counts.clear()
        component_chain_counts.update(inferred)
    chain_to_component: dict[int, str] = {}
    cursor = 0
    for component in components:
        count = component_chain_counts[component]
        for chain_id in ordered_chain_ids[cursor : cursor + count]:
            chain_to_component[chain_id] = component
        cursor += count
    return chain_to_component, requested_component_chain_counts, dict(component_chain_counts)


def _infer_component_counts_from_molecule_size_runs(
    ordered_chain_ids: list[int],
    components: list[str],
    molecule_sizes: dict[int, int],
) -> dict[str, int]:
    """Infer EMC component chain counts from consecutive molecule-size runs.

    EMC can adjust requested chain counts to satisfy composition/size targets.
    For the batch2a binary melts, molecules are emitted grouped by component and
    the components have distinct atom counts. Use that actual LAMMPS ordering
    when the requested metadata count is stale.
    """
    if len(components) < 2 or len(ordered_chain_ids) != len(molecule_sizes):
        return {}
    runs: list[tuple[int, int]] = []
    last_size: int | None = None
    for chain_id in ordered_chain_ids:
        size = molecule_sizes[chain_id]
        if size != last_size:
            runs.append((size, 1))
            last_size = size
        else:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
    if len(runs) != len(components):
        return {}
    if len({size for size, _count in runs}) != len(runs):
        return {}
    return {component: count for component, (_size, count) in zip(components, runs)}


def _find_six_member_carbon_rings(symbols: list[str], bonds: list[list[int]]) -> list[list[int]]:
    carbon = {idx for idx, symbol in enumerate(symbols) if symbol == "C"}
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in bonds:
        if a in carbon and b in carbon:
            adjacency[a].add(b)
            adjacency[b].add(a)

    cycles: set[tuple[int, ...]] = set()
    for start in sorted(carbon):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == 6:
                if start in adjacency[node]:
                    cycle = tuple(sorted(path))
                    # Phenyl rings are simple six-carbon cycles; reject obvious fused/chorded cycles.
                    internal_edges = 0
                    cycle_set = set(cycle)
                    for item in cycle:
                        internal_edges += len(adjacency[item] & cycle_set)
                    if internal_edges == 12:
                        cycles.add(cycle)
                continue
            for neighbor in adjacency[node]:
                if neighbor < start or neighbor in path:
                    continue
                stack.append((neighbor, path + [neighbor]))
    return [list(cycle) for cycle in sorted(cycles)]


def build_topology_from_lammps(
    data_path: str | Path,
    symbols: list[str],
    components: list[str],
    component_chain_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    parsed = parse_lammps_data(data_path)
    atoms = parsed["atoms"]
    if len(atoms) != len(symbols):
        raise ValueError(f"LAMMPS atom count {len(atoms)} != structure atom count {len(symbols)}")

    components = [component.upper() for component in components]
    molecule_ids = [int(row["molecule_id"]) for row in atoms]
    ordered_molecules = sorted(set(molecule_ids))
    molecule_to_chain = {molecule_id: idx for idx, molecule_id in enumerate(ordered_molecules)}
    molecule_sizes = {
        molecule_id: sum(1 for row in atoms if int(row["molecule_id"]) == molecule_id)
        for molecule_id in ordered_molecules
    }
    chain_to_component, requested_component_chain_counts, effective_component_chain_counts = _component_ids_for_chains(
        ordered_molecules,
        components,
        dict(component_chain_counts or {}),
        molecule_sizes,
    )
    component_count_source = "metadata"
    if requested_component_chain_counts != effective_component_chain_counts:
        component_count_source = "inferred_from_lammps_molecule_size_runs"

    phenyl_rings = _find_six_member_carbon_rings(symbols, parsed["bonds"])
    phenyl_atoms = {idx for ring in phenyl_rings for idx in ring}
    carbon_atoms = [idx for idx, symbol in enumerate(symbols) if symbol == "C"]
    backbone_atoms = [idx for idx in carbon_atoms if idx not in phenyl_atoms]
    sidegroup_atoms = sorted(phenyl_atoms)
    if "PS" not in components:
        sidegroup_atoms = []

    topology = {
        "atom_indexing": "0-based",
        "component_id": [chain_to_component[molecule_id] for molecule_id in molecule_ids],
        "chain_id": [molecule_to_chain[molecule_id] for molecule_id in molecule_ids],
        "monomer_id": [-1] * len(symbols),
        "segment_id": [molecule_to_chain[molecule_id] for molecule_id in molecule_ids],
        "molecule_id": molecule_ids,
        "atom_type": [int(row["atom_type"]) for row in atoms],
        "charge": [float(row["charge"]) for row in atoms],
        "bonds": parsed["bonds"],
        "angles": parsed["angles"],
        "dihedrals": parsed["dihedrals"],
        "impropers": parsed["impropers"],
        "backbone_atoms": backbone_atoms,
        "sidegroup_atoms": sidegroup_atoms,
        "phenyl_rings": phenyl_rings,
        "metadata": {
            "topology_source": "emc",
            "complete_polymer_topology": bool(parsed["bonds"] and parsed["dihedrals"]),
            "component_chain_counts": effective_component_chain_counts,
            "requested_component_chain_counts": requested_component_chain_counts,
            "component_chain_count_source": component_count_source,
            "molecule_sizes": molecule_sizes,
            "n_molecules": len(ordered_molecules),
            "n_bonds": len(parsed["bonds"]),
            "n_angles": len(parsed["angles"]),
            "n_dihedrals": len(parsed["dihedrals"]),
            "n_phenyl_rings": len(phenyl_rings),
            "note": "Topology imported from EMC/LAMMPS data. monomer_id remains -1 until a chain-specific monomer annotator is available.",
        },
    }
    if "PS" in components and not phenyl_rings:
        topology["metadata"]["complete_polymer_topology"] = False
    return topology
