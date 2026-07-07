import json
from pathlib import Path

import numpy as np
import yaml

from pepp_mlff.analysis.trajectory_validation import analyze_batch, write_csv, write_markdown


def _write_extxyz(path: Path, symbols: list[str]) -> None:
    lines = [
        str(len(symbols)),
        'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"',
    ]
    lines.extend(f"{symbol} {idx}.0 0.0 0.0" for idx, symbol in enumerate(symbols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analyze_batch_writes_validation_outputs(tmp_path: Path):
    run_dir = tmp_path / "batch" / "PE100_N2_C1_emc_seed1" / "mace_mh_0" / "replica_0001"
    run_dir.mkdir(parents=True)
    summary = {
        "ok": True,
        "status": "ok",
        "system_id": "PE100_N2_C1_emc_seed1",
        "n_atoms": 4,
        "temperature_K": 523.0,
        "ensemble": {"nve_audit": {"energy_drift_eV_per_atom": 1e-6}},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "quality_report.json").write_text(
        json.dumps({"ok": True, "status": "ok", "failures": []}),
        encoding="utf-8",
    )
    (run_dir / "system_manifest.yaml").write_text(
        yaml.safe_dump({"system_id": "PE100_N2_C1_emc_seed1", "components": ["PE"], "target_temperature_K": 523.0}),
        encoding="utf-8",
    )
    topology = {"bonds": [[0, 1]], "angles": [], "dihedrals": [[0, 1, 2, 3]], "phenyl_rings": []}
    (run_dir / "topology.json").write_text(json.dumps(topology), encoding="utf-8")
    _write_extxyz(run_dir / "input.extxyz", ["C", "H", "C", "H"])
    rows = [
        {
            "phase": "production",
            "time_fs": 0.0,
            "temperature_K": 520.0,
            "energy_potential_eV_per_atom": -5.0,
            "energy_total_eV": -10.0,
            "max_force_eV_A": 1.0,
            "mean_force_eV_A": 0.5,
            "bond_min_A": 1.0,
            "bond_max_A": 1.5,
        },
        {
            "phase": "production",
            "time_fs": 1000.0,
            "temperature_K": 526.0,
            "energy_potential_eV_per_atom": -5.1,
            "energy_total_eV": -10.1,
            "max_force_eV_A": 1.2,
            "mean_force_eV_A": 0.6,
            "bond_min_A": 0.9,
            "bond_max_A": 1.6,
        },
        {"phase": "nve_audit", "time_fs": 1500.0, "temperature_K": 523.0},
        {"phase": "nve_audit", "time_fs": 2000.0, "temperature_K": 524.0},
    ]
    (run_dir / "md_log.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    positions = np.zeros((3, 4, 3), dtype=np.float32)
    positions[1, :, 0] = 0.5
    positions[2, :, 0] = 1.0
    np.savez_compressed(
        run_dir / "trajectory_unwrapped.npz",
        positions_A=positions,
        times_fs=np.array([0.0, 500.0, 1000.0]),
        cell_A=np.diag([10.0, 10.0, 10.0]),
    )

    result = analyze_batch(tmp_path / "batch", frame_stride=1)
    assert result["n_runs"] == 1
    assert result["runs"][0]["gates"]["quality_ok"] is True
    assert result["runs"][0]["msd"]["available"] is True
    assert result["runs"][0]["dihedrals"]["available"] is True

    write_csv(result, tmp_path / "summary.csv")
    write_markdown(result, tmp_path / "summary.md")
    assert "PE100_N2_C1_emc_seed1" in (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "MACE-MD Trajectory Validation" in (tmp_path / "summary.md").read_text(encoding="utf-8")
