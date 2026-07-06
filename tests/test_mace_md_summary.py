import json
from pathlib import Path

import numpy as np
import yaml

from pepp_mlff.analysis.mace_md_summary import summarize_batch, summarize_run, temperature_stats


def test_temperature_stats_uses_production_rows():
    rows = [
        {"phase": "stabilization", "temperature_K": 100.0},
        {"phase": "production", "temperature_K": 520.0},
        {"phase": "production", "temperature_K": 526.0},
    ]
    stats = temperature_stats(rows, 523.0)
    assert stats["available"] is True
    assert stats["mean_K"] == 523.0
    assert stats["within_15_percent"] is True


def test_summarize_run_and_batch(tmp_path: Path):
    run_dir = tmp_path / "batch" / "PE100_N50_C10_emc_seed1" / "mace_mh_0" / "replica_0001"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "status": "ok",
                "n_atoms": 3140,
                "temperature_K": 523.0,
                "timestep_fs": 0.25,
                "strides": {
                    "trajectory_interval_steps": 100,
                    "short_highres_interval_steps": 20,
                    "short_highres_window_ps": 5.0,
                },
                "model": {"model_id": "MACE-MH-0", "selected_head": "mp_pbe_refit_add"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "quality_report.json").write_text(json.dumps({"ok": True, "status": "ok", "failures": []}), encoding="utf-8")
    (run_dir / "analysis_preview.json").write_text(json.dumps({"n_atoms": 3140}), encoding="utf-8")
    (run_dir / "system_manifest.yaml").write_text(
        yaml.safe_dump({"system_id": "PE100_N50_C10_emc_seed1", "components": ["PE"], "target_temperature_K": 523.0}),
        encoding="utf-8",
    )
    (run_dir / "md_log.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"phase": "production", "time_fs": 0.0, "temperature_K": 520.0}),
                json.dumps({"phase": "production", "time_fs": 1000.0, "temperature_K": 526.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    positions = np.zeros((2, 2, 3), dtype=np.float32)
    positions[1, :, 0] = 1.0
    np.savez_compressed(run_dir / "trajectory_unwrapped.npz", positions_A=positions, times_fs=np.array([0.0, 1000.0]))

    run_summary = summarize_run(run_dir)
    assert run_summary["ok"] is True
    assert run_summary["post_processing_targets"]["msd"]["available"] is True
    batch_summary = summarize_batch(tmp_path / "batch")
    assert batch_summary["n_runs"] == 1
    assert batch_summary["pure_melt_validation"]["ready_for_binary_mixtures"] is False
