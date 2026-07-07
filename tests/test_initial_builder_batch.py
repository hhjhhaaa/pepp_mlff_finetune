from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_run_initial_builder_mace_batch.py"
SPEC = spec_from_file_location("initial_builder_batch", SCRIPT)
batch = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(batch)


def test_metadata_ready_requires_relaxed_mlff_direct():
    metadata = {
        "status": "available_emc_built",
        "structure_task": {"lane": "mlff_direct"},
        "relaxation": {"lammps_thermal_relax_performed": False},
        "paths": {"mlff_start_extxyz": "polymer.extxyz"},
    }
    assert batch.metadata_is_ready(metadata, require_relaxed=True) is False
    metadata["status"] = "available_relaxed"
    metadata["relaxation"]["lammps_thermal_relax_performed"] = True
    metadata["paths"]["mlff_start_extxyz"] = "relaxed.extxyz"
    assert batch.metadata_is_ready(metadata, require_relaxed=True) is True


def test_md_command_uses_imported_topology_manifest_and_mace_mh0():
    args = Namespace(
        batch_id="batch",
        replica_id="replica_0001",
        model_config="configs/model/mace_mh0.yaml",
        temperature_K=523.0,
        timestep_fs=0.25,
        stabilization_ps=1.0,
        production_ps=2.0,
        nve_audit_ps=0.5,
        default_dtype="float32",
        overwrite=True,
    )
    command = batch.md_command(
        "PS100",
        {"components": ["PS"], "target_temperature_K": 523.0},
        args,
    )
    assert "scripts/05_run_local_mace_mh_md.py" in command[1]
    assert "--topology" in command
    assert "topology.json" in command[command.index("--topology") + 1]
    assert "--system-manifest" in command
    assert "--model-config" in command
    assert command[command.index("--model-config") + 1] == "configs/model/mace_mh0.yaml"
    assert "--overwrite" in command


def test_merge_manifest_rows_preserves_other_systems(tmp_path: Path):
    manifest = tmp_path / "initial_builder_batch_manifest.json"
    manifest.write_text(
        '[{"system_id": "PE100", "status": "ok"}, {"system_id": "PP100", "status": "md_failed"}]\n',
        encoding="utf-8",
    )
    rows = batch.merge_manifest_rows(manifest, [{"system_id": "PP100", "status": "ok"}])
    assert rows == [
        {"system_id": "PE100", "status": "ok"},
        {"system_id": "PP100", "status": "ok"},
    ]


def test_run_config_populates_batch_arguments(tmp_path: Path):
    run_config = tmp_path / "batch.yaml"
    run_config.write_text(
        "\n".join(
            [
                "batch_id: batch2a",
                "replica_id: replica_x",
                "model_config: configs/model/mace_mh0.yaml",
                "stabilization_ps: 2.0",
                "production_ps: 5.0",
                "nve_audit_ps: 0.5",
                "metadata_yamls:",
                "  - /tmp/a/metadata.yaml",
                "fail_unready: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = Namespace(
        run_config=str(run_config),
        metadata_yaml=None,
        batch_id="old",
        replica_id="old_replica",
        model_config="old.yaml",
        temperature_K=523.0,
        timestep_fs=0.25,
        stabilization_ps=1.0,
        production_ps=2.0,
        nve_audit_ps=0.0,
        default_dtype="float32",
        overwrite=False,
        allow_unrelaxed=False,
        fail_unready=False,
        stop_on_failure=False,
        dry_run=False,
        library_root="/tmp/library",
    )
    updated = batch.apply_run_config(args)
    assert updated.batch_id == "batch2a"
    assert updated.replica_id == "replica_x"
    assert updated.production_ps == 5.0
    assert updated.metadata_yaml == ["/tmp/a/metadata.yaml"]
    assert updated.fail_unready is True
