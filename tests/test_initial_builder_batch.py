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
