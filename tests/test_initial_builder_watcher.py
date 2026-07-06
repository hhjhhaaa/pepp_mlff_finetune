from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08_watch_initial_builder_relax_and_run_mace.py"
SPEC = spec_from_file_location("initial_builder_watcher", SCRIPT)
watcher = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(watcher)


def test_metadata_ready_requires_relaxed_files(tmp_path: Path):
    extxyz = tmp_path / "relaxed.extxyz"
    data = tmp_path / "relaxed.data"
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        "\n".join(
            [
                "status: available_relaxed",
                "structure_task:",
                "  lane: mlff_direct",
                "relaxation:",
                "  lammps_thermal_relax_performed: true",
                "paths:",
                f"  mlff_start_extxyz: {extxyz}",
                f"  mlff_start_lammps_data: {data}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert watcher.metadata_ready(metadata) is False
    extxyz.write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    data.write_text("LAMMPS data\n", encoding="utf-8")
    assert watcher.metadata_ready(metadata) is True
