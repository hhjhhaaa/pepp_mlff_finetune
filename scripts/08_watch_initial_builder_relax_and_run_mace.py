#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_ROOT = Path("/home/jinhao/mlff/pepp_initial_builder/data/structure_library/mlff_direct/emc_polymer")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def metadata_ready(path: Path) -> bool:
    metadata = load_yaml(path)
    paths = metadata.get("paths", {})
    extxyz = paths.get("mlff_start_extxyz")
    data = paths.get("mlff_start_lammps_data")
    return (
        metadata.get("status") == "available_relaxed"
        and metadata.get("structure_task", {}).get("lane") == "mlff_direct"
        and metadata.get("relaxation", {}).get("lammps_thermal_relax_performed") is True
        and bool(extxyz)
        and bool(data)
        and Path(str(extxyz)).exists()
        and Path(str(data)).exists()
    )


def metadata_paths(args: argparse.Namespace) -> list[Path]:
    if args.metadata_yaml:
        return [Path(item).expanduser().resolve() for item in args.metadata_yaml]
    return [Path(args.library_root).expanduser().resolve() / system_id / "metadata.yaml" for system_id in args.system_id]


def run_batch_for_metadata(path: Path, args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "07_run_initial_builder_mace_batch.py"),
        "--metadata-yaml",
        str(path),
        "--batch-id",
        args.batch_id,
        "--replica-id",
        args.replica_id,
        "--stop-on-failure",
        "--fail-unready",
    ]
    if args.overwrite:
        command.append("--overwrite")
    print(json.dumps({"event": "run_batch", "metadata_yaml": str(path), "command": command}), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def wait_and_run(args: argparse.Namespace) -> int:
    status = 0
    for path in metadata_paths(args):
        print(json.dumps({"event": "wait_start", "metadata_yaml": str(path)}), flush=True)
        while not metadata_ready(path):
            print(json.dumps({"event": "not_ready", "metadata_yaml": str(path), "sleep_s": args.poll_seconds}), flush=True)
            time.sleep(args.poll_seconds)
        print(json.dumps({"event": "ready", "metadata_yaml": str(path)}), flush=True)
        return_code = run_batch_for_metadata(path, args)
        if return_code != 0:
            status = return_code
            if args.stop_on_failure:
                break
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch initial_builder relaxed metadata and run MACE batch per system.")
    parser.add_argument("--library-root", default=str(DEFAULT_LIBRARY_ROOT))
    parser.add_argument("--metadata-yaml", action="append")
    parser.add_argument(
        "--system-id",
        action="append",
        default=["PE100_N50_C10_emc_seed1", "PP100_N30_C10_emc_seed1", "PS100_N12_C16_emc_seed1"],
    )
    parser.add_argument("--batch-id", default="pilot_20260706_initial_builder_mace_mh0")
    parser.add_argument("--replica-id", default="replica_0001")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    return wait_and_run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
