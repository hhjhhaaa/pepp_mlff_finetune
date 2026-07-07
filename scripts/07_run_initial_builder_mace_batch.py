#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def apply_run_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.run_config:
        return args
    config = load_yaml(Path(args.run_config).expanduser().resolve())
    aliases = {
        "metadata_yamls": "metadata_yaml",
        "temperature_K": "temperature_K",
        "timestep_fs": "timestep_fs",
        "stabilization_ps": "stabilization_ps",
        "production_ps": "production_ps",
        "nve_audit_ps": "nve_audit_ps",
        "default_dtype": "default_dtype",
        "model_config": "model_config",
        "batch_id": "batch_id",
        "replica_id": "replica_id",
        "overwrite": "overwrite",
        "allow_unrelaxed": "allow_unrelaxed",
        "fail_unready": "fail_unready",
        "stop_on_failure": "stop_on_failure",
        "dry_run": "dry_run",
        "library_root": "library_root",
    }
    for key, attr in aliases.items():
        if key not in config:
            continue
        value = config[key]
        if attr == "metadata_yaml":
            value = [str(item) for item in value]
        setattr(args, attr, value)
    return args


def discover_metadata(args: argparse.Namespace) -> list[Path]:
    if args.metadata_yaml:
        return [Path(item).expanduser().resolve() for item in args.metadata_yaml]
    library_root = Path(args.library_root).expanduser().resolve()
    return sorted(library_root.glob("*/metadata.yaml"))


def metadata_is_ready(metadata: dict[str, Any], require_relaxed: bool) -> bool:
    if metadata.get("structure_task", {}).get("lane") != "mlff_direct":
        return False
    if not require_relaxed:
        return True
    return (
        metadata.get("status") == "available_relaxed"
        and metadata.get("relaxation", {}).get("lammps_thermal_relax_performed") is True
        and bool(metadata.get("paths", {}).get("mlff_start_extxyz"))
    )


def import_command(metadata_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "06_import_emc_system.py"),
        "--metadata-yaml",
        str(metadata_path),
    ]


def md_command(system_id: str, metadata: dict[str, Any], args: argparse.Namespace) -> list[str]:
    imported_dir = ROOT / "data" / "emc_systems" / system_id
    components = ",".join(str(item).upper() for item in metadata.get("components", []))
    command = [
        sys.executable,
        str(ROOT / "scripts" / "05_run_local_mace_mh_md.py"),
        "--input",
        str(imported_dir / "input.extxyz"),
        "--system-id",
        system_id,
        "--components",
        components,
        "--topology",
        str(imported_dir / "topology.json"),
        "--system-manifest",
        str(imported_dir / "system_manifest.yaml"),
        "--batch-id",
        args.batch_id,
        "--replica-id",
        args.replica_id,
        "--model-config",
        args.model_config,
        "--temperature-K",
        str(float(metadata.get("target_temperature_K") or args.temperature_K)),
        "--timestep-fs",
        str(args.timestep_fs),
        "--stabilization-ps",
        str(args.stabilization_ps),
        "--production-ps",
        str(args.production_ps),
        "--nve-audit-ps",
        str(args.nve_audit_ps),
        "--default-dtype",
        args.default_dtype,
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def run_command(command: list[str], dry_run: bool) -> int:
    print(json.dumps({"command": command, "dry_run": dry_run}, sort_keys=True), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


def merge_manifest_rows(manifest: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            for row in existing:
                if not isinstance(row, dict):
                    continue
                system_id = str(row.get("system_id") or "")
                if not system_id:
                    continue
                if system_id in {str(item.get("system_id") or "") for item in rows}:
                    continue
                merged.append(row)
                seen.add(system_id)
    for row in rows:
        system_id = str(row.get("system_id") or "")
        if system_id and system_id not in seen:
            merged.append(row)
            seen.add(system_id)
    return merged


def run_batch(args: argparse.Namespace) -> int:
    rows = []
    status = 0
    for metadata_path in discover_metadata(args):
        metadata = load_yaml(metadata_path)
        system_id = str(metadata.get("system_id") or metadata_path.parent.name)
        if not metadata_is_ready(metadata, require_relaxed=not args.allow_unrelaxed):
            row = {
                "system_id": system_id,
                "metadata_yaml": str(metadata_path),
                "status": "skipped_unready",
                "reason": "requires mlff_direct available_relaxed metadata",
            }
            rows.append(row)
            if args.fail_unready:
                status = 3
            continue

        import_rc = run_command(import_command(metadata_path), args.dry_run)
        if import_rc != 0:
            rows.append({"system_id": system_id, "metadata_yaml": str(metadata_path), "status": "import_failed"})
            status = import_rc
            if args.stop_on_failure:
                break
            continue

        md_rc = run_command(md_command(system_id, metadata, args), args.dry_run)
        rows.append({
            "system_id": system_id,
            "metadata_yaml": str(metadata_path),
            "status": "dry_run" if args.dry_run else ("ok" if md_rc == 0 else "md_failed"),
            "return_code": md_rc,
        })
        if md_rc != 0:
            status = md_rc
            if args.stop_on_failure:
                break

    out_dir = ROOT / "runs" / "mace_md" / args.batch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "initial_builder_batch_manifest.json"
    manifest_rows = merge_manifest_rows(manifest, rows)
    manifest.write_text(json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"batch_manifest": str(manifest), "rows": rows}, indent=2, sort_keys=True))
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import initial_builder EMC library entries and run MACE-MH MD.")
    parser.add_argument("--run-config", default=None, help="YAML batch run config. CLI remains command-compatible.")
    parser.add_argument(
        "--library-root",
        default="/home/jinhao/mlff/pepp_initial_builder/data/structure_library/mlff_direct/emc_polymer",
    )
    parser.add_argument("--metadata-yaml", action="append")
    parser.add_argument("--batch-id", default="pilot_initial_builder_mace_mh0")
    parser.add_argument("--replica-id", default="replica_0001")
    parser.add_argument("--model-config", default="configs/model/mace_mh0.yaml")
    parser.add_argument("--temperature-K", type=float, default=523.0)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--stabilization-ps", type=float, default=1.0)
    parser.add_argument("--production-ps", type=float, default=2.0)
    parser.add_argument("--nve-audit-ps", type=float, default=0.5)
    parser.add_argument("--default-dtype", default="float32")
    parser.add_argument("--allow-unrelaxed", action="store_true")
    parser.add_argument("--fail-unready", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_batch(apply_run_config(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
