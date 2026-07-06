#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ase.io import read


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def find_lammps(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        shutil.which("lmp"),
        shutil.which("lammps"),
        "/home/jinhao/software/lammps/build-cmake/lmp",
        "/public/home/jinhao.hu/envs/lammps/bin/lmp",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return Path(candidate).expanduser()
    raise FileNotFoundError("LAMMPS executable not found; pass --lammps.")


def find_obabel(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        shutil.which("obabel"),
        "/home/jinhao/miniforge3/envs/pepp-graph-spib/bin/obabel",
        "/home/jinhao/miniconda3/envs/SurfDock/bin/obabel",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return Path(candidate).expanduser()
    raise FileNotFoundError("Open Babel executable not found; pass --obabel.")


def lammps_box(data_path: Path) -> tuple[float, float, float]:
    bounds: dict[str, float] = {}
    for raw_line in data_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.split()
        if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
            bounds["x"] = float(parts[1]) - float(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
            bounds["y"] = float(parts[1]) - float(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ["zlo", "zhi"]:
            bounds["z"] = float(parts[1]) - float(parts[0])
    if set(bounds) != {"x", "y", "z"}:
        raise ValueError(f"Could not parse orthorhombic box from {data_path}")
    return bounds["x"], bounds["y"], bounds["z"]


def patch_extxyz_cell(path: Path, box: tuple[float, float, float]) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Open Babel output is not a valid XYZ/exyz file: {path}")
    lattice = f'{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}'
    lines[1] = f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"'
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_xyz_with_obabel(obabel: Path, src_xyz: Path, dst_extxyz: Path, box: tuple[float, float, float]) -> None:
    completed = subprocess.run(
        [str(obabel), "-ixyz", str(src_xyz), "-oexyz", "-O", str(dst_extxyz)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Open Babel conversion failed:\n{completed.stdout}\n{completed.stderr}")
    patch_extxyz_cell(dst_extxyz, box)


def thermo_block(every: int) -> str:
    return f"""thermo {every}
thermo_style custom step temp density press pe ke etotal vol lx ly lz
thermo_modify lost error flush yes
"""


def forcefield_block(data_file: str) -> str:
    return f"""units real
atom_style full
boundary p p p
read_data {data_file}
include input.params
if "${{flag_charged}} != 0" then "kspace_style pppm/cg 1.0e-4"
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
"""


def write_inputs(out_dir: Path, args: argparse.Namespace) -> list[Path]:
    scripts: list[Path] = []
    common = {
        "timestep": args.timestep_fs,
        "thermo": args.thermo_every,
        "tdamp": args.tdamp_fs,
        "pdamp": args.pdamp_fs,
        "initial_temp": args.initial_temperature_K,
        "target_temp": args.target_temperature_K,
        "anneal_temp": args.anneal_temperature_K,
        "pressure": args.pressure_atm,
    }

    p0 = out_dir / "in.00_minimize.lmp"
    p0.write_text(
        forcefield_block("input.data")
        + thermo_block(common["thermo"])
        + f"""min_style fire
minimize {args.minimize_etol} {args.minimize_ftol} {args.minimize_maxiter} {args.minimize_maxeval}
write_data stage00_minimized.data nocoeff
write_dump all xyz stage00_minimized.xyz modify sort id
""",
        encoding="utf-8",
    )
    scripts.append(p0)

    p1 = out_dir / "in.01_nve_limit_warmup.lmp"
    p1.write_text(
        forcefield_block("stage00_minimized.data")
        + thermo_block(common["thermo"])
        + f"""velocity all create {common["initial_temp"]:.3f} {args.velocity_seed} mom yes rot yes dist gaussian
timestep {common["timestep"]:.6f}
fix temp all langevin {common["initial_temp"]:.3f} {common["target_temp"]:.3f} {common["tdamp"]:.3f} {args.langevin_seed}
fix int all nve/limit {args.nve_limit_A}
run {args.warmup_steps}
unfix int
unfix temp
write_data stage01_warmup.data nocoeff
write_dump all xyz stage01_warmup.xyz modify sort id
""",
        encoding="utf-8",
    )
    scripts.append(p1)

    p2 = out_dir / "in.02_nvt_anneal.lmp"
    p2.write_text(
        forcefield_block("stage01_warmup.data")
        + thermo_block(common["thermo"])
        + f"""timestep {common["timestep"]:.6f}
fix mom all momentum 200 linear 1 1 1 angular
fix int all nvt temp {common["target_temp"]:.3f} {common["anneal_temp"]:.3f} {common["tdamp"]:.3f}
run {args.nvt_heat_steps}
unfix int
fix int all nvt temp {common["anneal_temp"]:.3f} {common["target_temp"]:.3f} {common["tdamp"]:.3f}
run {args.nvt_cool_steps}
unfix int
unfix mom
write_data stage02_nvt_annealed.data nocoeff
write_dump all xyz stage02_nvt_annealed.xyz modify sort id
""",
        encoding="utf-8",
    )
    scripts.append(p2)

    p3 = out_dir / "in.03_npt_density.lmp"
    p3.write_text(
        forcefield_block("stage02_nvt_annealed.data")
        + thermo_block(common["thermo"])
        + f"""timestep {common["timestep"]:.6f}
fix mom all momentum 200 linear 1 1 1 angular
fix int all npt temp {common["target_temp"]:.3f} {common["target_temp"]:.3f} {common["tdamp"]:.3f} iso {common["pressure"]:.6f} {common["pressure"]:.6f} {common["pdamp"]:.3f}
run {args.npt_steps}
unfix int
unfix mom
write_data stage03_npt_density.data nocoeff
write_dump all xyz stage03_npt_density.xyz modify sort id
""",
        encoding="utf-8",
    )
    scripts.append(p3)

    p4 = out_dir / "in.04_nvt_settle.lmp"
    p4.write_text(
        forcefield_block("stage03_npt_density.data")
        + thermo_block(common["thermo"])
        + f"""timestep {common["timestep"]:.6f}
fix mom all momentum 200 linear 1 1 1 angular
fix int all nvt temp {common["target_temp"]:.3f} {common["target_temp"]:.3f} {common["tdamp"]:.3f}
dump traj all custom {args.dump_every} thermal_relax.lammpstrj id mol type q x y z
dump_modify traj sort id
run {args.nvt_settle_steps}
unfix int
unfix mom
write_data relaxed.data nocoeff
write_dump all xyz relaxed.xyz modify sort id
""",
        encoding="utf-8",
    )
    scripts.append(p4)
    return scripts


def run_lammps(lammps: Path, script: Path, timeout_s: int) -> None:
    completed = subprocess.run(
        [str(lammps), "-in", script.name],
        cwd=script.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
    )
    (script.parent / f"{script.stem}.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"LAMMPS failed for {script.name}; see {script.stem}.log")


def parse_last_thermo(log_path: Path) -> dict[str, float]:
    header: list[str] = []
    last: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts[:2] == ["Step", "Temp"]:
            header = parts
            continue
        if header and len(parts) >= len(header):
            try:
                values = [float(item) for item in parts[: len(header)]]
            except ValueError:
                continue
            last = dict(zip(header, values))
    return last


def choose_source_files(system_dir: Path) -> tuple[Path, Path, Path]:
    manifest = load_yaml(system_dir / "system_manifest.yaml")
    builder = manifest.get("builder", {})
    data_path = Path(builder.get("lammps_data") or "")
    params_path = Path(builder.get("params") or "")
    if not data_path.exists():
        candidates = sorted(system_dir.glob("*.data"))
        data_path = candidates[0] if candidates else Path()
    if not params_path.exists():
        candidates = sorted(system_dir.glob("*.params"))
        params_path = candidates[0] if candidates else Path()
    if not data_path.exists() or not params_path.exists():
        raise FileNotFoundError("EMC LAMMPS data and params are required for thermal relaxation.")
    return data_path, params_path, system_dir / "system_manifest.yaml"


def thermal_relax(args: argparse.Namespace) -> Path:
    system_dir = Path(args.emc_system_dir).expanduser().resolve()
    manifest = load_yaml(system_dir / "system_manifest.yaml")
    if manifest.get("builder", {}).get("builder_used") != "emc" or manifest.get("builder", {}).get("emc_success") is not True:
        raise ValueError("Thermal relaxation requires a verified EMC system_manifest.yaml.")

    lammps = find_lammps(args.lammps)
    obabel = find_obabel(args.obabel)
    data_path, params_path, manifest_path = choose_source_files(system_dir)

    out_dir = system_dir / "lammps_thermal_relax"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_path, out_dir / "input.data")
    shutil.copy2(params_path, out_dir / "input.params")
    scripts = write_inputs(out_dir, args)

    status = "planned_only"
    if args.run:
        status = "ok"
        for script in scripts:
            run_lammps(lammps, script, args.timeout_seconds)
        final_box = lammps_box(out_dir / "relaxed.data")
        convert_xyz_with_obabel(obabel, out_dir / "relaxed.xyz", out_dir / "relaxed.extxyz", final_box)
        read(out_dir / "relaxed.extxyz")
    else:
        final_box = lammps_box(out_dir / "input.data")

    final_thermo = parse_last_thermo(out_dir / "in.04_nvt_settle.log") if (out_dir / "in.04_nvt_settle.log").exists() else {}
    protocol = {
        "name": "emc_polymer_lammps_thermal_relax_v1",
        "basis": "pepp_initial_builder polymer cleanup naming plus literature-standard minimize/NVT/NPT/NVT equilibration",
        "is_training_data": False,
        "is_production_md": False,
        "timestep_fs": args.timestep_fs,
        "temperature_K": args.target_temperature_K,
        "anneal_temperature_K": args.anneal_temperature_K,
        "pressure_atm": args.pressure_atm,
        "tdamp_fs": args.tdamp_fs,
        "pdamp_fs": args.pdamp_fs,
        "steps": {
            "warmup_nve_limit": args.warmup_steps,
            "nvt_heat": args.nvt_heat_steps,
            "nvt_cool": args.nvt_cool_steps,
            "npt_density": args.npt_steps,
            "nvt_settle": args.nvt_settle_steps,
        },
        "format_converter": "openbabel",
        "obabel_executable": str(obabel),
        "lammps_executable": str(lammps),
    }
    summary = {
        "ok": status == "ok" or not args.run,
        "status": status,
        "system_id": manifest.get("system_id"),
        "output_dir": str(out_dir),
        "relaxed_data": str(out_dir / "relaxed.data") if args.run else "",
        "relaxed_extxyz": str(out_dir / "relaxed.extxyz") if args.run else "",
        "final_box_A": list(final_box),
        "final_thermo": final_thermo,
        "protocol": protocol,
    }
    (out_dir / "thermal_relax_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.run:
        manifest.setdefault("preprocess_history", []).append(
            {
                "operation": "lammps_thermal_relax",
                "protocol": protocol["name"],
                "output_dir": str(out_dir),
                "relaxed_data": str(out_dir / "relaxed.data"),
                "relaxed_extxyz": str(out_dir / "relaxed.extxyz"),
                "classical_md_is_label_source": False,
            }
        )
        manifest["mlff_start_structure"] = {
            "stage": "emc_lammps_thermal_relaxed",
            "extxyz": str(out_dir / "relaxed.extxyz"),
            "lammps_data": str(out_dir / "relaxed.data"),
        }
        dump_yaml(manifest_path, manifest)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAMMPS thermal relaxation for imported EMC polymer systems.")
    parser.add_argument("--emc-system-dir", required=True)
    parser.add_argument("--lammps", default=None)
    parser.add_argument("--obabel", default=None)
    parser.add_argument("--run", action="store_true", help="Actually execute LAMMPS; otherwise only write inputs.")
    parser.add_argument("--initial-temperature-K", type=float, default=300.0)
    parser.add_argument("--target-temperature-K", type=float, default=523.0)
    parser.add_argument("--anneal-temperature-K", type=float, default=650.0)
    parser.add_argument("--pressure-atm", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--tdamp-fs", type=float, default=100.0)
    parser.add_argument("--pdamp-fs", type=float, default=1000.0)
    parser.add_argument("--velocity-seed", type=int, default=4928459)
    parser.add_argument("--langevin-seed", type=int, default=91023)
    parser.add_argument("--nve-limit-A", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--nvt-heat-steps", type=int, default=50000)
    parser.add_argument("--nvt-cool-steps", type=int, default=50000)
    parser.add_argument("--npt-steps", type=int, default=200000)
    parser.add_argument("--nvt-settle-steps", type=int, default=100000)
    parser.add_argument("--minimize-etol", default="1.0e-6")
    parser.add_argument("--minimize-ftol", default="1.0e-8")
    parser.add_argument("--minimize-maxiter", type=int, default=10000)
    parser.add_argument("--minimize-maxeval", type=int, default=20000)
    parser.add_argument("--thermo-every", type=int, default=1000)
    parser.add_argument("--dump-every", type=int, default=10000)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    return parser.parse_args()


def main() -> int:
    thermal_relax(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
