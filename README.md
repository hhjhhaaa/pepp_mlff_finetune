# PE/PP-Silica MACE-MH Fine-Tuning

This project turns CP2K-labeled PE/PP patch, bulk, and subcell samples into a quality-gated MACE fine-tuning dataset.

Workflow:

```text
CP2K patch/bulk labels -> extxyz dataset -> MACE-MH checkpoint check -> foundation baseline -> MACE fine-tuning command -> validation -> export
```

This repository does not build initial PE/PP structures, run Packmol or LAMMPS relaxation, crop patches, train Graph-SPIB, or mine descriptors.

## Why MACE-MH

The production target includes PE/PP-silica samples, so the foundation model must cover C/H/O/Si chemistry. MACE-OFF is not the right production default for Si-containing interfaces. Use MACE-MH-1 first, with local MACE-MH-0 as the fallback checkpoint.

For formal fine-tuning, prefer local MACE-MH checkpoints:

```text
models/pretrained/mace_mh_1.model
models/pretrained/mace_mh_0.model
```

API fallback is disabled in production. Formal runs record the local checkpoint hash, `mace-torch` version, PyTorch version, CUDA availability, and selected device.

The default MACE-MH head is `mp_pbe_refit_add`, matching the first CP2K/PBE-style label route. If the CP2K label level changes, update `foundation_head` together with `label_level_id` and reference E0s.

## Install

Install PyTorch first for the local CUDA/CPU environment, then install this project:

```bash
python -m pip install -e ".[mace,dev]"
```

For code-only inspection without loading MACE:

```bash
python -m pip install -e ".[dev]"
```

This project intentionally does not pin a GPU Torch wheel because WSL and HPC CUDA versions differ.

## Data Types And Metadata

Samples must declare:

```text
capped_patch, periodic_bulk, periodic_subcell
```

The manifest also tracks `sample_type`, `pbc`, `cell_source`, `cap_type`, global/local PE/PP composition, CP2K quality settings, force-label validity, and dataset provenance.

For fixed or frozen boundaries, standard MACE fine-tuning accepts only samples with `force_label_valid=true`. If forces are valid only for mobile atoms and the installed MACE CLI lacks atom-wise force masks, mark the sample excluded or reserve it for later custom training. Per-atom `atom_role` values are:

```text
polymer_core, cap, frozen_boundary
```

## CP2K Quality Gate

The dataset check requires a `label_level_id` so data from different CP2K settings are not silently mixed. It also checks SCF/completion flags, sample type versus pbc/cell consistency, force-label validity, required metadata, and C/H/O/Si chemistry.

## Split Profiles

`interpolation` uses grouped stratification by `parent_frame_id`, while balancing density, composition, and sample type where possible. It measures interpolation inside covered density/composition regimes.

`extrapolation_density` leaves out one `density_g_cm3` group. `extrapolation_composition` leaves out one `composition` group. These splits measure transfer to held-out physical regimes and should be reported separately from interpolation metrics.

## Pretrained Checks

```bash
python scripts/00_check_pretrained_mace_off.py
python scripts/00b_check_mace_cli.py
```

Before training, evaluate the un-fine-tuned local foundation model:

```bash
python scripts/00c_evaluate_foundation_on_dataset.py
```

This writes `logs/foundation_baseline_eval.json` and does not train.

## Data Preparation

Place CP2K outputs under `data/raw_cp2k` or symlink them there. Create:

```text
manifests/cp2k_patches.csv
```

from `manifests/cp2k_patches.example.csv`.

## Fine-Tuning

Edit `configs/train/mace_finetune.yaml`, then dry-run:

```bash
bash scripts/03_finetune_mace.sh
```

Run explicitly:

```bash
bash scripts/03_finetune_mace.sh --run
```

The first phase is force-dominant:

```yaml
energy_weight: 0.1
forces_weight: 1.0
```

This is deliberate: capped patch total energies can include boundary capping and reference-energy offsets. `E0s.mode: average_debug` is for debugging only; production should use C/H/O/Si reference E0s obtained at the same CP2K label level through `configs/train/e0s_cp2k_reference.yaml`.

## Stress And Validation

`periodic_bulk` and `periodic_subcell` reserve `stress_available`, `stress_key`, `stress_unit`, and `train_stress`. If stress is unavailable, first-stage validation is fixed-volume/NVT only; do not claim NPT density-prediction quality from those runs.

Validation gates define thresholds for force RMSE, energy MAE per atom, NVE drift, bond explosions, NaN prevention, and unphysical C-C/C-H bonds. Stubs reserve interfaces for RDF, Rg, dihedrals, PE-PP contacts, short-time MSD, and density-dependent structure sanity checks.

## Tracking Policy

Do not commit CP2K outputs, large extxyz files, model weights, runs, or logs. MACE code and MACE foundation model licenses may differ; do not commit pretrained weights.
