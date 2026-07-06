# Workflow

1. Put CP2K outputs or symlinks under `data/raw_cp2k`.
2. Create `manifests/cp2k_patches.csv` from the example manifest.
3. Build extxyz once `cp2k_reader.py` has a real parser for the selected CP2K output format.
4. Run dataset checks and create interpolation/extrapolation splits.
5. Check local MACE-MH checkpoint and installed MACE CLI.
6. Evaluate the foundation model on validation/test extxyz before fine-tuning.
7. Dry-run the MACE fine-tuning command, then run explicitly.
8. Validate with static errors and reserved short-rollout/structure gates.
