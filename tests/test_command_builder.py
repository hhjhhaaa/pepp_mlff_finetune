from pepp_mlff.training.mace_finetune import build_mace_finetune_command


def test_build_mace_finetune_command_minimal():
    config = {
        "name": "test",
        "foundation_model": "models/pretrained/mace_mh_1.model",
        "foundation_head": "mp_pbe_refit_add",
        "train_file": "data/splits/train.extxyz",
        "test_file": "data/splits/test.extxyz",
        "device": "cpu",
        "E0s": {"mode": "average_debug"},
        "energy_weight": 0.1,
        "forces_weight": 1.0,
        "multiheads_finetuning": False,
    }
    command = build_mace_finetune_command(config)
    assert command[0] == "mace_run_train"
    assert "--foundation_model" in command
    assert "--foundation_head" in command
    assert "--train_file" in command
    assert "--test_file" in command
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cpu"
    assert "--E0s" in command
    assert "average" in command
