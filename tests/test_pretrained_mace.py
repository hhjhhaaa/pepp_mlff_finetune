from pathlib import Path

from pepp_mlff.models import pretrained_mace


def test_mace_select_head_falls_back_to_python_bin(monkeypatch, tmp_path: Path):
    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir()
    fake_python.write_text("", encoding="utf-8")
    fake_select = fake_python.parent / "mace_select_head"
    fake_select.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(pretrained_mace.shutil, "which", lambda name: None)
    monkeypatch.setattr(pretrained_mace.sys, "executable", str(fake_python))
    assert pretrained_mace.mace_select_head_executable() == str(fake_select)
