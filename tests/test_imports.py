def test_import_package_and_modules():
    import pepp_mlff
    import pepp_mlff.analysis.mace_md_summary
    import pepp_mlff.config.load_config
    import pepp_mlff.data.check_dataset
    import pepp_mlff.data.split_dataset
    import pepp_mlff.io.cp2k_reader
    import pepp_mlff.io.extxyz_writer
    import pepp_mlff.models.pretrained_mace
    import pepp_mlff.training.mace_finetune
    import pepp_mlff.validation.evaluate_static

    assert pepp_mlff.__version__ == "0.1.0"
