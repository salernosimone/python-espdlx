def test_import_espdlx():
    import espdlx

    assert espdlx.__version__
    assert hasattr(espdlx, "Model")
