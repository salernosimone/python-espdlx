def test_import_espnn():
    import espnn

    assert espnn.__version__
    assert hasattr(espnn, "Model")
