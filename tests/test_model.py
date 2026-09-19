import pytest
import torch

import espdlx
import espdlx.layers as L


def _conv_mlp():
    return espdlx.Model(
        [
            L.Conv2d(1, 8, 3, padding=1),
            L.ReLU6(),
            L.MaxPool2d(2),
            L.Conv2d(8, 8, 3, padding=1),
            L.ReLU6(),
            L.Flatten(),
            L.Linear(128, 16),
            L.Softmax(),
        ],
        name="conv_mlp",
    )


def test_model_forward_shape():
    torch.manual_seed(0)
    model = _conv_mlp()
    out = model(torch.randn(1, 1, 8, 8))
    assert out.shape == (1, 16)


def test_validate_shapes():
    assert _conv_mlp().validate_shapes((1, 1, 8, 8)) == (1, 16)


def test_model_rejects_non_layer():
    with pytest.raises(TypeError):
        espdlx.Model([L.Conv2d(1, 8, 3), torch.nn.Conv2d(8, 8, 3)])


def _residual():
    return espdlx.Model(
        [
            L.Conv2d(2, 2, 1),   # 0
            L.ReLU(),            # 1
            L.Conv2d(2, 2, 1),   # 2
            L.Add(input_index=0),  # 3: out2 + out0
            L.ReLU(),
        ],
        name="residual",
    )


def test_model_skip_connection_matches_manual():
    torch.manual_seed(0)
    model = _residual()
    x = torch.randn(1, 2, 4, 4)
    c0, relu, c1 = model.layers[0], model.layers[1], model.layers[2]
    expected = torch.relu(c1(relu(c0(x))) + c0(x))
    assert torch.allclose(model(x), expected)
    assert model.validate_shapes((1, 2, 4, 4)) == (1, 2, 4, 4)


def test_model_skip_negative_index():
    torch.manual_seed(0)
    model = espdlx.Model(
        [L.Conv2d(2, 2, 1), L.ReLU(), L.Add(input_index=-2)],
        name="neg",
    )
    x = torch.randn(1, 2, 4, 4)
    c0 = model.layers[0]
    assert torch.allclose(model(x), torch.relu(c0(x)) + c0(x))


def test_model_skip_out_of_range():
    model = espdlx.Model([L.ReLU(), L.Add(input_index=5)], name="bad")
    with pytest.raises(ValueError, match="out of range"):
        model(torch.zeros(1, 2, 2, 2))
    with pytest.raises(ValueError, match="out of range"):
        model.validate_shapes((1, 2, 2, 2))


def test_model_skip_shape_mismatch():
    model = espdlx.Model(
        [L.Conv2d(2, 4, 1), L.MaxPool2d(2), L.Add(input_index=0)],
        name="mismatch",  # (1,4,2,2) + (1,4,4,4)
    )
    with pytest.raises(ValueError, match="must match"):
        model.validate_shapes((1, 2, 4, 4))
    with pytest.raises(RuntimeError):
        model(torch.zeros(1, 2, 4, 4))
