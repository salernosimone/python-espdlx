import pytest
import torch

import espnn
import espnn.layers as L


def _conv_mlp():
    return espnn.Model(
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
        espnn.Model([L.Conv2d(1, 8, 3), torch.nn.Conv2d(8, 8, 3)])
