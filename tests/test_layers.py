import numpy as np
import pytest
import torch

import espnn.layers as L


def test_conv2d_forward_matches_torch():
    torch.manual_seed(0)
    conv = L.Conv2d(3, 16, 3, stride=1, padding=1)
    x = torch.randn(1, 3, 8, 8)
    y = conv(x)
    assert y.shape == (1, 16, 8, 8)
    ref = torch.nn.functional.conv2d(x, conv.weight.data, conv.bias.data, padding=1)
    assert torch.allclose(y, ref, atol=1e-5)


def test_conv2d_shape_validation():
    conv = L.Conv2d(3, 16, 3, stride=2, padding=1)
    assert conv.validate_shapes((1, 3, 8, 8)) == (1, 16, 4, 4)


def test_conv2d_invalid_dilation_rejected():
    with pytest.raises(ValueError, match="dilation"):
        L.Conv2d(3, 16, 3, dilation=2)


def test_conv2d_odd_padding_rejected():
    with pytest.raises(ValueError):
        L.Conv2d(3, 16, 3, padding=1.5)


def test_depthwise_conv2d():
    torch.manual_seed(0)
    dw = L.DepthwiseConv2d(16, 32, 3, stride=1, padding=1, multiplier=2)
    x = torch.randn(1, 16, 8, 8)
    y = dw(x)
    assert y.shape == (1, 32, 8, 8)
    ref = torch.nn.functional.conv2d(
        x, dw.weight.data, dw.bias.data, padding=1, groups=16
    )
    assert torch.allclose(y, ref, atol=1e-5)


def test_linear_forward():
    torch.manual_seed(0)
    fc = L.Linear(64, 32)
    x = torch.randn(8, 64)
    y = fc(x)
    assert y.shape == (8, 32)
    ref = torch.nn.functional.linear(x, fc.weight.data, fc.bias.data)
    assert torch.allclose(y, ref, atol=1e-5)


def test_linear_rejects_non_mult8_output():
    with pytest.raises(ValueError, match="multiple of 8"):
        L.Linear(64, 30)


def test_linear_rejects_non_mult8_input():
    with pytest.raises(ValueError, match="multiple of 8"):
        L.Linear(60, 32)


def test_pool_shapes():
    assert L.MaxPool2d(2).validate_shapes((1, 8, 32, 32)) == (1, 8, 16, 16)
    assert L.AvgPool2d(2, stride=2).validate_shapes((1, 8, 30, 30)) == (1, 8, 15, 15)
    assert L.MaxPool2d(2, padding=1).validate_shapes((1, 8, 32, 32)) == (1, 8, 17, 17)


def test_pool_forward_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16, 16)
    p = L.MaxPool2d(2)
    assert torch.allclose(p(x), torch.nn.functional.max_pool2d(x, 2), atol=1e-6)
    a = L.AvgPool2d(2)
    assert torch.allclose(a(x), torch.nn.functional.avg_pool2d(x, 2), atol=1e-6)


def test_activations_forward():
    x = torch.tensor([[-2.0, -0.5, 0.5, 10.0]])
    assert torch.allclose(L.ReLU6()(x), torch.clamp(x, 0, 6))
    assert torch.allclose(L.ReLU()(x), torch.clamp(x, 0))
    assert torch.allclose(L.HardSwish()(x), torch.nn.functional.hardswish(x))
    assert torch.allclose(L.Sigmoid()(x), torch.sigmoid(x))
    assert torch.allclose(L.Softmax()(x), torch.softmax(x, dim=-1))


def test_flatten_mean_add_mul():
    x = torch.randn(1, 3, 4, 4)
    fl = L.Flatten()
    assert fl.validate_shapes((1, 3, 4, 4)) == (1, 48)
    assert torch.allclose(fl(x), x.reshape(1, 48))
    assert torch.allclose(L.Mean()(x), x.mean(dim=(2, 3), keepdim=True))
    assert torch.allclose(L.Add(constant=1.0)(x), x + 1.0)
    assert torch.allclose(L.Mul(constant=2.0)(x), x * 2.0)


def test_layers_are_nn_modules():
    for layer in [
        L.Conv2d(3, 16, 3),
        L.DepthwiseConv2d(16, 16, 3),
        L.Linear(64, 32),
        L.MaxPool2d(2),
        L.AvgPool2d(2),
        L.ReLU6(),
        L.HardSwish(),
        L.Sigmoid(),
        L.Softmax(),
        L.Flatten(),
    ]:
        assert isinstance(layer, torch.nn.Module)


def test_conv2d_weight_run():
    torch.manual_seed(1)
    conv = L.Conv2d(1, 4, 1)
    x = torch.randn(1, 1, 8, 8)
    y = conv(x)
    assert list(y.shape) == [1, 4, 8, 8]
    weights = conv.weight.data.detach().cpu().numpy()
    assert weights.dtype == np.float32
