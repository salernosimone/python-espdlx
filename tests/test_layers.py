import numpy as np
import pytest
import torch

import espdlx.layers as L


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


def test_linear_arbitrary_dims_allowed():
    # %8 rule dropped: Linear(12, 5) proven on-device (linrelax_proof 3/3 MATCH).
    torch.manual_seed(0)
    fc = L.Linear(12, 5)
    x = torch.randn(2, 12)
    assert fc(x).shape == (2, 5)
    assert torch.allclose(fc(x), torch.nn.functional.linear(x, fc.weight, fc.bias))
    assert fc.validate_shapes((2, 12)) == (2, 5)
    with pytest.raises(ValueError, match=">= 1"):
        L.Linear(0, 8)


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


def test_sub_div_binary_tensors():
    torch.manual_seed(0)
    x, y = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)
    assert torch.allclose(L.Sub()(x, y), x - y)
    assert torch.allclose(L.Sub(constant=1.0)(x), x - 1.0)
    assert torch.allclose(L.Div()(x, y), x / y)
    assert torch.allclose(L.Div(constant=2.0)(x), x / 2.0)
    assert L.Sub().validate_shapes((1, 3, 4, 4), (1, 3, 4, 4)) == (1, 3, 4, 4)
    assert L.Div().validate_shapes((1, 3, 4, 4), (1, 3, 4, 4)) == (1, 3, 4, 4)
    with pytest.raises(ValueError, match="non-zero"):
        L.Div(constant=0.0)
    with pytest.raises(ValueError, match="two tensors"):
        L.Sub()(x)
    with pytest.raises(ValueError, match="must match"):
        L.Div().validate_shapes((1, 3, 4, 4), (1, 3, 2, 2))
    with pytest.raises(ValueError, match="exclusive"):
        L.Sub(constant=1.0, input_index=0)


def test_neg_exp_log_sqrt():
    x = torch.tensor([[[[0.25, 1.0, 4.0]]]])
    assert torch.allclose(L.Neg()(x), -x)
    assert torch.allclose(L.Exp()(x), torch.exp(x))
    assert torch.allclose(L.Log()(x), torch.log(x))
    assert torch.allclose(L.Sqrt()(x), torch.sqrt(x))
    for lyr in [L.Neg(), L.Exp(), L.Log(), L.Sqrt()]:
        assert lyr.validate_shapes((1, 1, 1, 3)) == (1, 1, 1, 3)


def test_new_activations():
    x = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
    assert torch.allclose(L.LeakyReLU()(x), torch.nn.functional.leaky_relu(x, 0.01))
    assert torch.allclose(L.LeakyReLU(negative_slope=0.2)(x), torch.nn.functional.leaky_relu(x, 0.2))
    assert torch.allclose(L.Tanh()(x), torch.tanh(x))
    assert torch.allclose(L.Swish()(x), torch.nn.functional.silu(x))
    assert torch.allclose(L.Elu()(x), torch.nn.functional.elu(x))
    assert torch.allclose(L.HardSigmoid()(x), torch.nn.functional.hardsigmoid(x))
    assert torch.allclose(L.Clip()(x), torch.clamp(x, 0, 6))
    assert torch.allclose(L.Clip(min_val=-1.0, max_val=1.0)(x), torch.clamp(x, -1, 1))
    with pytest.raises(ValueError, match="max_val"):
        L.Clip(min_val=2.0, max_val=1.0)


def test_mul_binary_tensors():
    torch.manual_seed(0)
    x, y = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)
    assert torch.allclose(L.Mul()(x, y), x * y)
    assert L.Mul().validate_shapes((1, 3, 4, 4), (1, 3, 4, 4)) == (1, 3, 4, 4)
    with pytest.raises(ValueError, match="two tensors"):
        L.Mul()(x)
    with pytest.raises(ValueError, match="must match"):
        L.Mul().validate_shapes((1, 3, 4, 4), (1, 3, 2, 2))
    with pytest.raises(ValueError, match="exclusive"):
        L.Mul(constant=2.0, input_index=0)


def test_add_binary_tensors():
    torch.manual_seed(0)
    x, y = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)
    assert torch.allclose(L.Add()(x, y), x + y)
    assert L.Add().validate_shapes((1, 3, 4, 4), (1, 3, 4, 4)) == (1, 3, 4, 4)
    with pytest.raises(ValueError, match="two tensors"):
        L.Add()(x)
    with pytest.raises(ValueError, match="must match"):
        L.Add().validate_shapes((1, 3, 4, 4), (1, 3, 2, 2))
    with pytest.raises(ValueError, match="exclusive"):
        L.Add(constant=1.0, input_index=0)


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
        L.LeakyReLU(),
        L.Tanh(),
        L.Swish(),
        L.Elu(),
        L.HardSigmoid(),
        L.Clip(),
        L.Sub(constant=1.0),
        L.Div(constant=2.0),
        L.Neg(),
        L.Exp(),
        L.Log(),
        L.Sqrt(),
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
