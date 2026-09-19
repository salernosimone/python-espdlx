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
        L.Gemm(64, 32, alpha=0.5, beta=2.0),
        L.Resize(scale_factor=2.0, mode="bilinear"),
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
        L.MatMul(),
        L.LayerNorm(8),
        L.RMSNorm(8),
        L.Transpose((0, 1)),
        L.Reshape((4,)),
        L.Squeeze(0),
        L.Unsqueeze(0),
        L.Slice(starts=[0], ends=[1]),
        L.Gather(indices=[0]),
        L.Pad(padding=(0, 0)),
        L.Split(2),
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


def test_concat_forward_and_shapes():
    torch.manual_seed(0)
    x, y = torch.randn(2, 3, 4, 4), torch.randn(2, 5, 4, 4)
    z = torch.randn(2, 2, 4, 4)
    cat = L.Concat(dim=1)
    assert torch.allclose(cat(x, y, z), torch.cat([x, y, z], dim=1))
    # dim=-1 (channel-last style) works too
    a, b = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 8)
    assert L.Concat(dim=-1)(a, b).shape == (2, 3, 4, 12)
    assert cat.validate_shapes((2, 3, 4, 4), (2, 5, 4, 4), (2, 2, 4, 4)) == (2, 10, 4, 4)
    with pytest.raises(ValueError, match=">= 2"):
        cat(x)
    with pytest.raises(ValueError, match="non-axis dims"):
        cat.validate_shapes((2, 3, 4, 4), (2, 5, 4, 8))
    with pytest.raises(ValueError, match="same rank"):
        cat.validate_shapes((2, 3, 4, 4), (2, 5, 4))
    with pytest.raises(ValueError, match="out of range"):
        L.Concat(dim=7).validate_shapes((1, 2, 3, 3), (1, 2, 3, 3))


def test_concat_input_indices_validation():
    assert L.Concat(input_indices=[0, -1]).input_indices == [0, -1]
    with pytest.raises(ValueError, match="empty"):
        L.Concat(input_indices=[])


def test_resize_forward_and_shapes():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8, 8)
    assert torch.allclose(
        L.Resize(scale_factor=2.0)(x), torch.nn.functional.interpolate(x, scale_factor=2.0)
    )
    assert L.Resize(scale_factor=2.0).validate_shapes((1, 3, 8, 8)) == (1, 3, 16, 16)
    bil = L.Resize(scale_factor=(2.0, 0.5), mode="bilinear")
    assert bil(x).shape == (1, 3, 16, 4)
    assert L.Resize(size=(64, 32)).validate_shapes((1, 3, 8, 8)) == (1, 3, 64, 32)
    assert torch.allclose(
        L.Resize(size=16)(x),
        torch.nn.functional.interpolate(x, size=16, mode="nearest"),
    )
    bic = L.Resize(scale_factor=2.0, mode="bicubic")
    assert torch.allclose(
        bic(x), torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bicubic", align_corners=False)
    )
    assert L.Resize(scale_factor=2.0).onnx_mode == "nearest"
    assert L.Resize(scale_factor=2.0, mode="bilinear").onnx_mode == "linear"
    with pytest.raises(ValueError, match="scale_factor or size"):
        L.Resize()
    with pytest.raises(ValueError, match="exclusive"):
        L.Resize(scale_factor=2.0, size=4)
    with pytest.raises(ValueError, match="mode"):
        L.Resize(scale_factor=2.0, mode="area")
    with pytest.raises(ValueError, match="align_corners"):
        L.Resize(scale_factor=2.0, mode="nearest", align_corners=True)
    with pytest.raises(ValueError, match="positive"):
        L.Resize(scale_factor=-1.0)


def _onnx_gemm_reference(gemm, x):
    """Float reference matching ONNX Gemm semantics: alpha*A'B' + beta*C."""
    w = gemm.weight if not gemm.transB else gemm.weight.t()
    y = gemm.alpha * (x @ w)
    if gemm.bias is not None:
        y = y + gemm.beta * gemm.bias
    return y


def test_gemm_forward_parity():
    torch.manual_seed(0)
    for transB, alpha, beta in [
        (True, 1.0, 1.0),   # Linear parity
        (False, 1.0, 1.0),
        (True, 0.5, 2.0),
        (False, 0.5, 2.0),
        (True, 2.0, 0.0),
    ]:
        gemm = L.Gemm(12, 5, alpha=alpha, beta=beta, transB=transB)
        x = torch.randn(3, 12)
        assert torch.allclose(gemm(x), _onnx_gemm_reference(gemm, x), atol=1e-6), (
            transB,
            alpha,
            beta,
        )
        assert gemm.validate_shapes((3, 12)) == (3, 5)
    # batched inputs (leading dims) compute elementwise the same way
    gemm = L.Gemm(8, 4, alpha=0.25, beta=1.5, transB=False)
    xb = torch.randn(2, 5, 8)
    assert torch.allclose(gemm(xb), _onnx_gemm_reference(gemm, xb), atol=1e-6)


def test_gemm_bias_none_and_transA_rejected():
    gemm = L.Gemm(6, 3, bias=False)
    assert gemm.bias is None
    x = torch.randn(2, 6)
    assert torch.allclose(gemm(x), x @ gemm.weight.t())
    assert L.Gemm(6, 3, bias=False).validate_shapes((2, 6)) == (2, 3)
    with pytest.raises(ValueError, match="transA"):
        L.Gemm(6, 3, transA=True)
    with pytest.raises(ValueError, match=">= 1"):
        L.Gemm(0, 3)
    with pytest.raises(ValueError, match="input features"):
        L.Gemm(6, 3).validate_shapes((2, 5))


def test_gemm_matches_linear_plain():
    torch.manual_seed(0)
    lin = L.Linear(16, 8)
    gemm = L.Gemm(16, 8, transB=True)  # default = Linear semantics
    x = torch.randn(4, 16)
    assert gemm(x).shape == lin(x).shape
    # same math once weights are shared
    gemm.weight.data.copy_(lin.weight.data)
    gemm.bias.data.copy_(lin.bias.data)
    assert torch.allclose(gemm(x), lin(x), atol=1e-6)


def test_matmul_forward_and_shapes():
    torch.manual_seed(0)
    mm = L.MatMul()
    a, b = torch.randn(2, 4, 8), torch.randn(2, 8, 6)
    assert torch.allclose(mm(a, b), a @ b)
    assert mm.validate_shapes((2, 4, 8), (2, 8, 6)) == (2, 4, 6)
    # broadcast batch dims (unbatched weight shared across the batch)
    xb, wb = torch.randn(2, 3, 4, 8), torch.randn(8, 6)
    assert torch.allclose(mm(xb, wb), xb @ wb)
    assert mm.validate_shapes((2, 3, 4, 8), (8, 6)) == (2, 3, 4, 6)
    # skip / pair wiring metadata
    assert L.MatMul(input_index=0).input_index == 0
    assert L.MatMul(input_indices=[0, 2]).input_indices == [0, 2]
    with pytest.raises(ValueError, match="two tensors"):
        mm(a)
    with pytest.raises(ValueError, match="need both shapes"):
        mm.validate_shapes((2, 4, 8))
    with pytest.raises(ValueError, match="inner dims"):
        mm.validate_shapes((2, 4, 8), (2, 7, 6))
    with pytest.raises(ValueError, match="broadcast"):
        mm.validate_shapes((2, 4, 8), (3, 8, 6))
    with pytest.raises(ValueError, match="rank >= 2"):
        mm.validate_shapes((8,), (8, 6))
    with pytest.raises(ValueError, match="exactly 2"):
        L.MatMul(input_indices=[0])
    with pytest.raises(ValueError, match="exclusive"):
        L.MatMul(input_index=0, input_indices=[0, 1])


def test_layernorm_parity():
    torch.manual_seed(0)
    ln = L.LayerNorm(8)
    x = torch.randn(2, 4, 8)
    ref = torch.nn.LayerNorm(8)
    ref.weight.data.copy_(ln.weight.data)
    ref.bias.data.copy_(ln.bias.data)
    assert torch.allclose(ln(x), ref(x), atol=1e-6)
    assert ln.validate_shapes((2, 4, 8)) == (2, 4, 8)
    # multi-dim normalized shape + no bias
    ln2 = L.LayerNorm((4, 8), bias=False)
    assert ln2.bias is None
    x2 = torch.randn(2, 4, 8)
    assert torch.allclose(ln2(x2), torch.nn.functional.layer_norm(x2, (4, 8), ln2.weight, None), atol=1e-6)
    assert ln2.validate_shapes((2, 4, 8)) == (2, 4, 8)
    with pytest.raises(ValueError, match="trailing dims"):
        ln.validate_shapes((2, 4, 7))
    with pytest.raises(ValueError, match="normalized_shape"):
        L.LayerNorm(0)
    with pytest.raises(ValueError, match="normalized_shape"):
        L.LayerNorm((8, 0))


def test_rmsnorm_parity():
    torch.manual_seed(0)
    rms = L.RMSNorm(8)
    x = torch.randn(2, 4, 8)
    var = (x * x).mean(dim=-1, keepdim=True)
    ref = x / torch.sqrt(var + rms.eps) * rms.weight.data
    assert torch.allclose(rms(x), ref, atol=1e-6)
    assert rms.validate_shapes((2, 4, 8)) == (2, 4, 8)
    # multi-dim trailing shape reduces over the last k dims
    rms2 = L.RMSNorm((4, 8))
    x2 = torch.randn(2, 4, 8)
    var2 = (x2 * x2).mean(dim=(-2, -1), keepdim=True)
    assert torch.allclose(rms2(x2), x2 / torch.sqrt(var2 + rms2.eps) * rms2.weight.data, atol=1e-6)
    assert rms2.validate_shapes((2, 4, 8)) == (2, 4, 8)
    # reduction uses negative dims (so the exporter emits negative ReduceMean
    # axes — what ESP-PPQ's RMSNorm import fusion validates)
    assert rms._reduce_dims(3) == (-1,)
    assert rms2._reduce_dims(3) == (-2, -1)
    with pytest.raises(ValueError, match="trailing dims"):
        rms.validate_shapes((2, 4, 7))
    with pytest.raises(ValueError, match="normalized_shape"):
        L.RMSNorm(-3)


def test_transpose_forward_and_shapes():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    assert torch.allclose(L.Transpose((0, 2, 1))(x), x.permute(0, 2, 1))
    assert L.Transpose((0, 2, 1)).validate_shapes((1, 4, 8)) == (1, 8, 4)
    # negative entries count from the last axis
    assert L.Transpose((0, -1, 1)).validate_shapes((1, 4, 8)) == (1, 8, 4)
    assert L.Transpose((0, -1, 1))(x).shape == (1, 8, 4)
    with pytest.raises(ValueError, match="permutation"):
        L.Transpose((0, 1, 1)).validate_shapes((1, 4, 8))
    with pytest.raises(ValueError, match="permutation"):
        L.Transpose((0, 1)).validate_shapes((1, 4, 8))


def test_reshape_per_sample():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    r = L.Reshape((4, 2, 4))
    assert torch.allclose(r(x), x.reshape(2, 4, 2, 4))
    assert r.validate_shapes((2, 4, 8)) == (2, 4, 2, 4)
    # batch flows through (calibration batch-N vs device batch-1 agree)
    assert r(torch.randn(5, 4, 8)).shape == (5, 4, 2, 4)
    # -1 infers, 0 copies the sample dim
    assert L.Reshape((-1, 4)).validate_shapes((3, 4, 8)) == (3, 8, 4)
    assert L.Reshape((0, 8)).validate_shapes((3, 4, 8)) == (3, 4, 8)
    assert torch.allclose(L.Reshape((-1, 4))(x), x.reshape(2, 8, 4))
    with pytest.raises(ValueError, match="does not match"):
        L.Reshape((3, 3)).validate_shapes((2, 4, 8))
    with pytest.raises(ValueError, match="cannot infer"):
        L.Reshape((-1, 3)).validate_shapes((2, 4, 8))
    with pytest.raises(ValueError, match="at most one -1"):
        L.Reshape((-1, -1))
    with pytest.raises(ValueError, match="must not be empty"):
        L.Reshape(())


def test_squeeze_unsqueeze():
    x = torch.randn(2, 1, 4, 8)
    assert torch.allclose(L.Squeeze(1)(x), x.squeeze(1))
    assert L.Squeeze(1).validate_shapes((2, 1, 4, 8)) == (2, 4, 8)
    assert L.Squeeze().validate_shapes((2, 1, 4, 1)) == (2, 4)
    assert L.Unsqueeze(1)(torch.randn(2, 4, 8)).shape == (2, 1, 4, 8)
    assert L.Unsqueeze(-1).validate_shapes((2, 4, 8)) == (2, 4, 8, 1)
    with pytest.raises(ValueError, match="not 1"):
        L.Squeeze(0).validate_shapes((2, 1, 4, 8))
    with pytest.raises(ValueError, match="out of range"):
        L.Unsqueeze(5).validate_shapes((2, 4, 8))


def test_slice_forward_and_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    s = L.Slice(starts=[1], ends=[3], axes=[1])
    assert torch.allclose(s(x), x[:, 1:3])
    assert s.validate_shapes((2, 4, 8)) == (2, 2, 8)
    # negative bounds count from the end, out-of-range clamps (ONNX semantics)
    assert L.Slice(starts=[-2], ends=[10], axes=[1]).validate_shapes((2, 4, 8)) == (2, 2, 8)
    assert torch.allclose(L.Slice(starts=[-2], ends=[10], axes=[1])(x), x[:, 2:4])
    stepped = L.Slice(starts=[0], ends=[4], axes=[1], steps=[2])
    assert stepped.validate_shapes((2, 4, 8)) == (2, 2, 8)
    assert torch.allclose(stepped(x), x[:, 0:4:2])
    with pytest.raises(ValueError, match="pair up"):
        L.Slice(starts=[0, 1], ends=[2], axes=[0, 1])
    with pytest.raises(ValueError, match="positive"):
        L.Slice(starts=[0], ends=[2], steps=[0])
    with pytest.raises(ValueError, match="unique"):
        L.Slice(starts=[0, 0], ends=[1, 1], axes=[1, 1])


def test_gather_forward_and_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    g = L.Gather(indices=[0, 2], dim=1)
    assert torch.allclose(g(x), torch.index_select(x, 1, torch.tensor([0, 2])))
    assert g.validate_shapes((2, 4, 8)) == (2, 2, 8)
    # binary override for manual wiring
    assert torch.allclose(L.Gather(indices=[0], dim=0)(x, [1]), x[[1]])
    with pytest.raises(ValueError, match="empty"):
        L.Gather(indices=[], dim=1)
    with pytest.raises(ValueError, match="non-negative"):
        L.Gather(indices=[-1], dim=1)
    with pytest.raises(ValueError, match="out of range"):
        L.Gather(indices=[0, 9], dim=1).validate_shapes((2, 4, 8))


def test_pad_forward_and_shapes():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4)
    p = L.Pad(padding=(0, 0, 1, 1))
    assert torch.allclose(p(x), torch.nn.functional.pad(x, (0, 0, 1, 1)))
    assert p.validate_shapes((1, 2, 4, 4)) == (1, 2, 6, 4)
    v = L.Pad(padding=(2, 0), value=0.5)
    assert torch.allclose(v(x), torch.nn.functional.pad(x, (2, 0), value=0.5))
    assert v.validate_shapes((1, 2, 4, 4)) == (1, 2, 4, 6)
    with pytest.raises(ValueError, match="pairs"):
        L.Pad(padding=(1, 2, 3))
    with pytest.raises(ValueError, match="non-negative"):
        L.Pad(padding=(-1, 0))
    with pytest.raises(ValueError, match="exceed rank"):
        L.Pad(padding=(0, 0, 0, 0, 0, 0)).validate_shapes((1, 2))


def test_split_forward_and_shapes():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 4)
    parts = L.Split(3, dim=1)(x)
    assert isinstance(parts, tuple) and len(parts) == 3
    for got, want in zip(parts, torch.split(x, 2, dim=1)):
        assert torch.allclose(got, want)
    assert L.Split(3, dim=1).validate_shapes((1, 6, 4)) == ((1, 2, 4),) * 3
    sized = L.Split([1, 2, 3], dim=1)
    assert sized.validate_shapes((1, 6, 4)) == ((1, 1, 4), (1, 2, 4), (1, 3, 4))
    assert L.Split(3, dim=1).multi_output is True
    with pytest.raises(ValueError, match="divisible"):
        L.Split(4, dim=1).validate_shapes((1, 6, 4))
    with pytest.raises(ValueError, match="sum to"):
        L.Split([1, 2], dim=1).validate_shapes((1, 6, 4))
    with pytest.raises(ValueError, match=">= 2"):
        L.Split(1, dim=1)
