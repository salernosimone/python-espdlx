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


def _dense():
    return espdlx.Model(
        [
            L.Conv2d(2, 4, 3, padding=1),  # 0: h0
            L.ReLU(),                       # 1: h1
            L.Conv2d(4, 4, 3, padding=1),  # 2
            L.ReLU(),                       # 3: h3
            L.Concat(input_indices=[0, 1, 3], dim=1),  # 4: cat(h0, h1, h3)
            L.Conv2d(12, 2, 1),            # 5
            L.ReLU(),
        ],
        name="dense",
    )


def test_model_concat_matches_manual():
    torch.manual_seed(0)
    model = _dense()
    x = torch.randn(1, 2, 8, 8)
    l = model.layers
    h0 = l[0](x)
    h1 = l[1](h0)
    h2 = l[2](h1)
    h3 = l[3](h2)
    cat = torch.cat([h0, h1, h3], dim=1)
    expected = torch.relu(l[5](cat))
    assert torch.allclose(model(x), expected)
    assert model.validate_shapes((1, 2, 8, 8)) == (1, 2, 8, 8)


def test_model_concat_order_is_explicit():
    # input_indices order is authoritative: cat([h2, h1, h0]) not cat([h0, h1, h2])
    torch.manual_seed(0)
    model = espdlx.Model(
        [
            L.Conv2d(2, 4, 3, padding=1),  # 0
            L.ReLU(),                        # 1
            L.Conv2d(4, 4, 3, padding=1),   # 2
            L.ReLU(),                        # 3
            L.Concat(input_indices=[3, 1, 0], dim=1),  # 4
            L.Conv2d(12, 2, 1),              # 5
            L.ReLU(),
        ],
        name="dense_rev",
    )
    x = torch.randn(1, 2, 8, 8)
    l = model.layers
    h0 = l[0](x)
    h1 = l[1](h0)
    h2 = l[2](h1)
    h3 = l[3](h2)
    cat = torch.cat([h3, h1, h0], dim=1)  # the REVERSED order from input_indices
    expected = torch.relu(l[5](cat))
    assert torch.allclose(model(x), expected)


def test_model_concat_uses_previous_layer_index():
    # dense-style: concat of prior layer outputs (the running block-in stream is
    # layer 1's saved output; the raw model input is not part of `saved`)
    torch.manual_seed(0)
    model = espdlx.Model(
        [
            L.Conv2d(2, 4, 3, padding=1),  # 0: h0 = f(x)
            L.ReLU(),                       # 1
            L.Concat(input_indices=[0, 1], dim=1),  # 2: cat(h0, h1)
        ],
        name="dense_block",
    )
    x = torch.randn(1, 2, 8, 8)
    h0 = model.layers[0](x)
    h1 = model.layers[1](h0)
    assert torch.allclose(model(x), torch.cat([h0, h1], dim=1))
    assert model.validate_shapes((1, 2, 8, 8)) == (1, 8, 8, 8)


def test_model_concat_out_of_range_and_shape_mismatch():
    model = espdlx.Model([L.ReLU(), L.Concat(input_indices=[5, 1], dim=1)], name="bad")
    with pytest.raises(ValueError, match="out-of-range"):
        model(torch.zeros(1, 2, 2, 2))
    with pytest.raises(ValueError, match="out-of-range"):
        model.validate_shapes((1, 2, 2, 2))
    mismatch = espdlx.Model(
        [L.Conv2d(2, 4, 1), L.MaxPool2d(2), L.Concat(input_indices=[0, 1], dim=1)],
        name="mismatch",  # (1,4,4,4) vs (1,4,2,2)
    )
    with pytest.raises(ValueError, match="non-axis dims"):
        mismatch.validate_shapes((1, 2, 4, 4))


def _attention_pair():
    # Q @ K^T via saved pair, then weights @ V via saved pair — the
    # attention primitive in a sequential Model.
    return espdlx.Model(
        [
            L.Linear(8, 8),                  # 0: Q (1,4,8)
            L.Linear(8, 8),                  # 1: K (1,4,8)
            L.Transpose((0, 2, 1)),          # 2: K^T (1,8,4)
            L.MatMul(input_indices=[0, 2]),  # 3: scores (1,4,4)
            L.Softmax(dim=-1),               # 4: weights
            L.Linear(4, 8),                  # 5: V (1,4,8)
            L.MatMul(input_indices=[4, 5]),  # 6: out = weights @ V
            L.LayerNorm(8),                  # 7
            L.RMSNorm(8),                    # 8
        ],
        name="attention",
    )


def test_model_attention_pair_matches_manual():
    torch.manual_seed(0)
    model = _attention_pair()
    x = torch.randn(1, 4, 8)
    l = model.layers
    q = l[0](x)
    k = l[1](q)
    kt = k.permute(0, 2, 1)
    scores = q @ kt
    weights = torch.softmax(scores, dim=-1)
    v = l[5](weights)
    expected = l[8](l[7](weights @ v))
    assert torch.allclose(model(x), expected, atol=1e-6)
    assert model.validate_shapes((1, 4, 8)) == (1, 4, 8)


def test_model_matmul_input_index():
    # (1,4,8) @ (1,4,8) mismatches inner dims (8 vs 4) — must fail fast
    bad = espdlx.Model(
        [
            L.Linear(8, 8),          # 0: (1,4,8)
            L.Softmax(dim=-1),       # 1: (1,4,8)
            L.MatMul(input_index=0),  # 2: cur @ saved[0]
        ],
        name="mm_skip_bad",
    )
    with pytest.raises(ValueError, match="inner dims"):
        bad.validate_shapes((1, 4, 8))

    good = espdlx.Model(
        [
            L.Linear(8, 8),          # 0: (1,4,8)
            L.Transpose((0, 2, 1)),  # 1: (1,8,4)
            L.MatMul(input_index=0),  # 2: cur @ saved[0] = (1,8,4)@(1,4,8)
        ],
        name="mm_skip",
    )
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    l = good.layers
    h0 = l[0](x)
    h1 = l[1](h0)
    assert torch.allclose(good(x), h1 @ h0)
    assert good.validate_shapes((1, 4, 8)) == (1, 8, 8)


def test_model_head_split_merge_roundtrip():
    torch.manual_seed(0)
    model = espdlx.Model(
        [
            L.Reshape((4, 2, 4)),       # (B,4,8) -> (B,4,2,4)
            L.Transpose((0, 2, 1, 3)),  # -> (B,2,4,4)
            L.Transpose((0, 2, 1, 3)),  # -> back
            L.Reshape((4, 8)),          # -> (B,4,8)
        ],
        name="heads",
    )
    x = torch.randn(1, 4, 8)
    assert torch.allclose(model(x), x)
    assert model.validate_shapes((1, 4, 8)) == (1, 4, 8)


def test_model_slice_decomposition_equals_split():
    # Split is multi-output (manual wiring only); the Model-chain equivalent
    # is one Slice per section over the same runtime Slice op.
    torch.manual_seed(0)
    x = torch.randn(1, 6, 4)
    parts = L.Split(3, dim=1)(x)
    model = espdlx.Model(
        [
            L.Slice(starts=[0], ends=[2], axes=[1]),
        ],
        name="one_slice",
    )
    slices = [
        L.Slice(starts=[s], ends=[s + 2], axes=[1])(x) for s in (0, 2, 4)
    ]
    for got, want in zip(slices, parts):
        assert torch.allclose(got, want)
    assert model.validate_shapes((1, 6, 4)) == (1, 2, 4)


def test_model_split_rejected_with_slice_pointer():
    model = espdlx.Model([L.ReLU(), L.Split(2, dim=1)], name="bad")
    with pytest.raises(ValueError, match="Slice"):
        model(torch.zeros(1, 4, 4))
    with pytest.raises(ValueError, match="Slice"):
        model.validate_shapes((1, 4, 4))
