import pytest
import torch

from espdlx.fomo import (
    decode,
    decode_peaks,
    encode_centroids,
    encode_heatmap,
    init_bias_from_prior,
    soft_loss,
    weighted_loss,
)
from espdlx.zoo import MobileNetV2


def test_fomo_head_is_logits():
    torch.manual_seed(0)
    model = MobileNetV2.Fomo()
    out = model(torch.randn(2, 1, 96, 96))
    assert out.shape == (2, 2, 12, 12)
    # raw logits: must NOT already sum to 1 across channels
    assert not torch.allclose(
        out.softmax(dim=1).sum(dim=1),
        torch.full_like(out[:, 0], 2.0),
    )


def test_fomo_encode_loss_decode_roundtrip():
    torch.manual_seed(1)
    model = MobileNetV2.Fomo()
    tgt = encode_centroids([(0.5, 0.5, 0.2, 0.2)], grid=12)
    assert tgt.sum().item() == 1 and tgt.shape == (12, 12)
    assert encode_centroids([(0.5, 0.5, 0.0, 0.2)], grid=12).sum().item() == 0

    x = torch.randn(2, 1, 96, 96)
    tgts = tgt[None].expand(2, -1, -1).contiguous()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = weighted_loss(model(x), tgts).item()
    for _ in range(5):
        opt.zero_grad()
        weighted_loss(model(x), tgts).backward()
        opt.step()
    assert weighted_loss(model(x), tgts).item() < l0

    # decode: hot object logit decodes to its cell
    logits = torch.full((1, 2, 12, 12), -10.0)
    logits[0, 1, 6, 6] = 10.0
    dets = decode(logits, thresh=0.5)[0]
    assert dets == [(6, 6, dets[0][2])] and dets[0][2] > 0.99
    assert decode(logits, thresh=0.99999999)[0] == []


def test_fomo_bias_init_matches_prior():
    import math

    torch.manual_seed(2)
    model = MobileNetV2.Fomo()
    init_bias_from_prior(model, 1.0 / 144.0)
    out = model(torch.zeros(1, 1, 96, 96))
    prob = out.softmax(dim=1)[0, 1]
    assert torch.allclose(
        prob, torch.full_like(prob, 1.0 / 144.0), atol=5e-3
    ), prob.mean().item()
    # exact value check on the bias itself
    from espdlx.layers.conv import Conv2d as EspConv2d

    head = [lyr for lyr in model.layers if isinstance(lyr, EspConv2d)][-1]
    assert head.conv.bias[1].item() == pytest.approx(
        math.log((1 / 144) / (1 - 1 / 144)), abs=1e-6
    )


def _fire(G, *cells):
    l = torch.empty(1, 2, G, G)
    l[0, 0] = 5.0
    l[0, 1] = -5.0
    for i, j in cells:
        l[0, :, i, j] = torch.tensor([-5.0, 5.0])
    return l


def test_heatmap_peak_and_decay():
    h = encode_heatmap([(0.5, 0.5, 0.2, 0.2)], 12)
    assert h.shape == (12, 12) and h.max().item() == pytest.approx(1.0)
    assert h[6, 7].item() == pytest.approx(0.6065, abs=1e-3)  # adjacent
    assert h[0, 0].item() == 0.0  # far
    assert encode_heatmap([(0.5, 0.5, 0.0, 0.2)], 12).sum().item() == 0.0


def test_soft_loss_tolerates_near_miss():
    G = 12
    tgt = encode_centroids([(0.5, 0.5, 0.2, 0.2)], G)[None]
    heat = encode_heatmap([(0.5, 0.5, 0.2, 0.2)], G)[None]
    base = soft_loss(_fire(G, (6, 6)), tgt, heat).item()
    near = soft_loss(_fire(G, (6, 6), (6, 7)), tgt, heat).item() - base
    far = soft_loss(_fire(G, (6, 6), (0, 0)), tgt, heat).item() - base
    assert near < 0.3 and far > 9.0  # ~40x tolerance for roughly-right
    # old loss charges both equally
    o_near = weighted_loss(_fire(G, (6, 6), (6, 7)), tgt).item()
    o_far = weighted_loss(_fire(G, (6, 6), (0, 0)), tgt).item()
    assert o_near == pytest.approx(o_far)


def test_soft_loss_decreases_and_flows():
    torch.manual_seed(7)
    G = 12
    model = MobileNetV2.Fomo(grid=G)
    x = torch.randn(2, 1, 96, 96)
    tgt = encode_centroids([(0.5, 0.5, 0.2, 0.2)], G)[None].expand(2, -1, -1)
    heat = encode_heatmap([(0.5, 0.5, 0.2, 0.2)], G)[None].expand(2, -1, -1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = soft_loss(model(x), tgt, heat).item()
    for _ in range(5):
        opt.zero_grad()
        soft_loss(model(x), tgt, heat).backward()
        opt.step()
    assert soft_loss(model(x), tgt, heat).item() < l0


def test_decode_peaks_merges_bump():
    G = 12
    l = torch.full((1, 2, G, G), -5.0)
    l[0, 1, 6, 6] = 10.0
    l[0, 1, 6, 7] = 9.0
    l[0, 1, 7, 6] = 8.0
    dets = decode_peaks(l, thresh=0.5)[0]
    assert [(c, r) for c, r, _ in dets] == [(6, 6)]  # one centroid, not three
    assert len(decode(l, thresh=0.5)[0]) == 3  # plain decode keeps all
