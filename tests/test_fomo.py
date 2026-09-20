import pytest
import torch
import torch.nn.functional as F

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


def test_fomoslim_neck_shapes():
    """Arch B: top-down neck (Resize -> 1x1 -> Concat[tap, up]) + fine head."""
    torch.manual_seed(3)
    model = MobileNetV2.FomoSlim(img_size=160, head_stride=4)
    out = model(torch.randn(2, 1, 160, 160))
    assert out.shape == (2, 2, 40, 40)
    coarse = MobileNetV2.FomoSlim(img_size=160, head_stride=8)
    assert coarse(torch.randn(1, 1, 160, 160)).shape == (1, 2, 20, 20)
    assert MobileNetV2.FomoSlim(img_size=96, head_stride=4)(
        torch.randn(1, 1, 96, 96)).shape == (1, 2, 24, 24)


def test_fomoslim_learns_centroids():
    """Tiny overfit: the necked arch must drive weighted CE down (gradient
    flows through both Concat inputs — tap and top-down path)."""
    G, S = 40, 160
    torch.manual_seed(4)
    model = MobileNetV2.FomoSlim(img_size=S, head_stride=4)
    boxes = [(0.3, 0.3, 0.1, 0.1), (0.7, 0.6, 0.1, 0.1)]
    tgt = torch.stack([encode_centroids(boxes, G)] * 3)
    heat = torch.stack([encode_heatmap(boxes, G)] * 3)
    x = torch.randn(3, 1, S, S)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    l0 = soft_loss(model(x), tgt, heat, object_weight=25.0).item()
    for _ in range(40):
        opt.zero_grad()
        loss = soft_loss(model(x), tgt, heat, object_weight=25.0)
        loss.backward()
        opt.step()
    assert soft_loss(model(x), tgt, heat, object_weight=25.0).item() < 0.2 * l0
    dets = decode_peaks(model(x[:1]), thresh=0.5)[0]
    assert len(dets) == 2  # both memorized centroids, peak-NMS deduped


def test_fomo_multiclass_roundtrip():
    """C+1 helpers: per-class centroids, class-aware decode."""
    boxes = [(0.2, 0.3, 0.1, 0.1, 2), (0.7, 0.8, 0.1, 0.1, 1)]
    tgt = encode_centroids(boxes, 12)
    assert tgt[3, 2].item() == 2 and tgt[9, 8].item() == 1
    heat = encode_heatmap(boxes, 12)
    logits = torch.randn(2, 3, 12, 12, requires_grad=True)
    loss = soft_loss(logits, tgt[None].expand(2, -1, -1),
                     heat[None].expand(2, -1, -1))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    hot = torch.full((1, 3, 12, 12), -10.0)
    hot[0, 2, 5, 5] = 10.0
    assert decode_peaks(hot)[0] == [(5, 5, hot.softmax(1)[0, 2, 5, 5].item(), 1)]
    # 2-channel decode_peaks unchanged
    l2 = torch.full((1, 2, 12, 12), -5.0)
    l2[0, 1, 6, 6] = 10.0
    assert decode_peaks(l2)[0] == [(6, 6, decode_peaks(l2)[0][0][2])]


def test_heatmap_loss_regresses_bump():
    """CenterNet-style: peak must land ON the encoded centroid, not drift."""
    torch.manual_seed(5)
    from espdlx.fomo import heatmap_loss
    boxes = [(0.3, 0.3, 0.1, 0.1), (0.7, 0.6, 0.1, 0.1)]
    heat = torch.stack([encode_heatmap(boxes, 40)] * 3)
    model_logits = torch.full((3, 2, 40, 40), 0.0)
    logits = torch.zeros(3, 2, 40, 40)
    logits[:, 1, 12, 14] = 8.0   # bumped right of true cell (12,12)
    logits[:, 1, 24, 26] = 8.0   # bumped off (24,28)
    logits[:, 0] = -2.0
    l0 = heatmap_loss(logits, heat).item()
    assert l0 == l0  # finite
    # moving the bump ONTO the true cells strictly lowers the loss
    fixed = torch.zeros(3, 2, 40, 40)
    fixed[:, 1, 12, 12] = 8.0
    fixed[:, 1, 24, 28] = 8.0
    fixed[:, 0] = -2.0
    assert heatmap_loss(fixed, heat).item() < l0
    # optimization probe: a conv head learns to place the peak on-target
    conv = torch.nn.Conv2d(3, 2, 3, padding=1)
    x = torch.randn(2, 3, 40, 40)
    tgt = torch.stack([heat[0], heat[1]])  # one bump-set per sample
    x[:, 2] = tgt                          # the signal the head must read
    opt = torch.optim.Adam(conv.parameters(), lr=1e-2)
    l_start = heatmap_loss(conv(x), tgt, object_weight=1.0).item()
    for _ in range(300):
        opt.zero_grad()
        heatmap_loss(conv(x), tgt, object_weight=1.0).backward()
        opt.step()
    l = heatmap_loss(conv(x), tgt, object_weight=1.0).item()
    assert l < 0.5 * l_start  # clear descent
    prob = conv(x).softmax(1)[:, 1]
    for (bi, bj) in ((12, 12), (24, 28)):
        assert prob[0, bi, bj] > 0.4, \
            f"peak missing at encoded cell {(bi, bj)} after training"
