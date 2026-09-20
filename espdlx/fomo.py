"""EXPERIMENTAL FOMO helpers: the two things Edge Impulse does better.

Status: synthetic unit tests only (`tests/test_fomo.py`); best real-data
run so far reached P=0.03/R=0.28 — not a working detector yet.

Kept deliberately small (no parity chase):

- logits-only training (softmax at export/decode, never in the model),
- imbalance handling: object-weighted cross-entropy + classifier bias init
  from the dataset foreground prior.

Targets are per-cell class maps ``(N, G, G)`` with ``0`` = background,
``1`` = object centroid cell (see :func:`encode_centroids`).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "encode_centroids",
    "encode_heatmap",
    "weighted_loss",
    "soft_loss",
    "init_bias_from_prior",
    "decode",
    "decode_peaks",
]


def encode_centroids(
    boxes: list[tuple[float, ...]],
    grid: int,
) -> torch.Tensor:
    """Map normalized boxes to a ``(G, G)`` class map.

    Boxes are ``(cx, cy, w, h)`` — or ``(cx, cy, w, h, cls)`` with a 1-based
    class id — and the cell containing each centroid is marked with the class
    id (``1`` for plain boxes); everything else stays ``0`` = background.
    Zero/negative-area boxes are skipped.
    """
    target = torch.zeros(grid, grid, dtype=torch.long)
    for box in boxes:
        cx, cy, w, h = box[:4]
        cls = int(box[4]) if len(box) > 4 else 1
        if w <= 0 or h <= 0:
            continue
        j = min(max(int(math.floor(cx * grid)), 0), grid - 1)
        i = min(max(int(math.floor(cy * grid)), 0), grid - 1)
        target[i, j] = max(cls, 1)
    return target


def weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    object_weight: float = 100.0,
) -> torch.Tensor:
    """Weighted cross-entropy over ``(N, C+1, G, G)`` logits.

    Background (class 0) weight is ``1.0``; every foreground class weight
    defaults to ``object_weight`` (Edge Impulse default 100) to counter the
    ~1:143 foreground/background ratio on a 12x12 grid.
    """
    n_cls = logits.shape[1]
    weight = torch.ones(n_cls, device=logits.device)
    weight[1:] = float(object_weight)
    return F.cross_entropy(logits, targets.long(), weight=weight)


def encode_heatmap(
    boxes: list[tuple[float, float, float, float]],
    grid: int,
    sigma: float = 1.0,
    radius: int = 2,
) -> torch.Tensor:
    """Splat normalized ``(cx, cy, w, h)`` boxes into a ``(G, G)`` heatmap.

    Each centroid stamps a Gaussian bump (peak 1 at the exact centroid,
    ``exp(-d^2 / 2*sigma^2)`` in cell units, ``radius`` cells wide);
    overlapping bumps merge by max. Zero/negative-area boxes are skipped.
    Pairs with :func:`soft_loss` / :func:`decode_peaks`.
    """
    heat = torch.zeros(grid, grid)
    for box in boxes:
        cx, cy, w, h = box[:4]
        if w <= 0 or h <= 0:
            continue
        gx, gy = cx * grid, cy * grid
        jc, ic = int(math.floor(gx)), int(math.floor(gy))
        for i in range(max(ic - radius, 0), min(ic + radius, grid - 1) + 1):
            for j in range(max(jc - radius, 0), min(jc + radius, grid - 1) + 1):
                d2 = (gx - j) ** 2 + (gy - i) ** 2
                v = math.exp(-d2 / (2 * sigma * sigma))
                if v > heat[i, j]:
                    heat[i, j] = v
    return heat


def soft_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    heatmaps: torch.Tensor,
    object_weight: float = 20.0,
    gamma: float = 4.0,
) -> torch.Tensor:
    """Localization-tolerant loss over ``(N, C+1, G, G)`` logits.

    - positives (centroid cells from :func:`encode_centroids`): NLL toward
      the *cell's own class* logit, scaled by ``object_weight``;
    - negatives: NLL toward the background channel (class 0), scaled by
      ``(1 - heat)^gamma`` with ``heat`` from :func:`encode_heatmap` — an
      adjacent cell (heat ~0.6) pays ~2% of a far cell's penalty, so "roughly
      right" barely hurts.
    Normalized by the positive-cell count (min 1), so the scale is stable
    across images with different object counts.
    """
    logp = F.log_softmax(logits, dim=1)
    cls = targets.long().clamp_min(0).clamp_max(logits.shape[1] - 1)
    pos = cls > 0
    n_pos = max(int(pos.sum().item()), 1)
    onehot = F.one_hot(cls, logits.shape[1]).permute(0, 3, 1, 2).float()
    cell_nll = -(logp * onehot).sum(dim=1)              # (N, G, G)
    loss_pos = cell_nll[pos].sum() * float(object_weight)
    w = (1.0 - heatmaps.float()).pow(float(gamma))
    loss_neg = -(w * logp[:, 0])[~pos].sum()
    return (loss_pos + loss_neg) / n_pos


def heatmap_loss(
    logits: torch.Tensor,
    heatmaps: torch.Tensor,
    object_weight: float = 50.0,
) -> torch.Tensor:
    """CenterNet-style heatmap regression over ``(N, 2, G, G)`` logits.

    Per-cell cross-entropy toward the Gaussian bump (from
    :func:`encode_heatmap`) as a SOFT target: the model must reproduce the
    bump *shape*, so the peak lands on the encoded centroid — hard CE with a
    tolerated neighborhood lets the bump drift to the visually strongest
    body part, which on size-varying objects turns into FP+FN pairs at
    decode (diagnosed on cat5k: 63% of memorized train GTs had zero peaks
    under :func:`soft_loss`).

    Per cell: ``-W·h·log p_obj - (1-h)·log p_bg``; normalized by ``sum(h)``
    so scale stays stable across images with different object counts.
    Softmax is never applied here (logits in, as everywhere in training).
    """
    if logits.shape[1] != 2:
        raise NotImplementedError(
            "heatmap_loss: multi-class (C+1) needs per-class heatmaps; "
            "single-class (2-channel) only for now"
        )
    logp = F.log_softmax(logits, dim=1)
    h = heatmaps.float().clamp(0.0, 1.0)
    loss = -(float(object_weight) * h * logp[:, 1] + (1.0 - h) * logp[:, 0]).sum()
    return loss / h.sum().clamp_min(1.0)


def init_bias_from_prior(model: torch.nn.Module, obj_prior: float,
                         head_channels: int = 2) -> None:
    """Init the final head bias so the model starts at the dataset prior.

    2-channel head: ``bias = [0, log(p / (1 - p))]`` makes the initial object
    probability equal the foreground cell fraction ``p`` (Edge Impulse's
    ``set_classifier_biases_from_dataset`` idea, reduced to one number).
    C+1 head: the object prior is spread uniformly over the C classes,
    ``bias_c = log((p/C) / (1 - p/C))``. Finds the LAST ``espdlx`` conv whose
    ``out_channels == head_channels`` in ``model.layers``.
    """
    from .layers.conv import Conv2d as EspConv2d

    p = min(max(float(obj_prior), 1e-6), 1.0 - 1e-6)
    head = None
    for lyr in getattr(model, "layers", []):
        if isinstance(lyr, EspConv2d) and lyr.out_channels == head_channels:
            head = lyr
    if head is None:
        raise ValueError(
            f"init_bias_from_prior: no {head_channels}-channel head conv found")
    with torch.no_grad():
        bias = head.conv.bias
        assert bias is not None and bias.numel() == head_channels
        bias.zero_()
        if head_channels == 2:
            bias[1] = math.log(p / (1.0 - p))
        else:
            pc = p / (head_channels - 1)
            bias[1:] = math.log(pc / (1.0 - pc))


@torch.no_grad()
def decode(
    logits: torch.Tensor,
    thresh: float = 0.5,
) -> list[list[tuple]]:
    """Logits ``(N, C+1, G, G)`` -> per image decoded centroids.

    Softmax is applied here (or at export), never during training.
    C+1 == 2 (single class): ``[(col, row, score)]`` with ``score`` = object
    probability. Multi-class: ``[(col, row, score, cls)]`` with ``cls`` the
    0-based argmax class.
    """
    prob = torch.softmax(logits, dim=1)
    out: list[list[tuple]] = []
    multi = logits.shape[1] > 2
    for n in range(logits.shape[0]):
        dets: list[tuple] = []
        conf, idx = prob[n].max(dim=0)
        keep = (idx > 0) & (conf > thresh) if multi else prob[n, 1] > thresh
        for i, j in keep.nonzero(as_tuple=False).tolist():
            score = float(conf[i, j] if multi else prob[n, 1, i, j])
            det = (int(j), int(i), score)
            if multi:
                det += (int(idx[i, j]) - 1,)
            dets.append(det)
        out.append(dets)
    return out


@torch.no_grad()
def decode_peaks(
    logits: torch.Tensor,
    thresh: float = 0.5,
) -> list[list[tuple]]:
    """Peak-NMS decode: 3x3 max-pool keeps local maxima above ``thresh``.

    Pairs with :func:`soft_loss` — the loss lets bumps spread over adjacent
    cells, this collapses each bump back to one centroid. Single-class (2
    channels) returns ``[(col, row, score)]``; multi-class adds the 0-based
    class id: ``[(col, row, score, cls)]``.
    """
    prob = torch.softmax(logits, dim=1)
    out: list[list[tuple]] = []
    multi = logits.shape[1] > 2
    if not multi:
        p = prob[:, 1]
        is_peak = F.max_pool2d(p[:, None], 3, stride=1, padding=1)[:, 0] == p
        for n in range(logits.shape[0]):
            dets: list[tuple] = []
            for i, j in ((p[n] > thresh) & is_peak[n]).nonzero(as_tuple=False).tolist():
                dets.append((int(j), int(i), float(p[n, i, j])))
            out.append(dets)
        return out
    for n in range(logits.shape[0]):
        dets: list[tuple] = []
        conf, idx = prob[n].max(dim=0)
        keep = (idx > 0) & (conf > thresh)
        is_peak = F.max_pool2d(conf[None, None], 3, stride=1, padding=1)[0, 0] == conf
        keep = keep & is_peak
        for i, j in keep.nonzero(as_tuple=False).tolist():
            det = (int(j), int(i), float(conf[i, j]), int(idx[i, j]) - 1)
            dets.append(det)
        out.append(dets)
    return out
