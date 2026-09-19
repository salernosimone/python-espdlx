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
    boxes: list[tuple[float, float, float, float]],
    grid: int,
) -> torch.Tensor:
    """Map normalized ``(cx, cy, w, h)`` boxes to a ``(G, G)`` class map.

    The cell containing each centroid is marked ``1``; everything else stays
    ``0``. Zero/negative-area boxes are skipped.
    """
    target = torch.zeros(grid, grid, dtype=torch.long)
    for cx, cy, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        j = min(max(int(math.floor(cx * grid)), 0), grid - 1)
        i = min(max(int(math.floor(cy * grid)), 0), grid - 1)
        target[i, j] = 1
    return target


def weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    object_weight: float = 100.0,
) -> torch.Tensor:
    """Weighted cross-entropy over ``(N, 2, G, G)`` logits.

    Background weight is ``1.0``; the object cell weight defaults to ``100.0``
    (Edge Impulse default) to counter the ~1:143 foreground/background ratio
    on a 12x12 grid.
    """
    weight = torch.tensor([1.0, float(object_weight)], device=logits.device)
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
    for cx, cy, w, h in boxes:
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
    """Localization-tolerant loss over ``(N, 2, G, G)`` logits.

    - positives (centroid cells from :func:`encode_centroids`): NLL toward
      the object logit, scaled by ``object_weight``;
    - negatives: NLL toward background, scaled by ``(1 - heat)^gamma`` with
      ``heat`` from :func:`encode_heatmap` — an adjacent cell (heat ~0.6)
      pays ~2% of a far cell's penalty, so "roughly right" barely hurts.
    Normalized by the positive-cell count (min 1), so the scale is stable
    across images with different object counts.
    """
    logp = F.log_softmax(logits, dim=1)
    pos = targets.long() == 1
    n_pos = max(int(pos.sum().item()), 1)
    loss_pos = -logp[:, 1][pos].sum() * float(object_weight)
    w = (1.0 - heatmaps.float()).pow(float(gamma))
    loss_neg = -(w * logp[:, 0])[~pos].sum()
    return (loss_pos + loss_neg) / n_pos


def init_bias_from_prior(model: torch.nn.Module, obj_prior: float) -> None:
    """Init the final head bias so the model starts at the dataset prior.

    With a 2-logit softmax head, ``bias = [0, log(p / (1 - p))]`` makes the
    initial object probability equal the foreground cell fraction ``p``
    (Edge Impulse's ``set_classifier_biases_from_dataset`` idea, reduced to
    one number). Finds the last ``espdlx`` conv layer in ``model.layers``.
    """
    from .layers.conv import Conv2d as EspConv2d

    p = min(max(float(obj_prior), 1e-6), 1.0 - 1e-6)
    head = None
    for lyr in getattr(model, "layers", []):
        if isinstance(lyr, EspConv2d) and lyr.out_channels == 2:
            head = lyr
    if head is None:
        raise ValueError("init_bias_from_prior: no 2-channel head conv found")
    with torch.no_grad():
        bias = head.conv.bias
        assert bias is not None and bias.numel() == 2
        bias.zero_()
        bias[1] = math.log(p / (1.0 - p))


@torch.no_grad()
def decode(
    logits: torch.Tensor,
    thresh: float = 0.5,
) -> list[list[tuple[int, int, float]]]:
    """Logits ``(N, 2, G, G)`` -> per image ``[(col, row, score)]`` centroids.

    Softmax is applied here (or at export), never during training. ``score``
    is the object-channel probability.
    """
    prob = torch.softmax(logits, dim=1)[:, 1]
    out: list[list[tuple[int, int, float]]] = []
    for n in range(logits.shape[0]):
        idx = (prob[n] > thresh).nonzero(as_tuple=False)
        out.append([(int(j), int(i), float(prob[n, i, j])) for i, j in idx.tolist()])
    return out


@torch.no_grad()
def decode_peaks(
    logits: torch.Tensor,
    thresh: float = 0.5,
) -> list[list[tuple[int, int, float]]]:
    """Peak-NMS decode: 3x3 max-pool keeps local maxima above ``thresh``.

    Pairs with :func:`soft_loss` — the loss lets bumps spread over adjacent
    cells, this collapses each bump back to one centroid.
    """
    prob = torch.softmax(logits, dim=1)[:, 1]
    peaks = F.max_pool2d(prob[:, None], 3, stride=1, padding=1)[:, 0] == prob
    out: list[list[tuple[int, int, float]]] = []
    for n in range(logits.shape[0]):
        idx = ((prob[n] > thresh) & peaks[n]).nonzero(as_tuple=False)
        out.append([(int(j), int(i), float(prob[n, i, j])) for i, j in idx.tolist()])
    return out
