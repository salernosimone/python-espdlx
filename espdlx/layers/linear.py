"""Fully-connected blocks (export to ONNX ``Gemm``, runtime ``dl::Gemm``).

No multiple-of-8 restriction: ``Linear(12, 5)`` quantizes and runs on-device
(verified 3/3 MATCH on ESP32-S3, maxabs ~0.003). The old esp-nn-era %8 rule
is dropped — esp-dl handles arbitrary shapes.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import Layer


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features < 1 or self.out_features < 1:
            raise ValueError(
                f"espdlx.Linear: dims must be >= 1, got "
                f"({self.in_features}, {self.out_features})"
            )
        self.fc = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.weight = self.fc.weight
        self.bias = self.fc.bias

    def validate_shapes(self, in_shape: tuple) -> tuple:
        *leading, features = in_shape
        if features != self.in_features:
            raise ValueError(
                f"espdlx.Linear: expected {self.in_features} input features, got "
                f"{features}"
            )
        return (*leading, self.out_features)

    def forward(self, x):
        return self.fc(x)


class Gemm(Layer):
    """Fully-connected block with full ONNX ``Gemm`` semantics.

    ``Y = alpha * A' * B' + beta * C`` with ``A' = X`` (transA must be False),
    ``B' = W`` (``transB=False``) or ``W.T`` (``transB=True``), and ``C`` the
    bias broadcast. Generalizes :class:`Linear`:

    - ``Gemm(in, out)`` is ``Linear(in, out)`` (default ``transB=True``:
      weight ``[out, in]``, ``x @ W.T + b``).
    - ``alpha``/``beta`` scale the matmul / bias term (ONNX ``Gemm`` attrs).
    - ``transB=False`` stores the weight as the raw ONNX ``B`` tensor
      ``[in, out]`` and computes ``x @ W``.

    ``transA=True`` is rejected at construction — the espdlx runtime
    ``dl::Gemm`` asserts ``transA == 0``.

    Export: 2-D input lowers to a single ONNX ``Gemm`` node. Plain
    ``alpha=1.0, beta=1.0`` with ``transB=True`` reuses the tier-1-proven
    ``nn.Linear`` path (``Gemm(transB=1)``, which ESP-PPQ folds). Any other
    combo emits explicit ``alpha``/``beta`` attributes (``transB=False`` by
    ``addmm``; ``transB=True`` via a ``Transpose`` of the weight) —
    :func:`espdlx.convert.make_espdl_friendly` folds those into the weight /
    bias initializers (ESP-PPQ asserts ``alpha == beta == 1.0``), so the
    device still runs a plain ``dl::Gemm``.

    Batch inputs (extra leading dims) work in float; they export as
    ``MatMul`` (+``Mul``/``Add``) like ``nn.Linear`` today (MatMul is a
    tier-3 device op, not yet proven on-device).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
        transB: bool = True,
        transA: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features < 1 or self.out_features < 1:
            raise ValueError(
                f"espdlx.Gemm: dims must be >= 1, got "
                f"({self.in_features}, {self.out_features})"
            )
        if transA:
            raise ValueError(
                "espdlx.Gemm: transA must be False — the espdlx runtime "
                "dl::Gemm asserts transA == 0"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.transB = bool(transB)
        if self.transB:
            # nn.Linear convention: weight [out, in], y = x @ W.T + b.
            self.weight = nn.Parameter(
                torch.empty(self.out_features, self.in_features)
            )
        else:
            # Raw ONNX B tensor: weight [in, out], y = x @ W + b.
            self.weight = nn.Parameter(
                torch.empty(self.in_features, self.out_features)
            )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            self.bias = nn.Parameter(
                torch.empty(self.out_features).uniform_(-bound, bound)
            )
        else:
            self.register_parameter("bias", None)

    def validate_shapes(self, in_shape: tuple) -> tuple:
        *leading, features = in_shape
        if features != self.in_features:
            raise ValueError(
                f"espdlx.Gemm: expected {self.in_features} input features, got "
                f"{features}"
            )
        return (*leading, self.out_features)

    def forward(self, x):
        import torch.nn.functional as F

        w = self.weight
        b = self.bias
        if self.transB and self.alpha == 1.0 and self.beta == 1.0:
            # tier-1-proven path: single Gemm(transB=1, alpha=1, beta=1).
            return F.linear(x, w, b)
        if x.dim() == 2:
            ww = w if not self.transB else w.t()
            if b is None:
                b = torch.zeros(self.out_features, dtype=x.dtype, device=x.device)
            return torch.addmm(b, x, ww, alpha=self.alpha, beta=self.beta)
        proj = x @ (w if not self.transB else w.t())
        y = self.alpha * proj
        if b is not None:
            y = y + self.beta * b
        return y


__all__ = ["Linear", "Gemm"]
