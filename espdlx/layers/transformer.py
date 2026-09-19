"""Transformer-path blocks: MatMul, LayerNorm, RMSNorm.

Attention needs batched matrix products plus a normalization over the
feature axis — with :class:`espdlx.layers.Softmax` (already shipped) that
completes the ``QK^T -> Softmax -> @V`` primitive::

    espdlx.Model([
        L.Linear(8, 8),                    # 0: Q
        L.Linear(8, 8),                    # 1: K
        L.Transpose((0, 2, 1)),            # 2: K^T
        L.MatMul(input_indices=[0, 2]),    # 3: Q @ K^T (both saved)
        L.Softmax(dim=-1),                 # 4: attention weights
        L.Linear(8, 8),                    # 5: V
        L.MatMul(input_index=5),           # 6: weights @ V (current @ saved)
        L.LayerNorm(8),                    # 7: feature norm
    ])

``MatMul`` exports to a plain ONNX ``MatMul`` node (runtime ``dl::MatMul``).
``LayerNorm`` exports natively as ``LayerNormalization``. ``RMSNorm`` lowers
to the standard ``Pow``/``ReduceMean``/``Add``/``Sqrt``/``Div``/``Mul``
composite — exactly the subgraph ESP-PPQ's importer fuses into a native
``RMSNormalization`` (``FORMATTER_FUSE_RMSNORM``), so the device still runs
one kernel; the ONNX graph itself stays standard (checker-clean) either way.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import Layer


def _matmul_shape(a_shape: tuple, b_shape: tuple) -> tuple:
    """Batch-matmul output shape for ``a @ b`` (both rank >= 2)."""
    a_shape = tuple(a_shape)
    b_shape = tuple(b_shape)
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise ValueError(
            f"espdlx.MatMul: both inputs must be rank >= 2 (batched matmul), "
            f"got {a_shape} and {b_shape}"
        )
    if a_shape[-1] != b_shape[-2]:
        raise ValueError(
            f"espdlx.MatMul: inner dims must match (esp-dl MatMul requirement), "
            f"got {a_shape} @ {b_shape}"
        )
    try:
        leading = torch.broadcast_shapes(a_shape[:-2], b_shape[:-2])
    except RuntimeError:
        raise ValueError(
            f"espdlx.MatMul: batch dims must broadcast, got {a_shape} and {b_shape}"
        ) from None
    return (*leading, a_shape[-2], b_shape[-1])


class MatMul(Layer):
    """Batched matrix product: binary, skip, or saved-pair.

    - ``MatMul()``: binary ``forward(x, y)`` -> ``x @ y`` for manual wiring.
    - ``MatMul(input_index=i)``: ``x @ saved[i]`` inside an
      :class:`espdlx.Model` (e.g. attention weights ``@`` a saved ``V``).
    - ``MatMul(input_indices=[i, j])``: ``saved[i] @ saved[j]`` — both
      operands are prior layer outputs (e.g. ``Q @ K^T`` with a transposed
      ``K`` in between). Order follows ``input_indices``.

    Both operands must be rank >= 2 with matching inner dims
    (``a[-1] == b[-2]``); leading batch dims must broadcast (esp-dl
    ``MatMul`` requirement). Exports to a plain ONNX ``MatMul`` node
    (runtime ``dl::MatMul``). There are no transpose flags — compose with
    :class:`espdlx.layers.Transpose` (``K^T`` above), which keeps the export
    1:1 with the graph.
    """

    def __init__(self, input_index=None, input_indices=None):
        super().__init__()
        if input_index is not None and input_indices is not None:
            raise ValueError(
                "espdlx.MatMul: input_index and input_indices are exclusive"
            )
        self.input_index = input_index
        if input_indices is not None:
            if isinstance(input_indices, int):
                input_indices = [input_indices]
            input_indices = [int(i) for i in input_indices]
            if len(input_indices) != 2:
                raise ValueError(
                    "espdlx.MatMul: input_indices must hold exactly 2 layer "
                    f"positions (a, b) for a @ b, got {input_indices!r}"
                )
        self.input_indices = input_indices

    def forward(self, x, y=None):
        if y is None:
            raise ValueError(
                "espdlx.MatMul: binary matmul needs two tensors — use "
                "MatMul(input_index=i) or MatMul(input_indices=[i, j]) inside "
                "espdlx.Model, or MatMul()(x, y)"
            )
        return torch.matmul(x, y)

    def validate_shapes(self, a_shape, b_shape=None):
        if b_shape is None:
            raise ValueError(
                "espdlx.MatMul: need both shapes to validate a matmul"
            )
        return _matmul_shape(tuple(a_shape), tuple(b_shape))


class LayerNorm(Layer):
    """Layer normalization over the last ``len(normalized_shape)`` dims.

    Standard torch semantics (wraps :class:`torch.nn.LayerNorm` with a
    learnable weight and bias). Exports to a native ONNX
    ``LayerNormalization`` node (runtime ``dl::LayerNormalization``).
    """

    def __init__(self, normalized_shape, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(int(d) for d in normalized_shape)
        if not normalized_shape or any(d < 1 for d in normalized_shape):
            raise ValueError(
                "espdlx.LayerNorm: normalized_shape must be one or more "
                f"positive dims, got {normalized_shape!r}"
            )
        self.normalized_shape = normalized_shape
        self.eps = float(eps)
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True,
                               bias=bias)
        self.weight = self.ln.weight
        self.bias = self.ln.bias

    def forward(self, x):
        return self.ln(x)

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        tail = in_shape[-len(self.normalized_shape):]
        if tail != self.normalized_shape:
            raise ValueError(
                f"espdlx.LayerNorm: trailing dims {tail} do not match "
                f"normalized_shape {self.normalized_shape}"
            )
        return in_shape


class RMSNorm(Layer):
    """Root-mean-square normalization over the trailing feature dims.

    ``weight * (x / sqrt(mean(x^2) + eps))`` with a learnable per-feature
    weight (no bias, no mean subtraction). The forward deliberately uses
    ``torch.pow(x, 2)`` (not ``x * x``): that is the exact subgraph head
    ESP-PPQ's importer recognizes (``Pow`` -> ``ReduceMean`` -> [``Add``] ->
    ``Sqrt`` -> ``Div`` -> ``Mul``), so quantization fuses it into a single
    native ``RMSNormalization`` (runtime ``dl::RMSNormalization``) instead
    of six quantized kernels.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(int(d) for d in normalized_shape)
        if not normalized_shape or any(d < 1 for d in normalized_shape):
            raise ValueError(
                "espdlx.RMSNorm: normalized_shape must be one or more "
                f"positive dims, got {normalized_shape!r}"
            )
        self.normalized_shape = normalized_shape
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def _reduce_dims(self, rank: int) -> tuple:
        # Negative dims on purpose: the exporter keeps them as negative axes
        # (e.g. [-1]), which is exactly what ESP-PPQ's RMSNorm import fusion
        # validates without needing static ranks (positive axes would abort
        # the fuse at import time).
        return tuple(range(-len(self.normalized_shape), 0))

    def forward(self, x):
        # NB: torch.pow (not x*x) — ESP-PPQ's RMSNorm import fusion keys on
        # the Pow head; with Mul it would stay six separate quantized ops.
        var = torch.pow(x, 2).mean(dim=self._reduce_dims(x.dim()), keepdim=True)
        return x / torch.sqrt(var + self.eps) * self.weight

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        tail = in_shape[-len(self.normalized_shape):]
        if tail != self.normalized_shape:
            raise ValueError(
                f"espdlx.RMSNorm: trailing dims {tail} do not match "
                f"normalized_shape {self.normalized_shape}"
            )
        return in_shape


__all__ = ["MatMul", "LayerNorm", "RMSNorm"]
