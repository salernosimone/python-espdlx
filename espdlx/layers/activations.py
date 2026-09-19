"""Activation blocks (plain torch semantics, esp-dl deployable)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import Layer


class ReLU(Layer):
    def forward(self, x):
        return F.relu(x)

    def validate_shapes(self, in_shape):
        return in_shape


class ReLU6(Layer):
    def forward(self, x):
        return F.relu6(x)

    def validate_shapes(self, in_shape):
        return in_shape


class HardSwish(Layer):
    def forward(self, x):
        return F.hardswish(x)

    def validate_shapes(self, in_shape):
        return in_shape


class Sigmoid(Layer):
    def forward(self, x):
        return F.sigmoid(x)

    def validate_shapes(self, in_shape):
        return in_shape


class LeakyReLU(Layer):
    """Leaky ReLU (exports to ONNX ``LeakyRelu``, runtime ``dl::LeakyRelu``)."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def validate_shapes(self, in_shape):
        return in_shape


class Tanh(Layer):
    """Tanh (exports to ONNX ``Tanh``, runtime ``dl::Tanh``)."""

    def forward(self, x):
        return torch.tanh(x)

    def validate_shapes(self, in_shape):
        return in_shape


class Swish(Layer):
    """Swish / SiLU (``x * sigmoid(x)``).

    Exports to ONNX ``Sigmoid`` + ``Mul`` (no native SiLU op at opset 13) —
    both already device-proven — so no new-op risk.
    """

    def forward(self, x):
        return F.silu(x)

    def validate_shapes(self, in_shape):
        return in_shape


class Elu(Layer):
    """ELU (exports to ONNX ``Elu``, runtime ``dl::Elu``)."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        return F.elu(x, alpha=self.alpha)

    def validate_shapes(self, in_shape):
        return in_shape


class HardSigmoid(Layer):
    """Hard sigmoid (exports to ONNX ``HardSigmoid``, runtime ``dl::HardSigmoid``)."""

    def forward(self, x):
        return F.hardsigmoid(x)

    def validate_shapes(self, in_shape):
        return in_shape


class Clip(Layer):
    """Clamp to ``[min_val, max_val]`` (exports to ONNX ``Clip``)."""

    def __init__(self, min_val: float = 0.0, max_val: float = 6.0):
        super().__init__()
        if float(max_val) < float(min_val):
            raise ValueError(
                f"espdlx.Clip: max_val must be >= min_val, got {min_val}, {max_val}"
            )
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)

    def validate_shapes(self, in_shape):
        return in_shape


class Softmax(Layer):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)

    def validate_shapes(self, in_shape):
        return in_shape


class BatchNorm2d(Layer):
    """Batch normalization (standard torch semantics, exports to ONNX)."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.bn = nn.BatchNorm2d(self.num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.bn(x)

    def validate_shapes(self, in_shape):
        return in_shape


__all__ = [
    "ReLU",
    "ReLU6",
    "HardSwish",
    "Sigmoid",
    "Softmax",
    "BatchNorm2d",
    "LeakyReLU",
    "Tanh",
    "Swish",
    "Elu",
    "HardSigmoid",
    "Clip",
]
