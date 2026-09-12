"""Activation blocks (plain torch semantics, esp-dl deployable)."""

from __future__ import annotations

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


__all__ = ["ReLU", "ReLU6", "HardSwish", "Sigmoid", "Softmax", "BatchNorm2d"]
