"""Elementwise and structural blocks: Add, Mul, Mean, Flatten."""

from __future__ import annotations

from torch import nn
import torch.nn.functional as F

from .base import Layer


class Add(Layer):
    """Elementwise addition with a constant."""

    def __init__(self, constant=None, input_index=None):
        super().__init__()
        if constant is None and input_index is None:
            raise ValueError("espnn.Add: provide a constant or input_index")
        self.constant = float(constant) if constant is not None else None
        self.input_index = input_index

    def forward(self, x):
        if self.constant is not None:
            return x + self.constant
        raise NotImplementedError("espnn.Add requires a constant")

    def validate_shapes(self, in_shape):
        return in_shape


class Mul(Layer):
    """Elementwise multiplication with a constant."""

    def __init__(self, constant=None, input_index=None):
        super().__init__()
        if constant is None and input_index is None:
            raise ValueError("espnn.Mul: provide a constant or input_index")
        self.constant = float(constant) if constant is not None else None
        self.input_index = input_index

    def forward(self, x):
        if self.constant is not None:
            return x * self.constant
        raise NotImplementedError("espnn.Mul requires a constant")

    def validate_shapes(self, in_shape):
        return in_shape


class Mean(Layer):
    """Spatial mean over H,W per channel (global average pooling)."""

    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)

    def validate_shapes(self, in_shape):
        n, c, _h, _w = in_shape
        return (n, c, 1, 1)


class Flatten(Layer):
    def forward(self, x):
        return x.flatten(1)

    def validate_shapes(self, in_shape):
        n = in_shape[0]
        total = 1
        for d in in_shape[1:]:
            total *= d
        return (n, total)


__all__ = ["Add", "Mul", "Mean", "Flatten"]
