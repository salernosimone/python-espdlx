"""Elementwise and structural blocks: Add, Mul, Mean, Flatten."""

from __future__ import annotations

from torch import nn
import torch.nn.functional as F

from .base import Layer


class Add(Layer):
    """Elementwise addition: constant, skip connection, or two tensors.

    - ``Add(constant=c)``: ``x + c`` (bias-style shift).
    - ``Add(input_index=i)``: ``x + saved[i]`` inside an :class:`espdlx.Model`,
      where ``saved[i]`` is the output of layer ``i`` (Python indexing, so
      negatives count back from the current layer) — i.e. skip connections.
      Both tensors must have the same shape (esp-dl ``Add`` requirement).
    - ``Add()``: binary ``forward(x, y)`` -> ``x + y`` for manual wiring.

    All three export to a plain ONNX ``Add`` node, which the espdlx Arduino
    runtime executes (``dl::Add``).
    """

    def __init__(self, constant=None, input_index=None):
        super().__init__()
        if constant is not None and input_index is not None:
            raise ValueError("espdlx.Add: constant and input_index are exclusive")
        self.constant = float(constant) if constant is not None else None
        self.input_index = input_index

    def forward(self, x, y=None):
        if self.constant is not None:
            return x + self.constant
        if y is None:
            raise ValueError(
                "espdlx.Add: binary add needs two tensors — use Add(constant=c), "
                "Add(input_index=i) inside espdlx.Model, or Add()(x, y)"
            )
        return x + y

    def validate_shapes(self, in_shape, other_shape=None):
        if self.constant is not None:
            return tuple(in_shape)
        if other_shape is None:
            raise ValueError("espdlx.Add: need both shapes to validate a tensor add")
        if tuple(in_shape) != tuple(other_shape):
            raise ValueError(
                f"espdlx.Add: tensor shapes must match (esp-dl Add requirement), "
                f"got {tuple(in_shape)} and {tuple(other_shape)}"
            )
        return tuple(in_shape)


class Mul(Layer):
    """Elementwise multiplication: constant, skip connection, or two tensors.

    - ``Mul(constant=c)``: ``x * c`` (gain-style scale).
    - ``Mul(input_index=i)``: ``x * saved[i]`` inside an :class:`espdlx.Model`,
      where ``saved[i]`` is the output of layer ``i`` (Python indexing) —
      e.g. gating paths. Both tensors must have the same shape
      (esp-dl ``Mul`` requirement).
    - ``Mul()``: binary ``forward(x, y)`` -> ``x * y`` for manual wiring.

    All three export to a plain ONNX ``Mul`` node, which the espdlx Arduino
    runtime executes (``dl::Mul``).
    """

    def __init__(self, constant=None, input_index=None):
        super().__init__()
        if constant is not None and input_index is not None:
            raise ValueError("espdlx.Mul: constant and input_index are exclusive")
        self.constant = float(constant) if constant is not None else None
        self.input_index = input_index

    def forward(self, x, y=None):
        if self.constant is not None:
            return x * self.constant
        if y is None:
            raise ValueError(
                "espdlx.Mul: binary mul needs two tensors — use Mul(constant=c), "
                "Mul(input_index=i) inside espdlx.Model, or Mul()(x, y)"
            )
        return x * y

    def validate_shapes(self, in_shape, other_shape=None):
        if self.constant is not None:
            return tuple(in_shape)
        if other_shape is None:
            raise ValueError("espdlx.Mul: need both shapes to validate a tensor mul")
        if tuple(in_shape) != tuple(other_shape):
            raise ValueError(
                f"espdlx.Mul: tensor shapes must match (esp-dl Mul requirement), "
                f"got {tuple(in_shape)} and {tuple(other_shape)}"
            )
        return tuple(in_shape)


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
