"""Elementwise and structural blocks: Add/Sub/Mul/Div, Neg/Exp/Log/Sqrt, Mean, Flatten, Concat."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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


class Sub(Layer):
    """Elementwise subtraction: constant, skip connection, or two tensors.

    Mirrors :class:`Add` (``x - c``, ``x - saved[i]``, ``forward(x, y)``).
    Exports to ONNX ``Sub`` (runtime ``dl::Sub``).
    """

    def __init__(self, constant=None, input_index=None):
        super().__init__()
        if constant is not None and input_index is not None:
            raise ValueError("espdlx.Sub: constant and input_index are exclusive")
        self.constant = float(constant) if constant is not None else None
        self.input_index = input_index

    def forward(self, x, y=None):
        if self.constant is not None:
            return x - self.constant
        if y is None:
            raise ValueError(
                "espdlx.Sub: binary sub needs two tensors — use Sub(constant=c), "
                "Sub(input_index=i) inside espdlx.Model, or Sub()(x, y)"
            )
        return x - y

    def validate_shapes(self, in_shape, other_shape=None):
        if self.constant is not None:
            return tuple(in_shape)
        if other_shape is None:
            raise ValueError("espdlx.Sub: need both shapes to validate a tensor sub")
        if tuple(in_shape) != tuple(other_shape):
            raise ValueError(
                f"espdlx.Sub: tensor shapes must match (esp-dl Sub requirement), "
                f"got {tuple(in_shape)} and {tuple(other_shape)}"
            )
        return tuple(in_shape)


class Div(Layer):
    """Elementwise division: constant, skip connection, or two tensors.

    Mirrors :class:`Add` (``x / c``, ``x / saved[i]``, ``forward(x, y)``).
    Exports to ONNX ``Div`` (runtime ``dl::Div``). A zero constant is rejected
    at construction; near-zero tensor divisors are the caller's responsibility.
    """

    def __init__(self, constant=None, input_index=None):
        super().__init__()
        if constant is not None and input_index is not None:
            raise ValueError("espdlx.Div: constant and input_index are exclusive")
        if constant is not None and float(constant) == 0.0:
            raise ValueError("espdlx.Div: constant must be non-zero")
        self.constant = float(constant) if constant is not None else None
        self.input_index = input_index

    def forward(self, x, y=None):
        if self.constant is not None:
            return x / self.constant
        if y is None:
            raise ValueError(
                "espdlx.Div: binary div needs two tensors — use Div(constant=c), "
                "Div(input_index=i) inside espdlx.Model, or Div()(x, y)"
            )
        return x / y

    def validate_shapes(self, in_shape, other_shape=None):
        if self.constant is not None:
            return tuple(in_shape)
        if other_shape is None:
            raise ValueError("espdlx.Div: need both shapes to validate a tensor div")
        if tuple(in_shape) != tuple(other_shape):
            raise ValueError(
                f"espdlx.Div: tensor shapes must match (esp-dl Div requirement), "
                f"got {tuple(in_shape)} and {tuple(other_shape)}"
            )
        return tuple(in_shape)


class Neg(Layer):
    """Negation (exports to ONNX ``Neg``, runtime ``dl::Neg``)."""

    def forward(self, x):
        return -x

    def validate_shapes(self, in_shape):
        return tuple(in_shape)


class Exp(Layer):
    """Elementwise exp (exports to ONNX ``Exp``, runtime ``dl::Exp``)."""

    def forward(self, x):
        return torch.exp(x)

    def validate_shapes(self, in_shape):
        return tuple(in_shape)


class Log(Layer):
    """Elementwise natural log (exports to ONNX ``Log``, runtime ``dl::Log``)."""

    def forward(self, x):
        return torch.log(x)

    def validate_shapes(self, in_shape):
        return tuple(in_shape)


class Sqrt(Layer):
    """Elementwise sqrt (exports to ONNX ``Sqrt``, runtime ``dl::Sqrt``)."""

    def forward(self, x):
        return torch.sqrt(x)

    def validate_shapes(self, in_shape):
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


class Concat(Layer):
    """Concatenate tensors along ``dim`` (exports to ONNX ``Concat``).

    - ``Concat(input_indices=[i, j, ...], dim=1)`` inside an
      :class:`espdlx.Model`: concat the *saved outputs* of layers ``i``,
      ``j``, ... in exactly that order — this is the dense-block / FPN
      primitive. ``saved`` holds each layer's output (the raw model input is
      NOT included — reference the layer that produced the block-input). The
      current stream is NOT implicit, which gives full control of membership
      and ordering.
    - ``Concat(dim=1)``: ``forward(x, y, ...)`` -> ``cat([x, y, ...])`` for
      manual wiring.

    All inputs must match on every non-axis dimension (esp-dl ``Concat``
    requirement). ``dim`` uses Python indexing (negatives count from the
    last axis).
    """

    def __init__(self, input_indices=None, dim: int = 1):
        super().__init__()
        if input_indices is not None:
            if isinstance(input_indices, int):
                input_indices = [input_indices]
            input_indices = [int(i) for i in input_indices]
            if not input_indices:
                raise ValueError("espdlx.Concat: input_indices must not be empty")
        self.input_indices = input_indices
        self.dim = int(dim)

    def forward(self, *xs):
        if len(xs) < 2:
            raise ValueError(
                "espdlx.Concat: needs >= 2 tensors — use Concat(input_indices=[...]) "
                "inside espdlx.Model, or Concat()(x, y, ...)"
            )
        return torch.cat(xs, dim=self.dim)

    def validate_shapes(self, *shapes) -> tuple:
        shapes = [tuple(s) for s in shapes]
        if len(shapes) < 2:
            raise ValueError(
                "espdlx.Concat: need >= 2 shapes to validate a tensor concat"
            )
        rank = len(shapes[0])
        dim = self.dim
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise ValueError(
                f"espdlx.Concat: dim={self.dim} out of range for rank {rank}"
            )
        out = list(shapes[0])
        for other in shapes[1:]:
            if len(other) != rank:
                raise ValueError(
                    f"espdlx.Concat: all inputs must have the same rank, got "
                    f"{shapes[0]} and {other}"
                )
            for j in range(rank):
                if j == dim:
                    continue
                if other[j] != shapes[0][j]:
                    raise ValueError(
                        f"espdlx.Concat: non-axis dims must match (esp-dl Concat "
                        f"requirement), got {shapes[0]} and {other} on dim {j}"
                    )
            out[dim] += other[dim]
        return tuple(out)


__all__ = ["Add", "Sub", "Mul", "Div", "Neg", "Exp", "Log", "Sqrt", "Mean", "Flatten", "Concat"]
