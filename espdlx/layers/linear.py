"""Fully-connected block (exports to ONNX ``Gemm``, runtime ``dl::Gemm``).

No multiple-of-8 restriction: ``Linear(12, 5)`` quantizes and runs on-device
(verified 3/3 MATCH on ESP32-S3, maxabs ~0.003). The old esp-nn-era %8 rule
is dropped — esp-dl handles arbitrary shapes.
"""

from __future__ import annotations

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


__all__ = ["Linear"]
