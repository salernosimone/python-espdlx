"""Fully-connected block mirroring esp_nn_fully_connected_s8.

The ESP32-S3 FC kernel requires both the row length (input features) and the
number of output channels to be multiples of 8.
"""

from __future__ import annotations

from torch import nn

from .base import Layer


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features % 8 != 0:
            raise ValueError(
                f"espdlx.Linear: in_features must be a multiple of 8 "
                f"(ESP32-S3 SIMD limit), got {self.in_features}"
            )
        if self.out_features % 8 != 0:
            raise ValueError(
                f"espdlx.Linear: out_features must be a multiple of 8 "
                f"(ESP32-S3 SIMD limit), got {self.out_features}"
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
