"""Convolution blocks: Conv2d (channelwise) and DepthwiseConv2d.

These mirror esp_nn_conv_s8 / esp_nn_depthwise_conv_s8. For the ESP32-S3 SIMD
path dilation must be 1; regular Conv2d is restricted to groups=1.
"""

from __future__ import annotations

import math

from torch import nn

from .base import Layer


def _pair(value, name: str) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be int or 2-tuple, got {value!r}")
        left, right = value
    else:
        left = right = value
    if isinstance(left, float) and not left.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(right, float) and not right.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(left), int(right)


class Conv2d(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding")
        dilation = _pair(dilation, "dilation")
        if dilation != (1, 1):
            raise ValueError(
                f"espdlx.Conv2d: dilation must be 1 (ESP32-S3 SIMD limit), got {dilation!r}"
            )
        if self.padding[0] > self.kernel[0] // 2 or self.padding[1] > self.kernel[1] // 2:
            raise ValueError(
                "espdlx.Conv2d: padding must keep a symmetric ('SAME') layout "
                "for odd kernels (ESP32-S3 leading-pad contract)"
            )
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1,
            bias=bias,
        )
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def validate_shapes(self, in_shape: tuple) -> tuple:
        n, c, h, w = in_shape
        if c != self.in_channels:
            raise ValueError(
                f"espdlx.Conv2d: expected {self.in_channels} input channels, got {c}"
            )
        out_h = _out_dim(h, self.kernel[0], self.stride[0], self.padding[0])
        out_w = _out_dim(w, self.kernel[1], self.stride[1], self.padding[1])
        return (n, self.out_channels, out_h, out_w)

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv2d(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        multiplier: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.multiplier = int(multiplier)
        if self.out_channels != self.in_channels * self.multiplier:
            raise ValueError(
                f"espdlx.DepthwiseConv2d: out_channels must equal "
                f"in_channels * multiplier ({self.in_channels}x{self.multiplier}"
                f"={self.in_channels * self.multiplier}), got {self.out_channels}"
            )
        self.kernel = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding")
        dilation = _pair(dilation, "dilation")
        if dilation != (1, 1):
            raise ValueError(
                f"espdlx.DepthwiseConv2d: dilation must be 1 (ESP32-S3 SIMD limit)"
            )
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=self.in_channels,
            bias=bias,
        )
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def validate_shapes(self, in_shape: tuple) -> tuple:
        n, c, h, w = in_shape
        if c != self.in_channels:
            raise ValueError(
                f"espdlx.DepthwiseConv2d: expected {self.in_channels} input channels, "
                f"got {c}"
            )
        out_h = _out_dim(h, self.kernel[0], self.stride[0], self.padding[0])
        out_w = _out_dim(w, self.kernel[1], self.stride[1], self.padding[1])
        return (n, self.out_channels, out_h, out_w)

    def forward(self, x):
        return self.conv(x)


def _out_dim(in_dim: int, k: int, s: int, p: int) -> int:
    num = in_dim + 2 * p - k
    if num < 0:
        raise ValueError(
            f"espdlx layer: input {in_dim} with kernel {k}, padding {p} leaves no "
            f"valid position"
        )
    return num // s + 1


__all__ = ["Conv2d", "DepthwiseConv2d"]
