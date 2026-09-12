"""Pooling blocks mirroring esp_nn_max_pool_s8 / esp_nn_avg_pool_s8."""

from __future__ import annotations

from torch import nn
import torch.nn.functional as F

from .base import Layer
from .conv import _out_dim, _pair


class _Pool(Layer):
    kind: str

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel = _pair(kernel_size, "kernel_size")
        self.stride = _pair(kernel_size, "stride") if stride is None else _pair(stride, "stride")
        self.padding = _pair(padding, "padding")

    def validate_shapes(self, in_shape: tuple) -> tuple:
        n, c, h, w = in_shape
        out_h = _out_dim(h, self.kernel[0], self.stride[0], self.padding[0])
        out_w = _out_dim(w, self.kernel[1], self.stride[1], self.padding[1])
        return (n, c, out_h, out_w)


class MaxPool2d(_Pool):
    kind = "max_pool"

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__(kernel_size, stride, padding)

    def forward(self, x):
        return F.max_pool2d(x, self.kernel, stride=self.stride, padding=self.padding)


class AvgPool2d(_Pool):
    kind = "avg_pool"

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__(kernel_size, stride, padding)

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel, stride=self.stride, padding=self.padding)


__all__ = ["MaxPool2d", "AvgPool2d"]
