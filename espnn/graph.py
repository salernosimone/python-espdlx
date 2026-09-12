"""The `Model` container: an ordered stack of espnn layers.

Deploy with :func:`espnn.convert.convert` (``torch.onnx.export`` ->
ESP-PPQ ``espdl_quantize_onnx`` -> ``.espdl``).
"""

from __future__ import annotations

from torch import nn

from .layers.base import Layer


class Model(nn.Module):
    def __init__(self, layers: list, name: str = "model") -> None:
        super().__init__()
        for lyr in layers:
            if not isinstance(lyr, Layer):
                raise TypeError(
                    f"{type(lyr).__name__} is not an espnn layer; use espnn.layers"
                )
        self.name = name
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x

    def validate_shapes(self, input_shape: tuple) -> tuple:
        shape = tuple(input_shape)
        for lyr in self.layers:
            shape = lyr.validate_shapes(shape)
        return shape

    def __repr__(self) -> str:
        return f"Model(name={self.name!r}, layers={len(self.layers)})"
