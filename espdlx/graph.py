"""The `Model` container: an ordered stack of espdlx layers.

Layers run sequentially, each output saved. A layer with ``input_index``
(currently :class:`espdlx.layers.Add`) additionally receives the saved output
of that layer — enabling skip connections::

    espdlx.Model([
        L.Conv2d(8, 8, 3, padding=1),   # layer 0
        L.ReLU(),                        # layer 1
        L.Conv2d(8, 8, 3, padding=1),   # layer 2
        L.Add(input_index=0),            # layer 3: out2 + out0 (residual)
        L.ReLU(),
    ])

Deploy with :func:`espdlx.convert.convert` (``torch.onnx.export`` ->
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
                    f"{type(lyr).__name__} is not an espdlx layer; use espdlx.layers"
                )
        self.name = name
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        saved = []
        for pos, lyr in enumerate(self.layers):
            idx = getattr(lyr, "input_index", None)
            if idx is not None:
                try:
                    other = saved[idx]
                except IndexError:
                    raise ValueError(
                        f"espdlx.Model: layer {pos} input_index={idx} is out of "
                        f"range (only {len(saved)} prior outputs)"
                    ) from None
                x = lyr(x, other)
            else:
                x = lyr(x)
            saved.append(x)
        return x

    def validate_shapes(self, input_shape: tuple) -> tuple:
        shapes: list = []
        shape = tuple(input_shape)
        for pos, lyr in enumerate(self.layers):
            idx = getattr(lyr, "input_index", None)
            if idx is not None:
                try:
                    other = shapes[idx]
                except IndexError:
                    raise ValueError(
                        f"espdlx.Model: layer {pos} input_index={idx} is out of "
                        f"range (only {len(shapes)} prior outputs)"
                    ) from None
                shape = lyr.validate_shapes(shape, other)
            else:
                shape = lyr.validate_shapes(shape)
            shapes.append(shape)
        return shape

    def __repr__(self) -> str:
        return f"Model(name={self.name!r}, layers={len(self.layers)})"
