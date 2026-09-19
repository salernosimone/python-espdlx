"""The `Model` container: an ordered stack of espdlx layers.

Layers run sequentially, each output saved. A layer with ``input_index``
(e.g. :class:`espdlx.layers.Add`, :class:`espdlx.layers.MatMul`) additionally
receives the saved output of that layer — enabling skip connections. A layer
with ``input_indices`` (:class:`espdlx.layers.Concat`, or
:class:`espdlx.layers.MatMul` with exactly two positions) receives the saved
outputs of several layers in the given order — enabling dense blocks /
FPN-style fusions and ``Q @ K^T`` attention pairs::

    espdlx.Model([
        L.Conv2d(8, 8, 3, padding=1),   # layer 0
        L.ReLU(),                        # layer 1
        L.Conv2d(8, 8, 3, padding=1),   # layer 2
        L.Add(input_index=0),            # layer 3: out2 + out0 (residual)
        L.ReLU(),
    ])

    espdlx.Model([
        L.Conv2d(2, 4, 3, padding=1),   # layer 0: h0
        L.ReLU(),                        # layer 1: h1
        L.Concat(input_indices=[0, 1], dim=1),  # layer 2: cat(h0, h1)
        ...
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
            if getattr(lyr, "multi_output", False):
                raise ValueError(
                    f"espdlx.Model: layer {pos} ({type(lyr).__name__}) returns "
                    f"multiple tensors, which a sequential Model cannot thread "
                    f"— decompose into Slice layers (one per section) instead"
                )
            idx = getattr(lyr, "input_index", None)
            indices = getattr(lyr, "input_indices", None)
            if indices is not None:
                # Multi-input layer (espdlx.Concat): gather saved outputs in
                # the user-specified order — the current stream is NOT
                # implicit, so ordering/membership is fully controlled here.
                try:
                    tensors = [saved[i] for i in indices]
                except IndexError:
                    raise ValueError(
                        f"espdlx.Model: layer {pos} input_indices={indices} has an "
                        f"out-of-range index (only {len(saved)} prior outputs)"
                    ) from None
                x = lyr(*tensors)
            elif idx is not None:
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
            if getattr(lyr, "multi_output", False):
                raise ValueError(
                    f"espdlx.Model: layer {pos} ({type(lyr).__name__}) returns "
                    f"multiple tensors, which a sequential Model cannot thread "
                    f"— decompose into Slice layers (one per section) instead"
                )
            idx = getattr(lyr, "input_index", None)
            indices = getattr(lyr, "input_indices", None)
            if indices is not None:
                try:
                    others = [shapes[i] for i in indices]
                except IndexError:
                    raise ValueError(
                        f"espdlx.Model: layer {pos} input_indices={indices} has an "
                        f"out-of-range index (only {len(shapes)} prior outputs)"
                    ) from None
                shape = lyr.validate_shapes(*others)
            elif idx is not None:
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
