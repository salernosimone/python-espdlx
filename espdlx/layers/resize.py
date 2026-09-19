"""Upsampling block (exports to ONNX ``Resize``, runtime ``dl::Resize``).

esp-dl has no ``ConvTranspose`` in its module registry, so ``Resize`` +
``Conv`` is the sanctioned decoder path for detector / segmenter heads
(see docs/OPS_ROADMAP.md, Tier 2).
"""

from __future__ import annotations

import torch.nn.functional as F

from .base import Layer

# torch.interpolate mode -> ONNX Resize "mode" attribute (esp-dl names).
_MODES = {
    "nearest": "nearest",
    "linear": "linear",
    "bilinear": "linear",
    "bicubic": "cubic",
}


def _pair(value, name: str) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a number or 2-tuple, got {value!r}")
        left, right = value
    else:
        left = right = value
    left = float(left)
    right = float(right)
    if left <= 0 or right <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return left, right


class Resize(Layer):
    """Upsample a 4-D ``(N, C, H, W)`` tensor along H and W.

    - ``Resize(scale_factor=2)`` or ``scale_factor=(2.0, 3.0)``: multiply
      H/W by the given factors.
    - ``Resize(size=(64, 64))`` or ``size=64``: exact output H/W.
    - ``mode``: ``"nearest"`` (default), ``"linear"`` (= ``"bilinear"``),
      or ``"bicubic"`` — mapped to esp-dl ``RESIZE_NEAREST`` /
      ``RESIZE_LINEAR`` / ``RESIZE_CUBIC``.
    - ``align_corners``: interpolation grid convention for ``linear`` /
      ``bicubic`` (ignored for ``nearest``, like ``F.interpolate``).

    Exports to a single ONNX ``Resize`` node (opset 13: ``scales`` or
    ``sizes`` input + ``mode`` / ``coordinate_transformation_mode``
    attributes) which the espdlx Arduino runtime executes (``dl::Resize``).
    """

    def __init__(
        self,
        scale_factor=None,
        size=None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        if scale_factor is None and size is None:
            raise ValueError(
                "espdlx.Resize: need scale_factor or size (not both None)"
            )
        if scale_factor is not None and size is not None:
            raise ValueError(
                "espdlx.Resize: scale_factor and size are exclusive"
            )
        self.mode = str(mode).lower()
        if self.mode not in _MODES:
            raise ValueError(
                f"espdlx.Resize: mode must be one of {sorted(_MODES)} — esp-dl "
                f"runs NEAREST/LINEAR/CUBIC, got {mode!r}"
            )
        self.onnx_mode = _MODES[self.mode]
        self.align_corners = bool(align_corners)
        if self.mode == "nearest" and align_corners:
            raise ValueError(
                "espdlx.Resize: align_corners only applies to linear/bicubic "
                "modes (nearest ignores the grid convention)"
            )
        if scale_factor is not None:
            sf = _pair(scale_factor, "scale_factor")
            self.scale_factor = sf
            self.size = None
        else:
            user_size = size
            if isinstance(user_size, (tuple, list)):
                user_size = list(user_size)
            else:
                user_size = [user_size]
            if len(user_size) == 1:
                user_size = [user_size[0], user_size[0]]  # scalar/(x,) -> (x, x)
            if len(user_size) != 2:
                raise ValueError(
                    f"espdlx.Resize: size must be a number or 2-tuple, got {size!r}"
                )
            for v in user_size:
                if float(v) != int(float(v)):
                    raise ValueError(
                        f"espdlx.Resize: size must be integer pixels, got {size!r}"
                    )
            self.size = tuple(int(float(v)) for v in user_size)
            self.scale_factor = None

    def validate_shapes(self, in_shape: tuple) -> tuple:
        n, c, h, w = in_shape
        if self.size is not None:
            sh, sw = self.size
        else:
            sh, sw = self.scale_factor
            sh = int(h * sh)
            sw = int(w * sw)
        return (n, c, sh, sw)

    def forward(self, x):
        kwargs = {}
        if self.size is not None:
            kwargs["size"] = self.size
        else:
            kwargs["scale_factor"] = self.scale_factor
        if self.mode != "nearest":
            kwargs["align_corners"] = self.align_corners
        return F.interpolate(x, mode=self.mode, **kwargs)


__all__ = ["Resize"]