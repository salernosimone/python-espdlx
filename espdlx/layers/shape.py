"""Shape-juggling blocks: Transpose / Reshape / Squeeze / Unsqueeze / Slice / Gather / Pad / Split.

Single-output ops (everything except :class:`Split`) chain freely inside an
:class:`espdlx.Model` — e.g. multi-head reshape-transpose patterns stay
single-tensor at every step::

    L.Reshape((4, 2, 4)),      # (B, T, D) -> (B, T, H, Dh)
    L.Transpose((0, 2, 1, 3)), # (B, H, T, Dh)
    ...
    L.Transpose((0, 2, 1, 3)), # back to (B, T, H, Dh)
    L.Reshape((4, 8)),         # merge heads -> (B, T, D)

:class:`Split` returns one tensor per section (a tuple), which a sequential
``Model`` cannot thread — decompose head/section splits into :class:`Slice`
layers instead (same runtime ``dl::Slice`` op, proven equivalent in
``tests/test_model.py``). ``Split`` itself is still provided for manual
wiring; it exports to a native ONNX ``Split`` node.

All reshape-style targets are per-sample (batch-agnostic): :class:`Reshape`
prepends the runtime batch dim itself, so calibration (batch-N) and device
(batch-1) agree. A baked batch dim would quantize fine but break on any
other batch size — fail fast beats debug later.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import Layer


def _normalize_dim(dim: int, rank: int, name: str) -> int:
    dim = int(dim)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise ValueError(
            f"espdlx.{name}: dim={dim} out of range for rank {rank}"
        )
    return dim


class Transpose(Layer):
    """Permute axes (exports to ONNX ``Transpose``, runtime ``dl::Transpose``).

    ``Transpose(perm)`` with a full permutation, e.g. ``(0, 2, 1)`` for
    ``K^T`` in attention or ``(0, 2, 1, 3)`` for head shuffles. Negative
    entries count from the last axis.
    """

    def __init__(self, perm):
        super().__init__()
        self.perm = tuple(int(p) for p in perm)

    def forward(self, x):
        return x.permute(self.perm)

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        rank = len(in_shape)
        perm = tuple(p + rank if p < 0 else p for p in self.perm)
        if sorted(perm) != list(range(rank)):
            raise ValueError(
                f"espdlx.Transpose: perm={self.perm} is not a permutation of "
                f"rank {rank} (esp-dl Transpose requirement)"
            )
        return tuple(in_shape[p] for p in perm)


class Reshape(Layer):
    """Reshape the per-sample dims (exports to ONNX ``Reshape``).

    ``Reshape((2, 4))`` maps ``(B, 4, 8)`` -> ``(B, 2, 4, 4)``: ``shape``
    covers every dim EXCEPT the batch, which is prepended automatically
    (exported as a leading ``-1``, so calibration batch-N and device
    batch-1 agree). Within ``shape``, ``-1`` infers one dim and ``0``
    copies the input dim at that position (ONNX semantics).

    :func:`espdlx.convert.make_espdl_friendly` still rewrites a
    flatten-equivalent ``Reshape`` to ``Flatten`` (the old-PPQ path);
    genuine reshapes (head split/merge) are preserved for ``dl::Reshape``.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(int(d) for d in shape)
        if not self.shape:
            raise ValueError("espdlx.Reshape: shape must not be empty")
        if sum(d == -1 for d in self.shape) > 1:
            raise ValueError(
                f"espdlx.Reshape: at most one -1 allowed, got {self.shape!r}"
            )
        if any(d < -1 for d in self.shape):
            raise ValueError(
                f"espdlx.Reshape: dims must be >= -1 (-1 infers, 0 copies "
                f"input), got {self.shape!r}"
            )

    def resolve(self, sample_shape) -> tuple:
        """Resolve ``shape`` against per-sample dims (no batch)."""
        sample_shape = tuple(int(d) for d in sample_shape)
        total = 1
        for d in sample_shape:
            total *= d
        out = []
        for i, d in enumerate(self.shape):
            if d == 0:
                if i >= len(sample_shape):
                    raise ValueError(
                        f"espdlx.Reshape: 0 at position {i} copies no input "
                        f"dim of {tuple(sample_shape)}"
                    )
                out.append(sample_shape[i])
            else:
                out.append(d)
        known = 1
        for d in out:
            if d != -1:
                known *= d
        if -1 in out:
            if known == 0 or total % known != 0:
                raise ValueError(
                    f"espdlx.Reshape: cannot infer -1 for shape {self.shape!r} "
                    f"from sample dims {tuple(sample_shape)}"
                )
            out[out.index(-1)] = total // known
        else:
            prod = 1
            for d in out:
                prod *= d
            if prod != total:
                raise ValueError(
                    f"espdlx.Reshape: target {self.shape!r} (={prod} elems) "
                    f"does not match sample dims {tuple(sample_shape)} "
                    f"(={total} elems)"
                )
        return tuple(out)

    def forward(self, x):
        sample = self.resolve(x.shape[1:])
        return x.reshape((-1,) + sample)

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        return (in_shape[0],) + self.resolve(in_shape[1:])


class Squeeze(Layer):
    """Drop size-1 dims (exports to ONNX ``Squeeze``).

    ``Squeeze(dim)`` drops that dim (it must be size 1 — fail fast, unlike
    torch's silent no-op); ``Squeeze()`` drops all size-1 dims.
    """

    def __init__(self, dim=None):
        super().__init__()
        self.dim = None if dim is None else int(dim)

    def forward(self, x):
        # Straight-line (no data-dependent branches): torch.squeeze on a
        # non-singleton dim is a no-op, so export tracing stays static.
        # Misuse fails fast in validate_shapes instead.
        if self.dim is None:
            return x.squeeze()
        return x.squeeze(_normalize_dim(self.dim, x.dim(), "Squeeze"))

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        if self.dim is None:
            return tuple(d for d in in_shape if d != 1)
        d = _normalize_dim(self.dim, len(in_shape), "Squeeze")
        if in_shape[d] != 1:
            raise ValueError(
                f"espdlx.Squeeze: dim {self.dim} has size {in_shape[d]}, not 1"
            )
        return tuple(v for i, v in enumerate(in_shape) if i != d)


class Unsqueeze(Layer):
    """Insert a size-1 dim (exports to ONNX ``Unsqueeze``)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x):
        return x.unsqueeze(self.dim)

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        rank = len(in_shape) + 1
        d = self.dim + rank if self.dim < 0 else self.dim
        if d < 0 or d >= rank:
            raise ValueError(
                f"espdlx.Unsqueeze: dim={self.dim} out of range for "
                f"output rank {rank}"
            )
        out = list(in_shape)
        out.insert(d, 1)
        return tuple(out)


class Slice(Layer):
    """Static window along axes (exports to ONNX ``Slice``, runtime ``dl::Slice``).

    ``Slice(starts, ends, axes=None, steps=None)`` with ONNX semantics:
    negative ``starts``/``ends`` count from the end and out-of-range bounds
    clamp. ``axes`` defaults to ``range(len(starts))``; ``steps`` default to
    1 and must be positive (negative-step reversal is a transpose-family
    concern, not a window).
    """

    def __init__(self, starts, ends, axes=None, steps=None):
        super().__init__()
        self.starts = [int(v) for v in starts]
        self.ends = [int(v) for v in ends]
        if len(self.starts) != len(self.ends):
            raise ValueError(
                f"espdlx.Slice: starts and ends must pair up, got "
                f"{self.starts!r} and {self.ends!r}"
            )
        n = len(self.starts)
        self.axes = list(range(n)) if axes is None else [int(v) for v in axes]
        self.steps = [1] * n if steps is None else [int(v) for v in steps]
        if len(self.axes) != n or len(self.steps) != n:
            raise ValueError(
                "espdlx.Slice: starts/ends/axes/steps must have equal length, got "
                f"{self.starts!r}, {self.ends!r}, {self.axes!r}, {self.steps!r}"
            )
        if any(s <= 0 for s in self.steps):
            raise ValueError(
                f"espdlx.Slice: steps must be positive, got {self.steps!r}"
            )
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(
                f"espdlx.Slice: axes must be unique, got {self.axes!r}"
            )

    def _window(self, axis: int, size: int, start: int, end: int, step: int):
        if start < 0:
            start += size
        if end < 0:
            end += size
        start = min(max(start, 0), size)
        end = min(max(end, 0), size)
        length = max(0, math.ceil((end - start) / step))
        return start, length

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        rank = len(in_shape)
        out = list(in_shape)
        for axis, start, end, step in zip(self.axes, self.starts, self.ends,
                                          self.steps):
            a = _normalize_dim(axis, rank, "Slice")
            _, length = self._window(a, in_shape[a], start, end, step)
            out[a] = length
        return tuple(out)

    def forward(self, x):
        rank = x.dim()
        picks = [slice(None)] * rank
        for axis, start, end, step in zip(self.axes, self.starts, self.ends,
                                          self.steps):
            a = _normalize_dim(axis, rank, "Slice")
            size = x.shape[a]
            s = start + size if start < 0 else start
            e = end + size if end < 0 else end
            s = min(max(s, 0), size)
            e = min(max(e, 0), size)
            picks[a] = slice(s, e, step)
        return x[tuple(picks)]


class Gather(Layer):
    """Select entries along ``dim`` at fixed ``indices`` (exports to ONNX ``Gather``).

    ``Gather(indices=[0, 2], dim=1)`` is ``torch.index_select`` with a frozen
    index list (e.g. CLS-token pick, channel subset). Indices are stored as a
    buffer so they export as an initializer. Only constant indices are
    supported: a float ``Model`` stream cannot supply int64 indices (there is
    no ``Cast`` op in espdlx), so there is deliberately no ``input_index``
    mode — use binary ``forward(x)`` with the frozen list, or wire
    ``Gather()(data, idx)`` manually.
    """

    def __init__(self, indices, dim: int = 0):
        super().__init__()
        idx = torch.as_tensor(list(indices), dtype=torch.long).flatten()
        if idx.numel() == 0:
            raise ValueError("espdlx.Gather: indices must not be empty")
        if bool((idx < 0).any()):
            raise ValueError(
                f"espdlx.Gather: indices must be non-negative, got {list(indices)!r}"
            )
        self.register_buffer("indices", idx)
        self.dim = int(dim)

    def forward(self, x, idx=None):
        # Straight-line (no data-dependent branches): out-of-range indices
        # raise in index_select itself; static misuse fails fast in
        # validate_shapes. Keeps torch.export tracing static.
        index = self.indices if idx is None else torch.as_tensor(idx)
        d = _normalize_dim(self.dim, x.dim(), "Gather")
        return torch.index_select(x, d, index.to(x.device))

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        d = _normalize_dim(self.dim, len(in_shape), "Gather")
        if bool((self.indices >= in_shape[d]).any()):
            raise ValueError(
                f"espdlx.Gather: index out of range for dim {self.dim} "
                f"(size {in_shape[d]})"
            )
        out = list(in_shape)
        out[d] = int(self.indices.numel())
        return tuple(out)


class Pad(Layer):
    """Constant-pad spatial/edge dims (exports to ONNX ``Pad``, runtime ``dl::Pad``).

    ``padding`` follows :func:`torch.nn.functional.pad` order (last dim
    first, ``(before, after)`` pairs outward), e.g. ``(0, 0, 1, 1)`` pads one
    row top/bottom of a 4-D tensor. Only constant ``value`` padding is
    supported (no reflect/edge — the runtime contract is constant); pads must
    be non-negative (cropping is :class:`Slice`'s job).
    """

    def __init__(self, padding, value: float = 0.0):
        super().__init__()
        self.padding = tuple(int(v) for v in padding)
        if len(self.padding) % 2 != 0:
            raise ValueError(
                f"espdlx.Pad: padding must hold (before, after) pairs, got "
                f"{self.padding!r}"
            )
        if any(v < 0 for v in self.padding):
            raise ValueError(
                f"espdlx.Pad: pads must be non-negative (use Slice to crop), "
                f"got {self.padding!r}"
            )
        self.value = float(value)

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        rank = len(in_shape)
        if len(self.padding) // 2 > rank:
            raise ValueError(
                f"espdlx.Pad: {len(self.padding) // 2} padded dims exceed rank "
                f"{rank}"
            )
        out = list(in_shape)
        for k in range(len(self.padding) // 2):
            dim = rank - 1 - k
            out[dim] += self.padding[2 * k] + self.padding[2 * k + 1]
        return tuple(out)

    def forward(self, x):
        return F.pad(x, self.padding, mode="constant", value=self.value)


class Split(Layer):
    """Split along ``dim`` into sections (exports to ONNX ``Split``).

    ``Split(4, dim=1)`` splits into 4 equal parts; ``Split([2, 2, 4], dim=1)``
    uses explicit sizes. Returns a tuple of tensors (one per section), so it
    is for manual wiring — inside an :class:`espdlx.Model` decompose into
    :class:`Slice` layers instead (see module docs). ``validate_shapes``
    returns a tuple of shapes.
    """

    multi_output = True

    def __init__(self, sections, dim: int = 1):
        super().__init__()
        if isinstance(sections, int):
            if sections < 2:
                raise ValueError(
                    f"espdlx.Split: need >= 2 sections, got {sections}"
                )
            self.sections: int | list = int(sections)
        else:
            self.sections = [int(v) for v in sections]
            if len(self.sections) < 2 or any(v < 1 for v in self.sections):
                raise ValueError(
                    f"espdlx.Split: sizes must hold >= 2 positive parts, got "
                    f"{sections!r}"
                )
        self.dim = int(dim)

    def _sizes(self, dim_size: int) -> list:
        if isinstance(self.sections, int):
            if dim_size % self.sections != 0:
                raise ValueError(
                    f"espdlx.Split: dim size {dim_size} not divisible into "
                    f"{self.sections} equal parts"
                )
            return [dim_size // self.sections] * self.sections
        if sum(self.sections) != dim_size:
            raise ValueError(
                f"espdlx.Split: sizes sum to {sum(self.sections)}, but dim has "
                f"{dim_size}"
            )
        return list(self.sections)

    def forward(self, x):
        d = _normalize_dim(self.dim, x.dim(), "Split")
        return torch.split(x, self._sizes(x.shape[d]), dim=d)

    def validate_shapes(self, in_shape):
        in_shape = tuple(in_shape)
        d = _normalize_dim(self.dim, len(in_shape), "Split")
        return tuple(
            tuple(in_shape[i] if i != d else s for i in range(len(in_shape)))
            for s in self._sizes(in_shape[d])
        )


__all__ = [
    "Transpose",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Slice",
    "Gather",
    "Pad",
    "Split",
]
