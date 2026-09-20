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

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from .layers.base import Layer


def _resolve_device(device: Any = None) -> torch.device:
    """Pick a torch device, preferring CUDA then MPS then CPU."""
    if device is not None:
        return torch.device(device) if not isinstance(device, torch.device) else device
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_loader(data: Dataset | DataLoader, *, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap a Dataset in a DataLoader; pass DataLoaders through untouched."""
    if isinstance(data, DataLoader):
        return data
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


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

    def fit(
        self,
        train_data: Dataset | DataLoader,
        val_data: Dataset | DataLoader | None = None,
        *,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: Any = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> dict:
        """Train a classifier with cross-entropy loss, hiding the PyTorch loop.

        Beginner-friendly shortcut for the standard recipe (Adam +
        ``cross_entropy`` + per-epoch validation accuracy)::

            history = model.fit(train_dataset)  # 80/20 split when no val set
            history = model.fit(train_loader, val_loader, epochs=20)

        - ``train_data``: a :class:`~torch.utils.data.Dataset` or an
          already-batched :class:`~torch.utils.data.DataLoader` yielding
          ``(images, labels)`` batches.
        - ``val_data``: optional validation set (same formats). When omitted,
          ``train_data`` is split 80/20 (train/validation) with ``seed`` for
          reproducibility.
        - ``batch_size`` applies when a Dataset (not a DataLoader) is given.
          When splitting a DataLoader with no ``val_data``, the new loaders
          reuse that loader's batch size.
        - Returns ``{"loss": [...], "val_accuracy": [...] or None}``
          (per-epoch mean train loss and validation accuracy).
        - Long runs (``epochs > 100``) log every ``epochs // 10`` epochs
          so the output stays ~10 lines; history still records every epoch.
        """
        import torch.nn.functional as F

        dev = _resolve_device(device)
        self.to(dev)

        if val_data is None:
            # 80/20 split of whatever training data was given.
            if isinstance(train_data, DataLoader):
                dataset = train_data.dataset
                loader_bs = train_data.batch_size or batch_size
            else:
                dataset = train_data
                loader_bs = batch_size
            n = len(dataset)
            if n < 2:
                raise ValueError(
                    "espdlx.Model.fit: need at least 2 samples to make an 80/20 split"
                )
            n_val = max(1, int(round(0.2 * n)))
            n_train = n - n_val
            gen = torch.Generator().manual_seed(seed)
            train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=gen)
            train_loader = DataLoader(train_subset, batch_size=loader_bs, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=loader_bs)
        else:
            train_loader = _as_loader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = _as_loader(val_data, batch_size=batch_size, shuffle=False)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        losses: list[float] = []
        val_accs: list[float] = []
        # Long runs (>100 epochs) log ~10 lines instead of one per epoch.
        log_interval = max(1, epochs // 10) if epochs > 100 else 1
        for epoch in range(epochs):
            self.train()
            running = 0.0
            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                loss = F.cross_entropy(self(x), y)
                loss.backward()
                opt.step()
                running += loss.item()
            avg = running / max(1, len(train_loader))
            losses.append(avg)
            acc = None
            if val_loader is not None:
                acc = self.evaluate(val_loader, device=dev, verbose=False)
                val_accs.append(acc)
            if verbose and ((epoch + 1) % log_interval == 0 or epoch + 1 == epochs):
                msg = f"epoch {epoch + 1:>2}/{epochs}  loss {avg:.3f}"
                if acc is not None:
                    msg += f"  val_accuracy {acc:.1%}"
                print(msg)
        return {"loss": losses, "val_accuracy": val_accs or None, "device": str(dev)}

    def evaluate(
        self,
        data: Dataset | DataLoader,
        *,
        batch_size: int = 32,
        device: Any = None,
        verbose: bool = True,
    ) -> float:
        """Report classification accuracy (``argmax`` vs labels).

        Accepts a Dataset or DataLoader yielding ``(images, labels)``
        batches; returns ``correct / total`` as a float in ``0..1``.
        """
        dev = _resolve_device(device)
        self.to(dev)
        loader = _as_loader(data, batch_size=batch_size, shuffle=False)
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                pred = self(x.to(dev)).argmax(-1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / total if total else 0.0
        if verbose:
            print(f"accuracy: {acc:.1%} ({correct}/{total})")
        return acc

    def __repr__(self) -> str:
        return f"Model(name={self.name!r}, layers={len(self.layers)})"
