"""ONNX export + esp-dl quantization for espnn models.

This is the shipped version of the pipeline previously duplicated across the
experiment scripts (``flower_to_espdl.py``, ``stl10_to_espdl.py``,
``micro_to_espdl.py`` — not shipped)::

    from espnn.convert import convert

    report = convert(model, example_input, calib_loader, "model.espdl")

Steps: ``torch.onnx.export`` -> esp-dl friendly rewrite
(``Relu + Min(6)`` -> ``Clip``, ``Reshape`` -> ``Flatten``) ->
``espdl_quantize_onnx`` (ESP-PPQ) -> ``.espdl`` + report.

Requires the ``convert`` extra (``pip install espnnpy[convert]``).
``onnx`` / ``esp_ppq`` are imported lazily so ``import espnn`` stays light.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "convert",
    "export_onnx",
    "make_espdl_friendly",
    "quantize_onnx",
]

DEFAULT_TARGET = "esp32s3"
DEFAULT_QUANT_TYPE = "w8a8"
DEFAULT_OPSET = 13
DEFAULT_CALIB_STEPS = 16


def _require_onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "espnn.convert needs the 'convert' extra "
            "(pip install espnnpy[convert]): missing package 'onnx'"
        ) from exc
    return onnx


def _require_espdl_quantize() -> Any:
    try:
        from esp_ppq.api import espdl_quantize_onnx
    except ImportError as exc:
        raise ImportError(
            "espnn.convert needs the 'convert' extra "
            "(pip install espnnpy[convert]): missing package 'esp-ppq'"
        ) from exc
    return espdl_quantize_onnx


def export_onnx(
    model: Any,
    example_input: Any,
    onnx_path: str | Path,
    *,
    input_name: str = "input",
    output_name: str = "output",
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export ``model`` to ONNX (classic exporter, fixed opset).

    Returns the resolved ``onnx_path``. The model is left untouched
    (callers typically ``model.eval()`` first — see :func:`convert`).
    """
    import torch

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example_input,
        str(onnx_path),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        do_constant_folding=True,
    )
    return onnx_path


def make_espdl_friendly(model: Any) -> Any:
    """Rewrite ONNX exporter artifacts to ops esp-dl supports, in place.

    - ``Relu + Min(6)`` -> ``Clip(0, 6)`` (esp-dl has Clip, no Min)
    - ``Reshape`` (static) -> ``Flatten`` (old ppq executor chokes on Reshape)

    Takes and returns the ``onnx.ModelProto``. Runs ``onnx.checker`` first
    (import) and last (result) so failures surface here, not in quantization.
    """
    onnx = _require_onnx()
    onnx.checker.check_model(model)

    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    six = [n for n, v in inits.items() if v.shape == () and float(v) == 6.0]
    if six:
        six_name = six[0]
    else:
        import numpy as _np

        six_name = "clip_max_6"
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(_np.array(6.0, dtype=_np.float32), six_name)
        )
    zero_name = "clip_min_0"
    if zero_name not in inits:
        import numpy as _np

        model.graph.initializer.append(
            onnx.numpy_helper.from_array(_np.array(0.0, dtype=_np.float32), zero_name)
        )

    by_out = {}
    for n in model.graph.node:
        for o in n.output:
            by_out[o] = n
    relu_outs = set()
    for n in model.graph.node:
        if n.op_type == "Min" and six_name in n.input:
            key = n.input[0] if n.input[1] == six_name else n.input[1]
            relu = by_out.get(key, None)
            if relu is not None and relu.op_type == "Relu" and key == relu.output[0]:
                relu_outs.add(key)

    new_nodes = []
    for n in model.graph.node:
        if n.op_type == "Relu" and n.output[0] in relu_outs:
            continue  # fused into the Clip below
        if n.op_type == "Min" and six_name in n.input:
            key = n.input[0] if n.input[1] == six_name else n.input[1]
            relu = by_out.get(key, None)
            if relu is not None and key in relu_outs:
                new_nodes.append(
                    onnx.helper.make_node(
                        "Clip",
                        inputs=[relu.input[0], zero_name, six_name],
                        outputs=[n.output[0]],
                        name=(n.name + "_clip") if n.name else "",
                    )
                )
                continue
        if n.op_type == "Reshape":
            flat = onnx.helper.make_node(
                "Flatten",
                inputs=[n.input[0]],
                outputs=[n.output[0]],
                name=(n.name + "_flat") if n.name else "",
            )
            flat.attribute.append(onnx.helper.make_attribute("axis", 1))
            new_nodes.append(flat)
            continue
        new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    used = {i for n in model.graph.node for i in n.input}
    for i in list(model.graph.initializer):
        if i.name == "val_1" and i.name not in used:
            model.graph.initializer.remove(i)
    onnx.checker.check_model(model)
    return model


def _default_collate(batch: Any) -> Any:
    """Default PPQ collate: ``DataLoader(TensorDataset(x))`` yields ``[x]``."""
    if isinstance(batch, (list, tuple)) and len(batch) == 1:
        batch = batch[0]
    to_cpu = getattr(batch, "to", None)
    return to_cpu("cpu") if callable(to_cpu) else batch


def quantize_onnx(
    onnx_path: str | Path,
    espdl_path: str | Path,
    calib_loader: Any,
    input_shape: list[int] | tuple[int, ...],
    *,
    target: str = DEFAULT_TARGET,
    quant_type: str = DEFAULT_QUANT_TYPE,
    calib_steps: int = DEFAULT_CALIB_STEPS,
    collate_fn: Any = None,
    device: str = "cpu",
    error_report: bool = False,
    export_test_values: bool = False,
    verbose: int = 0,
) -> Any:
    """Quantize a (friendly) ONNX model via ESP-PPQ; returns the PPQ graph."""
    espdl_quantize_onnx = _require_espdl_quantize()
    espdl_path = Path(espdl_path)
    espdl_path.parent.mkdir(parents=True, exist_ok=True)
    return espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=calib_steps,
        input_shape=list(input_shape),
        target=target,
        quant_type=quant_type,
        collate_fn=collate_fn or _default_collate,
        device=device,
        error_report=error_report,
        export_test_values=export_test_values,
        verbose=verbose,
    )


def _input_quant_params(graph: Any, input_name: str) -> tuple[Any, Any]:
    """Best-effort input scale / zero-point from a quantized PPQ graph."""
    try:
        g_in = graph.inputs[input_name]
        cfg = g_in.dest_op_configs[0]
        sc, off = cfg.scale, cfg.offset
        import torch

        sc = float(sc.reshape(-1)[0]) if torch.is_tensor(sc) else float(sc)
        off = int(off.reshape(-1)[0]) if torch.is_tensor(off) else int(off)
        return sc, off
    except Exception:
        return None, None


def convert(
    model: Any,
    example_input: Any,
    calib_loader: Any,
    espdl_path: str | Path,
    *,
    onnx_path: str | Path | None = None,
    report_path: str | Path | None = "auto",
    input_shape: list[int] | tuple[int, ...] | None = None,
    input_name: str = "input",
    output_name: str = "output",
    opset: int = DEFAULT_OPSET,
    target: str = DEFAULT_TARGET,
    quant_type: str = DEFAULT_QUANT_TYPE,
    calib_steps: int = DEFAULT_CALIB_STEPS,
    collate_fn: Any = None,
    device: str = "cpu",
    verbose: int = 0,
) -> dict:
    """Export ``model`` and quantize to ``.espdl``; return a JSON-able report.

    - ``example_input``: dummy tensor for ``torch.onnx.export``
      (also derives ``input_shape`` when omitted).
    - ``calib_loader``: calibration batches (e.g. ``DataLoader(TensorDataset)``).
    - ``espdl_path``: where to write the ``.espdl`` file.
    - ``onnx_path``: defaults to ``espdl_path`` with a ``.onnx`` suffix.
    - ``report_path``: ``"auto"`` writes ``espdl_report.json`` next to the
      ``.espdl`` file (scratch convention); pass a path to override,
      ``None`` to skip writing (the report dict is still returned).
    """
    onnx = _require_onnx()
    espdl_path = Path(espdl_path)
    if onnx_path is None:
        onnx_path = espdl_path.with_suffix(".onnx")
    else:
        onnx_path = Path(onnx_path)
    if input_shape is None:
        input_shape = [int(d) for d in example_input.shape]

    model.eval()
    export_onnx(
        model,
        example_input,
        onnx_path,
        input_name=input_name,
        output_name=output_name,
        opset=opset,
    )
    friendly = onnx.load(str(onnx_path))
    make_espdl_friendly(friendly)
    onnx.save(friendly, str(onnx_path))

    graph = quantize_onnx(
        onnx_path,
        espdl_path,
        calib_loader,
        input_shape,
        target=target,
        quant_type=quant_type,
        calib_steps=calib_steps,
        collate_fn=collate_fn,
        device=device,
        verbose=verbose,
    )
    scale, zero_point = _input_quant_params(graph, input_name)
    report = {
        "onnx_path": str(onnx_path),
        "espdl_path": str(espdl_path),
        "espdl_bytes": espdl_path.stat().st_size,
        "opset": opset,
        "target": target,
        "quant_type": quant_type,
        "calib_steps": calib_steps,
        "input_name": input_name,
        "input_shape": list(input_shape),
        "input_scale": scale,
        "input_zero_point": zero_point,
    }
    if report_path == "auto":
        report_path = espdl_path.parent / "espdl_report.json"
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    return report
