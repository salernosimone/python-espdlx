"""ONNX export + esp-dl quantization for espdlx models.

This is the shipped version of the pipeline previously duplicated across the
experiment scripts (``flower_to_espdl.py``, ``stl10_to_espdl.py``,
``micro_to_espdl.py`` — not shipped)::

    from espdlx.convert import convert

    report = convert(
        model=model.cpu(),
        example_input=example_input,
        calib_loader=calib_loader,
        out_dir="deploy/my_model",
    )

Steps: ``torch.onnx.export`` -> esp-dl friendly rewrite
(``Relu + Min(6)`` -> ``Clip``, ``Reshape`` -> ``Flatten``) ->
``espdl_quantize_onnx`` (ESP-PPQ) -> ``.espdl`` + Arduino ``.h`` header + report.

``out_dir`` receives ``<name>.onnx``, ``<name>.espdl``, ``<name>.h``
(ready to drop next to an Arduino sketch — no ``xxd`` needed) and
``espdl_report.json``.

Requires the ``convert`` extra (``pip install espdlx[convert]``).
``onnx`` / ``esp_ppq`` are imported lazily so ``import espdlx`` stays light.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "convert",
    "export_onnx",
    "make_espdl_friendly",
    "fold_gemm_alpha_beta",
    "quantize_onnx",
    "write_header",
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
            "espdlx.convert needs the 'convert' extra "
            "(pip install espdlx[convert]): missing package 'onnx'"
        ) from exc
    return onnx


def _require_espdl_quantize() -> Any:
    try:
        from esp_ppq.api import espdl_quantize_onnx
    except ImportError as exc:
        raise ImportError(
            "espdlx.convert needs the 'convert' extra "
            "(pip install espdlx[convert]): missing package 'esp-ppq'"
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


def _get_attr(node, name, default=None):
    """Read an ONNX node attribute (or ``default`` when absent)."""
    onnx = _require_onnx()
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def _set_attr(node, name, value):
    """Set an ONNX node attribute, replacing any existing one."""
    onnx = _require_onnx()
    for a in list(node.attribute):
        if a.name == name:
            node.attribute.remove(a)
    node.attribute.append(onnx.helper.make_attribute(name, value))


def _add_scaled_initializer(model, onnx, base_name: str, value, factor: float) -> str:
    """Append ``alpha * value`` as a fresh initializer; returns its name."""
    import numpy as _np

    scaled = (factor * value).astype(value.dtype)
    name = f"{base_name}_gemm_{factor}"
    if any(i.name == name for i in model.graph.initializer):
        name = f"{name}_{len(model.graph.initializer)}"
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(_np.asarray(scaled), name)
    )
    return name


def fold_gemm_alpha_beta(model) -> None:
    """Fold ONNX ``Gemm`` alpha/beta attributes into its weight/bias inputs.

    ESP-PPQ asserts ``alpha == beta == 1.0`` when exporting ``dl::Gemm``, so
    non-trivial scales must be baked into the parameters: ``alpha * (A·B) +
    beta * C`` == ``A · (alpha*B) + (beta*C)`` in float.

    Handles ``B`` given directly as an initializer (``transB=0`` from
    ``torch.addmm``) or as a ``Transpose`` / ``Constant`` of one (``transB=1``
    with non-plain scales). Unsupported shapes raise ``ValueError``.
    Mutates ``model`` in place.
    """
    onnx = _require_onnx()
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    by_out: dict = {}
    for n in model.graph.node:
        for o in n.output:
            by_out.setdefault(o, []).append(n)
    consumed: dict = {}
    for n in model.graph.node:
        for i in n.input:
            consumed.setdefault(i, []).append(n)

    dropped = set()  # ids of producer nodes redirected away from
    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue
        alpha = float(_get_attr(node, "alpha", 1.0))
        beta = float(_get_attr(node, "beta", 1.0))
        if alpha == 1.0 and beta == 1.0:
            continue

        if alpha != 1.0:
            b_name = node.input[1]
            producer = (by_out.get(b_name) or [None])[0]
            if b_name in inits:
                node.input[1] = _add_scaled_initializer(
                    model, onnx, b_name, inits[b_name], alpha
                )
            elif producer is not None and producer.op_type == "Transpose" and producer.input[0] in inits:
                val = inits[producer.input[0]]
                perm = _get_attr(producer, "perm", None)
                if perm is not None:
                    val = val.transpose(tuple(perm))
                else:
                    val = val.transpose()
                node.input[1] = _add_scaled_initializer(
                    model, onnx, producer.input[0], val, alpha
                )
                if len(consumed.get(b_name, [])) == 1:
                    dropped.add(id(producer))
            elif producer is not None and producer.op_type == "Constant":
                val = onnx.numpy_helper.to_array(_get_attr(producer, "value"))
                node.input[1] = _add_scaled_initializer(
                    model, onnx, producer.output[0], val, alpha
                )
                if len(consumed.get(b_name, [])) == 1:
                    dropped.add(id(producer))
            else:
                raise ValueError(
                    "espdlx.convert: cannot fold Gemm alpha — input B is neither "
                    "an initializer nor a Transpose/Constant of one (got "
                    f"{b_name!r} produced by "
                    f"{producer.op_type if producer is not None else 'nothing'})"
                )

        if beta != 1.0 and len(node.input) > 2 and node.input[2]:
            c_name = node.input[2]
            if c_name in inits:
                node.input[2] = _add_scaled_initializer(
                    model, onnx, c_name, inits[c_name], beta
                )
            else:
                raise ValueError(
                    "espdlx.convert: cannot fold Gemm beta — input C is not an "
                    f"initializer (got {c_name!r})"
                )

        _set_attr(node, "alpha", 1.0)
        _set_attr(node, "beta", 1.0)

    if dropped:
        kept = []
        for n in model.graph.node:
            if id(n) in dropped:
                continue
            kept.append(n)
        del model.graph.node[:]
        model.graph.node.extend(kept)


def _copy_proto(model, onnx):
    """Deep-copy an ``onnx.ModelProto`` without mutating the original."""
    clone = onnx.ModelProto()
    clone.ParseFromString(model.SerializeToString())
    return clone


def _is_flatten_reshape(node, by_out: dict, inits: dict, shapes: dict) -> bool:
    """Whether a ``Reshape`` node only flattens (``axis=1`` equivalent).

    True when the static target resolves (``0`` copies the input dim,
    ``-1`` infers) to ``(N, prod_rest)`` with the batch dim untouched —
    that is exactly ``Flatten(axis=1)``. Anything else (unknown shapes,
    non-constant shape input, genuine rank juggling) returns False and the
    node is preserved for ``dl::Reshape``.
    """
    if len(node.input) < 2:
        return False
    data, shape_in = node.input[0], node.input[1]
    in_shape = shapes.get(data)
    if not in_shape or len(in_shape) < 2:
        return False
    target = None
    if shape_in in inits:
        target = [int(v) for v in inits[shape_in].flatten().tolist()]
    else:
        producer = by_out.get(shape_in)
        if producer is not None and producer.op_type == "Constant":
            for a in producer.attribute:
                if a.name == "value":
                    onnx = _require_onnx()
                    target = [
                        int(v)
                        for v in onnx.numpy_helper.to_array(
                            onnx.helper.get_attribute_value(a)
                        )
                        .flatten()
                        .tolist()
                    ]
    if not target:
        return False
    if target.count(-1) > 1 or any(d < -1 for d in target):
        return False
    resolved = []
    for i, d in enumerate(target):
        if d == 0:
            if i >= len(in_shape):
                return False
            resolved.append(in_shape[i])
        else:
            resolved.append(d)
    total = 1
    for d in in_shape:
        total *= d
    if -1 in resolved:
        known = 1
        for d in resolved:
            if d != -1:
                known *= d
        if known == 0 or total % known != 0:
            return False
        resolved[resolved.index(-1)] = total // known
    else:
        prod = 1
        for d in resolved:
            prod *= d
        if prod != total:
            return False
    if len(resolved) != 2 or resolved[0] != in_shape[0]:
        return False
    rest = 1
    for d in in_shape[1:]:
        rest *= d
    return resolved[1] == rest


def _unbake_reshape_batch(model, full_shapes: dict) -> None:
    """Restore a leading ``-1`` the exporter folded to the static batch.

    ``Reshape`` targets are batch-agnostic by construction (see
    :class:`espdlx.layers.Reshape`: per-sample shape, leading ``-1``), but
    the torch exporter constant-folds that ``-1`` to the example batch
    (observed when the reshape input comes from a ``Transpose``) — which
    then breaks calibration at any other batch size. When the static shape
    input ``[s0, ...]`` has ``s0`` equal to the statically-known input
    batch, put the ``-1`` back: identical at batch-1, correct everywhere
    else. Shared shape initializers are duplicated before mutation.
    Untouched otherwise. Mutates ``model`` in place.
    """
    onnx = _require_onnx()
    import numpy as _np

    uses: dict = {}
    for n in model.graph.node:
        for i in n.input:
            uses[i] = uses.get(i, 0) + 1
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        data, shape_in = node.input[0], node.input[1]
        in_shape = full_shapes.get(data)
        if not in_shape:
            continue
        target_init = None
        for init in model.graph.initializer:
            if init.name == shape_in:
                target_init = init
                break
        if target_init is None:
            continue  # Constant-producer shapes are already dynamic-safe here
        target = _np.asarray(onnx.numpy_helper.to_array(target_init)).flatten()
        if len(target) < 2 or int(target[0]) != in_shape[0]:
            continue  # -1/0 batch already, or genuinely batch-coupled
        if uses.get(shape_in, 0) > 1:
            clone_name = f"{shape_in}_unbaked"
            if not any(i.name == clone_name for i in model.graph.initializer):
                model.graph.initializer.append(
                    onnx.numpy_helper.from_array(
                        _np.asarray(target).astype(_np.int64), clone_name
                    )
                )
            node.input[1] = clone_name
            target_init = next(
                i for i in model.graph.initializer if i.name == clone_name
            )
            target = _np.asarray(onnx.numpy_helper.to_array(target_init)).flatten()
        patched = _np.asarray(target).astype(_np.int64)
        patched[0] = -1
        target_init.CopyFrom(
            onnx.numpy_helper.from_array(patched, target_init.name)
        )


def _normalize_resize_sizes(model) -> None:
    """Rewrite ``Resize`` nodes that carry a ``sizes`` input to ``scales``.

    ESP-PPQ's calibration executor mishandles 4-input ``Resize`` (its
    ``Resize_forward`` expects ``values`` to line up as x/scales/... and fails
    on sizes-only nodes), while 3-input x/roi/scales is proven end-to-end.
    Replacing ``sizes`` by ``sizes / input_shape`` is exact: the output pixels
    are integers, so ``floor(in * sizes/in) == sizes`` for every axis.

    Needs static input shape (via ONNX shape inference); untouched otherwise.
    """
    onnx = _require_onnx()
    import numpy as np

    inferred = onnx.shape_inference.infer_shapes(_copy_proto(model, onnx))
    shapes: dict = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if all(d > 0 for d in dims):
            shapes[vi.name] = dims
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        if len(node.input) < 4 or not node.input[3]:
            continue  # scales-based (3 inputs) or no sizes at all
        in_name = node.input[0]
        sizes_name = node.input[3]
        if in_name not in shapes or sizes_name not in inits:
            continue
        sizes = np.asarray(inits[sizes_name], dtype=np.float32)
        in_shape = np.asarray(shapes[in_name], dtype=np.float32)
        if len(sizes) != len(in_shape):
            continue
        scales = sizes / in_shape
        name = f"{sizes_name}_scales"
        if not any(i.name == name for i in model.graph.initializer):
            model.graph.initializer.append(
                onnx.numpy_helper.from_array(scales.astype(np.float32), name)
            )
        node.input[2] = name
        del node.input[3]  # drop the sizes slot (scales is authoritative now)


def make_espdl_friendly(model: Any) -> Any:
    """Rewrite ONNX exporter artifacts to ops esp-dl supports, in place.

    - ``Relu + Min(6)`` -> ``Clip(0, 6)`` (esp-dl has Clip, no Min)
    - ``Reshape`` -> ``Flatten`` but ONLY when flatten-equivalent
      (``(N, prod_rest)``); genuine reshapes (attention head split/merge)
      are preserved — the runtime has ``dl::Reshape``
    - ``Gemm`` alpha/beta -> folded into weight/bias initializers
      (ESP-PPQ asserts plain ``alpha == beta == 1``)
    - ``RMSNorm`` composite (``Pow``/``ReduceMean``/``Add``/``Sqrt``/``Div``/
      ``Mul``) is left as standard ONNX — ESP-PPQ's importer fuses it into a
      native ``RMSNormalization`` (``FORMATTER_FUSE_RMSNORM``) at quantization
    - ``Reshape`` leading dim folded to the static batch (exporter artifact
      when the input comes from a ``Transpose``) is restored to ``-1`` —
      identical at batch-1, correct at calibration batch-N
    - ``Resize`` ``sizes`` -> ``scales`` (ESP-PPQ's executor only runs
      scales-based Resize)

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
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    # Static input shapes for the Reshape-keep decision (best effort).
    try:
        inferred = onnx.shape_inference.infer_shapes(_copy_proto(model, onnx))
        full_shapes: dict = {}
        for vi in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        ):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if all(d > 0 for d in dims):
                full_shapes[vi.name] = dims
    except Exception:
        full_shapes = {}
    _unbake_reshape_batch(model, full_shapes)
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
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
            if _is_flatten_reshape(n, by_out, inits, full_shapes):
                flat = onnx.helper.make_node(
                    "Flatten",
                    inputs=[n.input[0]],
                    outputs=[n.output[0]],
                    name=(n.name + "_flat") if n.name else "",
                )
                flat.attribute.append(onnx.helper.make_attribute("axis", 1))
                new_nodes.append(flat)
                continue
            new_nodes.append(n)  # genuine reshape (head split/merge) — keep
            continue
        new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    fold_gemm_alpha_beta(model)
    _normalize_resize_sizes(model)

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


def _sanitize_identifier(name: str) -> str:
    """Map ``name`` to a valid C identifier (fallback ``"model"``)."""
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(name))
    cleaned = cleaned.strip("_") or "model"
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def write_header(
    espdl_bytes: bytes,
    header_path: str | Path,
    var_name: str,
    *,
    bytes_per_line: int = 16,
) -> Path:
    """Write ``espdl_bytes`` as a C array header for Arduino sketches.

    The header is self-contained (``#pragma once`` + ``stdint.h``) so it can
    sit next to a ``.ino`` file with no extra tooling::

        #include "my_model.h"  // const unsigned char my_model_espdl[N] = {...};

    Returns the resolved ``header_path``.
    """
    header_path = Path(header_path)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#pragma once",
        "// Generated by espdlx.convert — do not edit.",
        "#include <stdint.h>",
        "",
        f"const unsigned char {var_name}[{len(espdl_bytes)}] = {{",
    ]
    for i in range(0, len(espdl_bytes), bytes_per_line):
        chunk = espdl_bytes[i : i + bytes_per_line]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines += [
        "};",
        f"const unsigned int {var_name}_len = {len(espdl_bytes)};",
        "",
    ]
    header_path.write_text("\n".join(lines))
    return header_path


def convert(
    model: Any,
    example_input: Any,
    calib_loader: Any,
    out_dir: str | Path,
    *,
    name: str | None = None,
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
    - ``out_dir``: folder receiving ``<name>.onnx``, ``<name>.espdl``,
      ``<name>.h`` (Arduino-ready C array, no ``xxd`` needed) and
      ``espdl_report.json``.
    - ``name``: file/symbol stem; defaults to ``model.name`` for
      :class:`espdlx.Model`, else ``"model"``.
    """
    onnx = _require_onnx()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = getattr(model, "name", "model") or "model"
    stem = _sanitize_identifier(name)
    onnx_path = out_dir / f"{stem}.onnx"
    espdl_path = out_dir / f"{stem}.espdl"
    header_path = out_dir / f"{stem}.h"
    report_path = out_dir / "espdl_report.json"
    var_name = f"{stem}_espdl"
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
    espdl_bytes = espdl_path.read_bytes()
    write_header(espdl_bytes, header_path, var_name)
    scale, zero_point = _input_quant_params(graph, input_name)
    report = {
        "name": stem,
        "out_dir": str(out_dir),
        "onnx_path": str(onnx_path),
        "espdl_path": str(espdl_path),
        "header_path": str(header_path),
        "header_var": var_name,
        "espdl_bytes": len(espdl_bytes),
        "opset": opset,
        "target": target,
        "quant_type": quant_type,
        "calib_steps": calib_steps,
        "input_name": input_name,
        "input_shape": list(input_shape),
        "input_scale": scale,
        "input_zero_point": zero_point,
    }
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2))
    return report
