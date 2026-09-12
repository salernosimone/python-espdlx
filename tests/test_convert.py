from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import espnn
import espnn.layers as L
from espnn.convert import (
    _default_collate,
    convert,
    export_onnx,
    make_espdl_friendly,
)

try:
    import onnx as _onnx
except ImportError:
    _onnx = None

requires_onnx = pytest.mark.skipif(_onnx is None, reason="needs 'convert' extra (onnx)")


def _tiny_model():
    return espnn.Model(
        [
            L.Conv2d(1, 4, 1),
            L.ReLU(),
            L.Flatten(),
            L.Linear(64, 8),
        ],
        name="tiny",
    )


def _relu_min_reshape_model():
    onnx = pytest.importorskip("onnx")
    import numpy as np

    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
    relu = onnx.helper.make_node("Relu", inputs=["X"], outputs=["R"])
    six = onnx.helper.make_tensor("six", onnx.TensorProto.FLOAT, [], [6.0])
    mn = onnx.helper.make_node("Min", inputs=["R", "six"], outputs=["M"])
    shape = onnx.helper.make_tensor(
        "shape", onnx.TensorProto.INT64, [2], np.array([1, 4], dtype=np.int64)
    )
    rs = onnx.helper.make_node("Reshape", inputs=["M", "shape"], outputs=["Y"])
    graph = onnx.helper.make_graph(
        [relu, mn, rs], "t", [X], [Y], initializer=[six, shape]
    )
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])


@requires_onnx
def test_make_espdl_friendly_rewrites_relu_min_and_reshape():
    m = _relu_min_reshape_model()
    out = make_espdl_friendly(m)
    assert out is m
    ops = Counter(n.op_type for n in m.graph.node)
    assert ops == {"Clip": 1, "Flatten": 1}
    clip = next(n for n in m.graph.node if n.op_type == "Clip")
    assert clip.input[0] == "X"  # fused back to the Relu input


@requires_onnx
def test_export_onnx_espnn_model(tmp_path):
    model = _tiny_model()
    model.eval()
    path = export_onnx(model, torch.zeros(1, 1, 4, 4), tmp_path / "tiny.onnx")
    assert path.exists()
    m = _onnx.load(str(path))
    _onnx.checker.check_model(m)
    assert [i.name for i in m.graph.input] == ["input"]
    assert [o.name for o in m.graph.output] == ["output"]
    assert len(m.graph.node) > 0


@requires_onnx
def test_convert_end_to_end_with_stub_quantizer(tmp_path, monkeypatch):
    ppq_api = pytest.importorskip("esp_ppq.api")

    def stub_quantize(*, onnx_import_file, espdl_export_file, **kwargs):
        assert Path(onnx_import_file).exists()
        Path(espdl_export_file).write_bytes(b"espdl-bytes")
        return SimpleNamespace(
            inputs={
                "input": SimpleNamespace(
                    dest_op_configs=[
                        SimpleNamespace(
                            scale=torch.tensor([0.5]), offset=torch.tensor([3])
                        )
                    ]
                )
            }
        )

    monkeypatch.setattr(ppq_api, "espdl_quantize_onnx", stub_quantize)

    model = _tiny_model()
    calib = DataLoader(TensorDataset(torch.zeros(8, 1, 4, 4)), batch_size=4)
    report = convert(
        model,
        torch.zeros(1, 1, 4, 4),
        calib,
        tmp_path / "tiny.espdl",
        report_path=None,
    )
    assert Path(report["onnx_path"]).exists()
    assert Path(report["espdl_path"]).exists()
    assert report["espdl_bytes"] == len(b"espdl-bytes")
    assert report["input_shape"] == [1, 1, 4, 4]
    assert report["input_scale"] == pytest.approx(0.5)
    assert report["input_zero_point"] == 3
    assert report["target"] == "esp32s3"
    assert report["quant_type"] == "w8a8"


def test_default_collate_unwraps_singleton_batches():
    x = torch.zeros(2, 1, 4, 4)
    out = _default_collate([x])
    assert torch.equal(out, x.cpu())
    y = torch.zeros(2, 3)
    assert torch.equal(_default_collate(y), y)
