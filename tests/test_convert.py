from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import espdlx
import espdlx.layers as L
from espdlx.convert import (
    _default_collate,
    convert,
    export_onnx,
    make_espdl_friendly,
    write_header,
)

try:
    import onnx as _onnx
except ImportError:
    _onnx = None

requires_onnx = pytest.mark.skipif(_onnx is None, reason="needs 'convert' extra (onnx)")


def _tiny_model():
    return espdlx.Model(
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
def test_export_onnx_espdlx_model(tmp_path):
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
def test_export_onnx_residual_model_has_add_node(tmp_path):
    model = espdlx.Model(
        [
            L.Conv2d(2, 2, 1),
            L.ReLU(),
            L.Conv2d(2, 2, 1),
            L.Add(input_index=0),
        ],
        name="residual",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 2, 4, 4), tmp_path / "res.onnx")
    m = _onnx.load(str(path))
    _onnx.checker.check_model(m)
    assert "Add" in {n.op_type for n in m.graph.node}


@requires_onnx
def test_export_onnx_mul_model_has_mul_node(tmp_path):
    model = espdlx.Model(
        [
            L.Conv2d(2, 2, 1),
            L.ReLU(),
            L.Mul(input_index=0),
            L.Mul(constant=0.5),
        ],
        name="mul",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 2, 4, 4), tmp_path / "m.onnx")
    m = _onnx.load(str(path))
    _onnx.checker.check_model(m)
    assert "Mul" in {n.op_type for n in m.graph.node}


@requires_onnx
def test_export_onnx_tier1_ops_present(tmp_path):
    model = espdlx.Model(
        [
            L.Conv2d(2, 2, 1),
            L.LeakyReLU(),
            L.Tanh(),
            L.Swish(),
            L.Elu(),
            L.HardSigmoid(),
            L.Clip(min_val=-1.0, max_val=1.0),
            L.Neg(),
            L.Exp(),
            L.Log(),
            L.Sqrt(),
            L.Sub(constant=0.5),
            L.Div(constant=2.0),
        ],
        name="tier1_unary",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 2, 2, 2), tmp_path / "t.onnx")
    m = _onnx.load(str(path))
    _onnx.checker.check_model(m)
    ops = {n.op_type for n in m.graph.node}
    # Swish lowers to Sigmoid+Mul (no native SiLU at opset 13); HardSigmoid
    # stays native. Log needs positive input at runtime — here shape-only.
    for expected in ["LeakyRelu", "Tanh", "Sigmoid", "Mul", "Elu",
                     "HardSigmoid", "Clip", "Neg", "Exp", "Log", "Sqrt",
                     "Sub", "Div"]:
        assert expected in ops, (expected, ops)


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
        model=model,
        example_input=torch.zeros(1, 1, 4, 4),
        calib_loader=calib,
        out_dir=tmp_path / "deploy",
        calib_steps=16,
    )
    assert Path(report["onnx_path"]).exists()
    assert Path(report["espdl_path"]).exists()
    assert Path(report["header_path"]).exists()
    assert Path(report["report_path"]).exists()
    assert report["name"] == "tiny"
    assert report["header_var"] == "tiny_espdl"
    assert report["espdl_bytes"] == len(b"espdl-bytes")
    assert report["input_shape"] == [1, 1, 4, 4]
    assert report["input_scale"] == pytest.approx(0.5)
    assert report["input_zero_point"] == 3
    assert report["target"] == "esp32s3"
    assert report["quant_type"] == "w8a8"
    header = Path(report["header_path"]).read_text()
    assert "const unsigned char tiny_espdl[11]" in header
    assert header.count("0x") == len(b"espdl-bytes")


def test_write_header_roundtrips_bytes(tmp_path):
    data = bytes(range(256))
    path = write_header(data, tmp_path / "m.h", "m_espdl")
    text = path.read_text()
    assert text.startswith("#pragma once\n")
    assert "const unsigned char m_espdl[256]" in text
    assert "const unsigned int m_espdl_len = 256;" in text
    assert text.count("0x") == 256
    assert "0x00" in text and "0xff" in text


def test_write_header_sanitizes_name_via_convert(tmp_path, monkeypatch):
    ppq_api = pytest.importorskip("esp_ppq.api")

    def stub_quantize(*, onnx_import_file, espdl_export_file, **kwargs):
        Path(espdl_export_file).write_bytes(b"\x00")
        return SimpleNamespace(inputs={})

    monkeypatch.setattr(ppq_api, "espdl_quantize_onnx", stub_quantize)

    model = _tiny_model()
    calib = DataLoader(TensorDataset(torch.zeros(4, 1, 4, 4)), batch_size=4)
    report = convert(
        model=model,
        example_input=torch.zeros(1, 1, 4, 4),
        calib_loader=calib,
        out_dir=tmp_path / "d",
        name="my model!",
    )
    assert report["name"] == "my_model"
    assert report["header_var"] == "my_model_espdl"
    assert Path(report["header_path"]).name == "my_model.h"


@requires_onnx
def test_export_onnx_tier2_ops_present(tmp_path):
    """Concat (multi-input in Model), Resize, Gemm all reach the ONNX graph."""
    model = espdlx.Model(
        [
            L.Conv2d(2, 4, 3, padding=1),
            L.ReLU(),
            L.Conv2d(4, 4, 3, padding=1),
            L.ReLU(),
            L.Concat(input_indices=[0, 1, 3], dim=1),
            L.Conv2d(12, 2, 1),
            L.Resize(scale_factor=2.0, mode="nearest"),
            L.Flatten(),
            L.Gemm(128, 4),
        ],
        name="tier2",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 2, 4, 4), tmp_path / "t2.onnx")
    m = _onnx.load(str(path))
    _onnx.checker.check_model(m)
    ops = {n.op_type for n in m.graph.node}
    assert "Concat" in ops and "Resize" in ops and "Gemm" in ops
    concat = next(n for n in m.graph.node if n.op_type == "Concat")
    assert len(concat.input) == 3  # the multi-input dense concat
    assert next(a for a in concat.attribute if a.name == "axis").i == 1
    gemm = next(n for n in m.graph.node if n.op_type == "Gemm")
    gemm_attrs = {a.name: a for a in gemm.attribute}
    assert gemm_attrs["transB"].i == 1 and gemm_attrs["alpha"].f == 1.0


@requires_onnx
def test_export_onnx_resize_modes_present(tmp_path):
    model = espdlx.Model(
        [
            L.Resize(scale_factor=2.0, mode="nearest"),
            L.Resize(size=(16, 16), mode="bilinear"),
            L.Resize(scale_factor=(2.0, 3.0), mode="bicubic"),
        ],
        name="resizes",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 2, 4, 4), tmp_path / "r.onnx")
    m = _onnx.load(str(path))
    _onnx.checker.check_model(m)
    resizes = [n for n in m.graph.node if n.op_type == "Resize"]
    assert len(resizes) == 3
    modes = []
    for n in resizes:
        attr = {a.name: a for a in n.attribute}
        modes.append(attr["mode"].s.decode())
    assert modes == ["nearest", "linear", "cubic"]
    # scales vs sizes input: scale-based Resize uses scales (idx 2, roi empty),
    # size-based uses sizes (idx 3) with an empty scales slot (opset 13)
    assert resizes[0].input[2] and (len(resizes[0].input) < 4 or not resizes[0].input[3])
    assert not resizes[1].input[2] and resizes[1].input[3]


@requires_onnx
def test_make_espdl_friendly_normalizes_resize_sizes(tmp_path):
    """size= Resize becomes scales= (PPQ's executor runs only scales-based)."""
    onnx = _onnx
    model = espdlx.Model(
        [L.Conv2d(2, 4, 1), L.Resize(size=(16, 16), mode="bilinear")],
        name="rs",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 2, 4, 4), tmp_path / "rs.onnx")
    m = onnx.load(str(path))
    assert len(m.graph.node) == 2  # Conv + Resize (sizes at input 3 as exported)
    make_espdl_friendly(m)
    onnx.checker.check_model(m)
    resize = next(n for n in m.graph.node if n.op_type == "Resize")
    assert len(resize.input) == 3  # sizes slot dropped
    assert resize.input[2].endswith("_scales")
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    import numpy as np

    assert np.allclose(inits[resize.input[2]], [1.0, 1.0, 4.0, 4.0])


@requires_onnx
def test_make_espdl_friendly_folds_gemm_alpha_beta(tmp_path):
    """Non-trivial alpha/beta Gemm folds into weight/bias, attrs reset to 1."""
    onnx = _onnx
    model = espdlx.Model(
        [L.Flatten(), L.Gemm(64, 8, alpha=0.5, beta=2.0)],
        name="gemm_ab",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 1, 8, 8), tmp_path / "g.onnx")
    m = onnx.load(str(path))
    make_espdl_friendly(m)
    onnx.checker.check_model(m)
    gemm = next(n for n in m.graph.node if n.op_type == "Gemm")
    attrs = {a.name: a for a in gemm.attribute}
    assert attrs["alpha"].f == 1.0 and attrs["beta"].f == 1.0
    assert "Transpose" not in {n.op_type for n in m.graph.node}
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    w = inits[gemm.input[1]]
    b = inits[gemm.input[2]]
    assert w.shape == (64, 8)  # folded to ONNX B convention [in, out]
    assert b.shape == (8,)


@requires_onnx
def test_gemm_folded_graph_onnxruntime_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    onnx = _onnx
    model = espdlx.Model(
        [L.Flatten(), L.Gemm(64, 8, alpha=0.5, beta=2.0, transB=True)],
        name="gemm_ab_t",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 1, 8, 8), tmp_path / "g.onnx")
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    make_espdl_friendly(m)
    onnx.save(m, str(tmp_path / "g_friendly.onnx"))
    run = ort.InferenceSession(str(tmp_path / "g_friendly.onnx"), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 1, 8, 8)
    got = run.run(None, {run.get_inputs()[0].name: x.numpy()})[0]
    want = model(x).detach().numpy()
    assert np.abs(got - want).max() < 1e-5


@requires_onnx
def test_convert_end_to_end_tier2_with_stub_quantizer(tmp_path, monkeypatch):
    ppq_api = pytest.importorskip("esp_ppq.api")

    def stub_quantize(*, onnx_import_file, espdl_export_file, **kwargs):
        assert Path(onnx_import_file).exists()
        Path(espdl_export_file).write_bytes(b"tier2-espdl")
        return SimpleNamespace(inputs={})

    monkeypatch.setattr(ppq_api, "espdl_quantize_onnx", stub_quantize)

    model = espdlx.Model(
        [
            L.Conv2d(2, 4, 1),
            L.ReLU(),
            L.Concat(input_indices=[0, 1], dim=1),
            L.Resize(scale_factor=2.0, mode="nearest"),
            L.Mean(),
            L.Flatten(),
            L.Gemm(8, 3, alpha=0.5, beta=2.0),
        ],
        name="tier2",
    )
    calib = DataLoader(TensorDataset(torch.zeros(4, 2, 4, 4)), batch_size=2)
    report = convert(
        model=model,
        example_input=torch.zeros(1, 2, 4, 4),
        calib_loader=calib,
        out_dir=tmp_path / "deploy_t2",
    )
    assert report["name"] == "tier2"
    assert Path(report["espdl_path"]).read_bytes() == b"tier2-espdl"
    assert report["input_shape"] == [1, 2, 4, 4]


def test_default_collate_unwraps_singleton_batches():
    x = torch.zeros(2, 1, 4, 4)
    out = _default_collate([x])
    assert torch.equal(out, x.cpu())
    y = torch.zeros(2, 3)
    assert torch.equal(_default_collate(y), y)


def _tier3_model():
    return espdlx.Model(
        [
            L.Linear(8, 8),                  # 0
            L.Linear(8, 8),                  # 1
            L.Transpose((0, 2, 1)),          # 2
            L.MatMul(input_indices=[0, 2]),  # 3
            L.Softmax(dim=-1),               # 4
            L.Linear(4, 8),                  # 5
            L.MatMul(input_indices=[4, 5]),  # 6
            L.LayerNorm(8),                  # 7
            L.RMSNorm(8),                    # 8
            L.Reshape((4, 2, 4)),            # 9: head split
            L.Transpose((0, 2, 1, 3)),       # 10
            L.MatMul(input_indices=[10, 10]),  # 11: head scores
            L.Softmax(dim=-1),               # 12: head weights
            L.Reshape((4, 8)),               # 13: merge
            L.Slice(starts=[1], ends=[3], axes=[1]),  # 12
            L.Gather(indices=[0, 1], dim=1),  # 13
            L.Pad(padding=(0, 0, 1, 1)),     # 14
            L.Unsqueeze(1),                  # 15
            L.Squeeze(1),                    # 16
        ],
        name="tier3",
    )


@requires_onnx
def test_export_onnx_tier3_ops_present(tmp_path):
    """Attention pair + norms + head reshape + shape tail reach the graph."""
    torch.manual_seed(0)
    model = _tier3_model()
    model.eval()
    path = export_onnx(model, torch.zeros(1, 4, 8), tmp_path / "t3.onnx", opset=18)
    m = _onnx.load(str(path))
    _onnx.checker.check_model(m)
    assert m.opset_import[0].version >= 17  # LayerNormalization needs 17+
    ops = {n.op_type for n in m.graph.node}
    for expected in ["MatMul", "Transpose", "LayerNormalization", "Pow",
                     "ReduceMean", "Reshape", "Slice", "Gather", "Pad",
                     "Squeeze", "Unsqueeze", "Softmax"]:
        assert expected in ops, (expected, ops)
    # RMSNorm lowers to the Pow-headed composite PPQ fuses at import, with
    # negative ReduceMean axes (the rank-free fusion condition)
    mean = next(n for n in m.graph.node if n.op_type == "ReduceMean")
    inits = {i.name: _onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    assert list(inits[mean.input[1]].flatten()) == [-1]
    # all three attention pairs are activation@activation MatMuls (neither
    # input is a weight initializer — those come from the 3-D Linears)
    matmuls = [n for n in m.graph.node if n.op_type == "MatMul"]
    pairs = [n for n in matmuls
             if n.input[0] not in inits and n.input[1] not in inits]
    assert len(pairs) == 3


@requires_onnx
def test_export_onnx_split_node(tmp_path):
    """Split exports natively (opset 18 — opset 13 has no Split adapter)."""
    class SplitSum(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.split = L.Split(3, dim=1)

        def forward(self, x):
            a, b, c = self.split(x)
            return a + b + c

    mod = SplitSum().eval()
    path = export_onnx(mod, torch.zeros(1, 6, 4), tmp_path / "sp.onnx", opset=18)
    m = _onnx.load(str(path))
    _onnx.checker.check_model(m)
    split = next(n for n in m.graph.node if n.op_type == "Split")
    assert {a.name: a for a in split.attribute}["axis"].i == 1


@requires_onnx
def test_make_espdl_friendly_unbakes_reshape_batch(tmp_path):
    """Folded leading batch (exporter artifact) is restored to -1."""
    onnx = _onnx
    torch.manual_seed(0)
    model = espdlx.Model(
        [L.Transpose((0, 2, 1, 3)), L.Reshape((4, 8))],
        name="merge",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 2, 4, 4), tmp_path / "m.onnx", opset=18)
    m = onnx.load(str(path))
    reshape = next(n for n in m.graph.node if n.op_type == "Reshape")
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    assert list(inits[reshape.input[1]].flatten()) == [1, 4, 8]  # folded batch
    make_espdl_friendly(m)
    onnx.checker.check_model(m)
    reshape = next(n for n in m.graph.node if n.op_type == "Reshape")
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    assert list(inits[reshape.input[1]].flatten()) == [-1, 4, 8]


@requires_onnx
def test_make_espdl_friendly_keeps_head_reshape_rewrites_flat(tmp_path):
    """Genuine head reshapes survive; flatten-equivalents still fold."""
    onnx = _onnx
    torch.manual_seed(0)
    model = espdlx.Model(
        [L.Reshape((4, 2, 4)), L.Reshape((2, 16))],
        name="heads",
    )
    model.eval()
    path = export_onnx(model, torch.zeros(1, 4, 8), tmp_path / "h.onnx", opset=18)
    m = onnx.load(str(path))
    make_espdl_friendly(m)
    onnx.checker.check_model(m)
    assert {n.op_type for n in m.graph.node} == {"Reshape"}

    flat = espdlx.Model([L.Reshape((32,))], name="flat")
    flat.eval()
    path = export_onnx(flat, torch.zeros(1, 4, 8), tmp_path / "f.onnx", opset=18)
    m = onnx.load(str(path))
    assert "Reshape" in {n.op_type for n in m.graph.node}
    make_espdl_friendly(m)
    onnx.checker.check_model(m)
    assert {n.op_type for n in m.graph.node} == {"Flatten"}


@requires_onnx
def test_tier3_friendly_graph_onnxruntime_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    torch.manual_seed(0)
    model = _tier3_model()
    model.eval()
    path = export_onnx(model, torch.zeros(1, 4, 8), tmp_path / "t3.onnx", opset=18)
    m = _onnx.load(str(path))
    make_espdl_friendly(m)
    _onnx.checker.check_model(m)
    _onnx.save(m, str(tmp_path / "t3_friendly.onnx"))
    run = ort.InferenceSession(str(tmp_path / "t3_friendly.onnx"), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 4, 8)
    got = run.run(None, {run.get_inputs()[0].name: x.numpy()})[0]
    want = model(x).detach().numpy()
    assert np.abs(got - want).max() < 1e-5


@requires_onnx
def test_convert_end_to_end_tier3_with_stub_quantizer(tmp_path, monkeypatch):
    ppq_api = pytest.importorskip("esp_ppq.api")

    def stub_quantize(*, onnx_import_file, espdl_export_file, **kwargs):
        assert Path(onnx_import_file).exists()
        Path(espdl_export_file).write_bytes(b"tier3-espdl")
        return SimpleNamespace(inputs={})

    monkeypatch.setattr(ppq_api, "espdl_quantize_onnx", stub_quantize)

    torch.manual_seed(0)
    model = _tier3_model()
    calib = DataLoader(TensorDataset(torch.zeros(4, 4, 8)), batch_size=2)
    report = convert(
        model=model,
        example_input=torch.zeros(1, 4, 8),
        calib_loader=calib,
        out_dir=tmp_path / "deploy_t3",
        opset=18,
    )
    assert report["name"] == "tier3"
    assert Path(report["espdl_path"]).read_bytes() == b"tier3-espdl"
    assert report["input_shape"] == [1, 4, 8]


def test_write_header_with_metadata_emits_quant_recipe(tmp_path):
    from espdlx.convert import write_header

    path = write_header(
        bytes(range(16)),
        tmp_path / "m.h",
        "my_classifier_espdl",
        num_classes=3,
        input_shape=[3, 96, 96],
        input_scale=0.023529412,
        input_zero_point=0,
    )
    text = path.read_text()
    assert "static const int NUM_CLASSES = 3;" in text
    assert "static const int INPUT_SHAPE[] = {3, 96, 96};" in text
    assert "static const float INPUT_SCALE = 0.023529412f;" in text
    assert "static const int INPUT_ZERO_POINT = 0;" in text
    assert "static inline int8_t quantize(float v)" in text
    assert "static inline float dequantize(int8_t q)" in text
    assert text.count("0x") == 16  # metadata adds no hex-looking text


def test_write_header_without_metadata_stays_bare(tmp_path):
    from espdlx.convert import write_header

    text = (write_header(bytes(range(8)), tmp_path / "m.h", "m_espdl")).read_text()
    assert "num_classes" not in text
    assert "quantize" not in text


def test_infer_num_classes_from_2d_head():
    from espdlx.convert import _infer_num_classes

    torch.manual_seed(0)
    assert _infer_num_classes(_tiny_model(), torch.zeros(1, 1, 4, 4)) == 8
    spatial = espdlx.Model([L.Conv2d(1, 4, 1)], name="spatial")
    assert _infer_num_classes(spatial, torch.zeros(1, 1, 4, 4)) is None


def test_convert_report_and_header_carry_num_classes(tmp_path, monkeypatch):
    ppq_api = pytest.importorskip("esp_ppq.api")

    def stub_quantize(*, onnx_import_file, espdl_export_file, **kwargs):
        Path(espdl_export_file).write_bytes(b"cls-espdl")
        return SimpleNamespace(inputs={})

    monkeypatch.setattr(ppq_api, "espdl_quantize_onnx", stub_quantize)

    model = _tiny_model()  # head Linear(64, 8) -> (N, 8)
    calib = DataLoader(TensorDataset(torch.zeros(4, 1, 4, 4)), batch_size=2)
    report = convert(
        model=model,
        example_input=torch.zeros(1, 1, 4, 4),
        calib_loader=calib,
        out_dir=tmp_path / "deploy_cls",
    )
    assert report["num_classes"] == 8
    assert report["input_shape"] == [1, 1, 4, 4]  # full shape incl. batch
    header = Path(report["header_path"]).read_text()
    assert "static const int NUM_CLASSES = 8;" in header
    assert "static const int INPUT_SHAPE[] = {1, 4, 4};" in header  # no batch dim
