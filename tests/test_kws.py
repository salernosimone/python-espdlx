"""End-to-end test of the kws recipe on synthetic audio (fast, CPU)."""
import wave

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from espdlx.recipes.kws import train_wakeword_detection  # noqa: E402


def _write_wav(path, x, sr=16000):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _burst(sr=16000, dur=0.3, freq=880.0):
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = np.hanning(n)
    return 0.8 * np.sin(2 * np.pi * freq * t) * env


def test_train_wakeword_detection(tmp_path):
    sr = 16000
    ww = tmp_path / "wakeword"
    ww.mkdir()
    for k in range(2):  # multiple wakeword files supported
        x = np.zeros(4 * sr, dtype=np.float32)
        for start in (0.5, 1.5, 2.5, 3.5):
            b = _burst()
            x[int(start * sr):int(start * sr) + len(b)] += b
        _write_wav(ww / f"w{k}.wav", x)
    rng = np.random.RandomState(0)
    for k in range(2):
        _write_wav(tmp_path / f"noise{k}.wav",
                   0.2 * rng.randn(8 * sr).astype(np.float32))

    summary = train_wakeword_detection(
        tmp_path, arch="dscnn_tiny", epochs=4, patience=100,
        calib_steps=2, device="cpu")

    assert summary["n_wakeword_windows"] >= 6
    assert summary["n_noise_windows"] >= 8
    assert 0.0 <= summary["test_acc"] <= 1.0
    assert summary["epochs_run"] == 4
    assert not summary["stopped_early"]
    assert summary["macs"] > 0 and summary["params"] > 0
    import os
    for key in ("onnx_path", "espdl_path", "header_path"):
        assert os.path.exists(summary[key]), key
    assert os.path.exists(f"{summary['checkpoint_dir']}/{summary['name']}_best.pt")


def test_train_wakeword_detection_validates(tmp_path):
    import pytest as _pt
    with _pt.raises(ValueError, match="wakeword"):
        train_wakeword_detection(tmp_path)  # empty dir
    (tmp_path / "wakeword").mkdir()
    with _pt.raises(ValueError, match="no audio"):
        train_wakeword_detection(tmp_path)  # no audio in wakeword/
