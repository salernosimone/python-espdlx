"""One-call wake-word recipe: folder of audio -> trained ``.espdl`` + header.

Expected layout::

    data_dir/
      wakeword/*.wav|*.mp3|*.m4a   (one or more files, repetitions of the word)
      **/*.wav|*.mp3|*.m4a         (anything else, at any depth, is noise)

Usage::

    from espdlx.kws import train_wakeword_detection
    summary = train_wakeword_detection("data/kws_hey")
    # -> data/kws_hey/{hey.onnx, hey.espdl, hey.h, espdl_report.json, ...}

Pipeline: decode to 16 kHz mono -> isolate repetitions (DC-free energy,
``wakeword_duration`` windows) -> log-mel 40x98 + train norm -> train a
depthwise-separable zoo net (200 epochs, early stopping, checkpoint every
25 + best) with waveform + SpecAugment augmentation -> ``convert()`` into
the same root. Returns an accuracy summary dict.

Requires ``ffmpeg`` on PATH for compressed audio (``.wav`` works without it).
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}

ARCHS = ("dscnn_small", "dscnn_tiny", "mnv2_slim")


# --------------------------------------------------------------------------
# audio ingest
# --------------------------------------------------------------------------

def _decode_mono16k(path: Path, sample_rate: int, tmp: Path) -> np.ndarray:
    """Decode any supported audio file to float32 mono at ``sample_rate``."""
    if path.suffix.lower() == ".wav" and shutil.which("ffmpeg") is None:
        with wave.open(str(path), "rb") as w:
            raw = w.readframes(w.getnframes())
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() > 1:
                x = x.reshape(-1, w.getnchannels()).mean(axis=1)
            if w.getframerate() != sample_rate:
                raise ValueError(
                    f"{path}: {w.getframerate()} Hz wav without ffmpeg on PATH "
                    f"(need {sample_rate} Hz) — install ffmpeg or resample first")
            return x
    if shutil.which("ffmpeg") is None:
        raise ValueError(
            f"{path}: decoding {path.suffix} needs `ffmpeg` on PATH")
    out = tmp / (path.stem + f"_{sample_rate}.wav")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
           "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise ValueError(f"ffmpeg failed on {path}: {r.stderr.strip()}")
    with wave.open(str(out), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _frame_rms_dc_free(x: np.ndarray, sr: int, frame_s=0.03, hop_s=0.01):
    x = x - float(x.mean())
    fl, hop = int(frame_s * sr), int(hop_s * sr)
    if len(x) < fl:
        return np.array([np.sqrt(float((x ** 2).mean()))]), fl, hop
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    idx = np.arange((len(x) - fl) // hop + 1) * hop
    return np.sqrt((c[idx + fl] - c[idx]) / fl), fl, hop


def _isolate_repetitions(x: np.ndarray, sr: int, win: int,
                         min_sep_s=0.8, min_dur_s=0.25, merge_gap_s=0.15) -> list[int]:
    """Sample indices (centers) of energy bursts: DC-free RMS -> regions."""
    rms, fl, hop = _frame_rms_dc_free(x, sr)
    hop_s = hop / sr
    med, mx = float(np.median(rms)), float(rms.max())
    if mx < 1e-4:
        return []
    thr = max(med * 5.0, mx * 0.25, 0.03)
    active = rms > thr
    regions, i, nfr = [], 0, len(rms)
    while i < nfr:
        if active[i]:
            j = i
            while j < nfr and active[j]:
                j += 1
            regions.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for a, b in regions:
        if merged and a - merged[-1][1] <= int(merge_gap_s / hop_s):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    centers, min_len = [], int(min_dur_s / hop_s)
    for a, b in merged:
        if b - a < min_len:
            continue  # clicks / mouth noise
        c = int((a + b) // 2 * hop + fl // 2)
        if centers and (c - centers[-1]) / sr < min_sep_s:
            continue
        centers.append(min(c, len(x) - 1))
    return centers


def _take_window(x: np.ndarray, center: int, win: int) -> np.ndarray:
    seg = np.zeros(win, dtype=np.float32)
    s0, s1 = max(center - win // 2, 0), min(center + win // 2, len(x))
    seg[s0 - (center - win // 2):s0 - (center - win // 2) + (s1 - s0)] = x[s0:s1]
    return seg


# --------------------------------------------------------------------------
# features (log-mel 40 x T) + augmentation
# --------------------------------------------------------------------------

def _mel_matrix(sr: int, n_fft: int, n_mels: int, fmax: float) -> torch.Tensor:
    hz2mel = lambda h: 2595.0 * np.log10(1.0 + h / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(0.0), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f0, f1, f2 = bins[m - 1], bins[m], bins[m + 1]
        if f1 > f0:
            fb[m - 1, f0:f1] = (np.arange(f0, f1) - f0) / max(f1 - f0, 1)
        if f2 > f1:
            fb[m - 1, f1:f2] = (f2 - np.arange(f1, f2)) / max(f2 - f1, 1)
    return torch.from_numpy(fb).float()


def _logmel(wav: torch.Tensor, mel_fb: torch.Tensor, hann: torch.Tensor,
            n_fft: int, win: int, hop: int) -> torch.Tensor:
    fr = wav.unfold(0, win, hop)[: (len(wav) - win) // hop + 1]
    spec = torch.fft.rfft(fr * hann, n=n_fft)
    mel = (spec.abs() ** 2) @ mel_fb.T
    return torch.log(mel.T.clamp_min(1e-6))


def _aug_waveform(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if rng.random() < 0.8:
        s = rng.randint(-len(x) // 10, len(x) // 10)
        x = torch.roll(x, s)
        x[:s] = 0 if s > 0 else x[:s]
        if s < 0:
            x[s:] = 0
    if rng.random() < 0.5:
        x = x * rng.uniform(0.7, 1.3)
    if rng.random() < 0.8:
        snr = rng.uniform(5, 20)
        sig_p = (x ** 2).mean().clamp_min(1e-9)
        x = x + torch.randn_like(x) * torch.sqrt(sig_p / (10 ** (snr / 10)))
    return x.clamp(-1, 1)


def _aug_spec(m: torch.Tensor, rng: random.Random, n_frames: int, n_mels: int) -> torch.Tensor:
    m = m.clone()
    if rng.random() < 0.5:  # time masking
        w = rng.randint(1, max(n_frames // 8, 1))
        t = rng.randint(0, max(n_frames - w, 0))
        m[:, t:t + w] = 0
    if rng.random() < 0.5:  # frequency masking
        h = rng.randint(1, max(n_mels // 6, 1))
        f = rng.randint(0, max(n_mels - h, 0))
        m[f:f + h, :] = 0
    return m


def _macs_params(model, input_shape) -> tuple[int, int]:
    shape = tuple(input_shape)
    macs = 0
    for lyr in model.layers:
        tn = type(lyr).__name__
        if tn in ("Conv2d", "DepthwiseConv2d"):
            _, c, h, w = shape
            kh, kw = lyr.kernel
            sh, sw = lyr.stride
            ph, pw = lyr.padding
            oh = (h + 2 * ph - kh) // sh + 1
            ow = (w + 2 * pw - kw) // sw + 1
            groups = c if tn == "DepthwiseConv2d" else 1
            macs += (c // groups) * lyr.out_channels * kh * kw * oh * ow
        elif tn == "Linear":
            macs += lyr.in_features * lyr.out_features
        shape = lyr.validate_shapes(shape)
    return macs, sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _evaluate(model, X, y, device) -> dict:
    model.eval()
    pred = torch.log_softmax(model(X.to(device))[:, :2], dim=1).cpu().argmax(1)
    out = {"acc": (pred == y).float().mean().item()}
    mw, mn = y == 1, y == 0
    out["wake_recall"] = (pred[mw] == 1).float().mean().item() if mw.any() else 0.0
    out["noise_spec"] = (pred[mn] == 0).float().mean().item() if mn.any() else 0.0
    return out


# --------------------------------------------------------------------------
# recipe
# --------------------------------------------------------------------------

def train_wakeword_detection(
    data_dir,
    *,
    wakeword_duration: float = 1.0,
    sample_rate: int = 16000,
    n_mels: int = 40,
    arch: str = "dscnn_small",
    test_frac: float = 0.3,
    seed: int = 7,
    epochs: int = 200,
    patience: int = 20,
    min_delta: float = 0.01,
    batch_size: int = 32,
    lr: float = 2e-3,
    calib_steps: int = 16,
    name: str | None = None,
    device: str | None = None,
) -> dict:
    """Train a wake-word detector from a folder of audio; export ``.espdl``.

    :param data_dir: root with ``wakeword/`` (≥1 audio file, repetitions of
        the word; several files OK) — every other audio file below root is
        treated as noise.
    :param wakeword_duration: seconds per training window cut around each
        isolated repetition (also the noise split length).
    :param arch: one of ``"dscnn_small"`` (default, int8-safe),
        ``"dscnn_tiny"``, ``"mnv2_slim"``.
    :returns: accuracy summary dict (metrics, MACs/params, artifact paths).
    """
    from ..zoo import DSCNN, MobileNetV2

    root = Path(data_dir)
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    ww_dir = root / "wakeword"
    if not ww_dir.is_dir():
        raise ValueError(f"{root}: missing `wakeword/` folder with the wake word audio")
    ww_files = sorted(p for p in ww_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not ww_files:
        raise ValueError(f"{ww_dir}: no audio files "
                         f"({sorted(AUDIO_EXTS)}) — add at least one wake-word recording")
    noise_files = sorted(p for p in root.rglob("*")
                         if p.is_file() and p.suffix.lower() in AUDIO_EXTS
                         and ww_dir not in p.parents)
    if not noise_files:
        raise ValueError(f"{root}: no noise audio found outside `wakeword/` "
                         f"— add background/other-speech recordings for negatives")
    if not (0.2 <= wakeword_duration <= 5.0):
        raise ValueError(f"wakeword_duration must be 0.2..5 s, got {wakeword_duration}")

    tag = name or root.name.replace(" ", "_").replace("-", "_")
    win = int(wakeword_duration * sample_rate)
    tmp = Path(tempfile.mkdtemp(prefix="kws_recipe_"))
    try:
        pos, neg = [], []
        for f in ww_files:
            x = _decode_mono16k(f, sample_rate, tmp)
            centers = _isolate_repetitions(x, sample_rate, win)
            segs = [_take_window(x, c, win) for c in centers] if len(centers) >= 3 else \
                [x[i * win:(i + 1) * win] for i in range(len(x) // win)]
            pos.extend(segs)
        for f in noise_files:
            x = _decode_mono16k(f, sample_rate, tmp)
            neg.extend(x[i * win:(i + 1) * win] for i in range(len(x) // win))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if len(pos) < 4:
        raise ValueError(f"only {len(pos)} wake-word windows isolated — "
                         f"record more repetitions")
    if len(neg) < 4:
        raise ValueError(f"only {len(neg)} noise windows — add more noise audio")

    n_fft, win_s, hop_s = 512, int(0.025 * sample_rate), int(0.010 * sample_rate)
    n_frames = 1 + (win - win_s) // hop_s
    mel_fb = _mel_matrix(sample_rate, n_fft, n_mels, sample_rate / 2)
    hann = torch.hann_window(win_s)
    to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    feats_pos = torch.stack([_logmel(to_t(s - s.mean()), mel_fb, hann, n_fft, win_s, hop_s)
                             for s in pos]).unsqueeze(1)
    feats_neg = torch.stack([_logmel(to_t(s - s.mean()), mel_fb, hann, n_fft, win_s, hop_s)
                             for s in neg]).unsqueeze(1)
    mu = torch.cat([feats_pos, feats_neg]).mean()
    sd = torch.cat([feats_pos, feats_neg]).std().clamp_min(1e-6)
    feats_pos, feats_neg = (feats_pos - mu) / sd, (feats_neg - mu) / sd

    rng = random.Random(seed)
    ip, inn = list(range(len(feats_pos))), list(range(len(feats_neg)))
    rng.shuffle(ip)
    rng.shuffle(inn)
    n_te_pos, n_te_neg = max(1, int(len(ip) * test_frac)), max(1, int(len(inn) * test_frac))
    trp, tep = ip[n_te_pos:], ip[:n_te_pos]
    trn, ten = inn[n_te_neg:], inn[:n_te_neg]
    Xtr = torch.cat([feats_pos[trp], feats_neg[trn]])
    ytr = torch.cat([torch.ones(len(trp)), torch.zeros(len(trn))]).long()
    Xte = torch.cat([feats_pos[tep], feats_neg[ten]])
    yte = torch.cat([torch.ones(len(tep)), torch.zeros(len(ten))]).long()
    raw_tr = [to_t(pos[i]) for i in trp] + [to_t(neg[i]) for i in trn]

    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "mps" if torch.backends.mps.is_available() else "cpu"))
    builder = {"dscnn_small": DSCNN.Small, "dscnn_tiny": DSCNN.Tiny,
               "mnv2_slim": MobileNetV2.Slim}[arch]
    model = builder(num_classes=2, in_channels=1).to(dev)
    macs, params = _macs_params(builder(num_classes=2, in_channels=1),
                                (1, 1, n_mels, n_frames))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_pos_tr, n_neg_tr = len(trp), len(trn)
    repeat = min(max(round(n_neg_tr / max(n_pos_tr, 1)), 1), 8)
    best, best_ep, bad, stopped = -1.0, 0, 0, False
    # early stopping: val accuracy must improve by >= min_delta within
    # `patience` epochs, else stop and keep the best checkpoint.
    for ep in range(1, epochs + 1):
        idx = list(range(n_pos_tr, n_pos_tr + n_neg_tr)) + \
            [r for _ in range(repeat) for r in range(n_pos_tr)]
        rng.shuffle(idx)
        model.train()
        for i in range(0, len(idx), batch_size):
            bi = idx[i:i + batch_size]
            xb = torch.stack([_aug_spec(
                (_logmel(_aug_waveform(raw_tr[j], rng), mel_fb, hann,
                                   n_fft, win_s, hop_s) - mu) / sd,
                rng, n_frames, n_mels) for j in bi]).unsqueeze(1).to(dev)
            yb = torch.tensor([1 if j < n_pos_tr else 0 for j in bi],
                              dtype=torch.long).to(dev)
            opt.zero_grad()
            loss = F.nll_loss(torch.log_softmax(model(xb)[:, :2], dim=1), yb)
            loss.backward()
            opt.step()
        sched.step()
        val = _evaluate(model, Xte, yte, dev)["acc"]
        if val > best + min_delta:
            best, best_ep, bad = val, ep, 0
            torch.save({"state_dict": {k: v.detach().cpu()
                                       for k, v in model.state_dict().items()}},
                       ckpt_dir / f"{tag}_best.pt")
        else:
            bad += 1
        if ep % 25 == 0:
            torch.save({"state_dict": {k: v.detach().cpu()
                                       for k, v in model.state_dict().items()}},
                       ckpt_dir / f"{tag}_ep{ep}.pt")
        if bad >= patience:
            stopped = True
            break
    model.load_state_dict(torch.load(ckpt_dir / f"{tag}_best.pt",
                                     map_location="cpu", weights_only=False)["state_dict"])

    tr_m = _evaluate(model, Xtr, ytr, dev)
    te_m = _evaluate(model, Xte, yte, dev)

    from ..convert import convert
    ex = torch.zeros(1, 1, n_mels, n_frames)
    calib = DataLoader(TensorDataset(Xtr), batch_size=16)
    report = convert(model.cpu(), ex, calib, root, name=tag,
                     num_classes=2, calib_steps=calib_steps, device="cpu")

    return {"name": tag, "arch": arch, "data_dir": str(root),
            "n_wakeword_windows": len(pos), "n_noise_windows": len(neg),
            "train_pos": n_pos_tr, "train_neg": n_neg_tr,
            "test_pos": len(tep), "test_neg": len(ten),
            "epochs_run": ep, "best_epoch": best_ep, "stopped_early": stopped,
            "train_acc": round(tr_m["acc"], 4), "test_acc": round(te_m["acc"], 4),
            "wake_recall": round(te_m["wake_recall"], 4),
            "noise_spec": round(te_m["noise_spec"], 4),
            "macs": macs, "params": params,
            "checkpoint_dir": str(ckpt_dir),
            "onnx_path": report["onnx_path"], "espdl_path": report["espdl_path"],
            "header_path": report["header_path"]}
