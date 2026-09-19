"""Prebuilt espdlx architectures (the `.scratch` zoo, cleaned up).

All are residual-free variants of well-known families (espdlx has no runtime
tensor-tensor `Add`, so skip connections are omitted). Bottlenecks keep the
`1x1 expand -> 3x3 depthwise -> 1x1 project` shape with `ReLU6`.

Usage:
    from espdlx.zoo import MobileNetV2, MobileNetV1, VGG, DSCNN
    model = MobileNetV2.Slim(num_classes=5)   # flowers, 128px RGB
    model = MobileNetV2.Base(num_classes=5)   # bigger flowers / transfer backbone
    model = MobileNetV2.Fomo()                # EXPERIMENTAL grayscale centroid detector, 96px

`Linear` heads are padded to a multiple of 8 (ESP32-S3 SIMD limit). The true
class count is kept as `model.num_classes`; slice logits `[:, :num_classes]`
when training/evaluating (see `benchmark_cifar20.py` arms in `.scratch`).
"""

from __future__ import annotations

import espdlx
import espdlx.layers as L


def _pad8(n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError(f"num_classes must be >= 1, got {n}")
    return ((n + 7) // 8) * 8


def _mnv2_bottleneck(in_ch: int, exp_ch: int, out_ch: int, stride: int) -> list:
    """Inverted-residual bottleneck without the residual add."""
    return [
        L.Conv2d(in_ch, exp_ch, 1),
        L.ReLU6(),
        L.DepthwiseConv2d(exp_ch, exp_ch, 3, stride=stride, padding=1),
        L.ReLU6(),
        L.Conv2d(exp_ch, out_ch, 1),
    ]


def _cls_head(head_in: int, num_classes: int) -> list:
    """GAP + padded Linear + Softmax classifier head."""
    return [L.Mean(), L.Flatten(), L.Linear(head_in, _pad8(num_classes)), L.Softmax()]


def _gap_conv_head(head_in: int, num_classes: int) -> list:
    """1x1-conv + GAP classifier head (avoids the Linear %8 rule)."""
    return [L.Conv2d(head_in, num_classes, 1), L.Mean(), L.Flatten()]


def _cbr(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1) -> list:
    return [L.Conv2d(cin, cout, k, stride=s, padding=p), L.BatchNorm2d(cout), L.ReLU()]


def _ds(cin: int, cmid: int, s: int = 1) -> list:
    return (
        [L.DepthwiseConv2d(cin, cin, 3, stride=s, padding=1), L.BatchNorm2d(cin), L.ReLU()]
        + [L.Conv2d(cin, cmid, 1), L.BatchNorm2d(cmid), L.ReLU()]
    )


def _btn(cin: int, cmid: int, cout: int, s: int = 1) -> list:
    layers = [L.Conv2d(cin, cmid, 1), L.BatchNorm2d(cmid), L.ReLU()]
    layers += [L.Conv2d(cmid, cmid, 3, stride=s, padding=1), L.BatchNorm2d(cmid), L.ReLU()]
    layers += [L.Conv2d(cmid, cout, 1), L.BatchNorm2d(cout), L.ReLU()]
    return layers


class MobileNetV2:
    """Inverted-residual family, residual-free (`ReLU6`, no `Add`)."""

    SPECS_SLIM = [(16, 24, 16, 1), (16, 32, 24, 2), (24, 40, 24, 1), (24, 48, 32, 2), (32, 64, 32, 1)]
    SPECS_BASE = [(32, 96, 32, 1), (32, 128, 48, 2), (48, 160, 48, 1), (48, 192, 64, 2), (64, 224, 64, 1), (64, 256, 80, 2)]

    @staticmethod
    def Slim(num_classes: int = 5, in_channels: int = 3) -> espdlx.Model:
        """13.9k-param flower model (`examples/flower_mnv2.py`). Expects ~128px RGB."""
        layers: list = [L.Conv2d(in_channels, 16, 3, stride=2, padding=1), L.ReLU6()]
        for in_c, exp_c, out_c, st in MobileNetV2.SPECS_SLIM:
            layers += _mnv2_bottleneck(in_c, exp_c, out_c, st)
        layers += _cls_head(32, num_classes)
        model = espdlx.Model(layers, name="mnv2_slim")
        model.num_classes = int(num_classes)
        return model

    @staticmethod
    def Base(num_classes: int = 5, in_channels: int = 3) -> espdlx.Model:
        """~130k-param model (`flower_mnv2_full` / `mnv2_cifar100` backbone).

        Spatial-size agnostic (32px CIFAR pretrain through 128px flowers).
        """
        layers: list = [L.Conv2d(in_channels, 32, 3, stride=2, padding=1), L.ReLU6()]
        for in_c, exp_c, out_c, st in MobileNetV2.SPECS_BASE:
            layers += _mnv2_bottleneck(in_c, exp_c, out_c, st)
        layers += _cls_head(80, num_classes)
        model = espdlx.Model(layers, name="mnv2_base")
        model.num_classes = int(num_classes)
        return model

    @staticmethod
    def Fomo(
        grid: int = 12,
        expand: int = 48,
        nblocks: int = 4,
        base: int = 8,
        in_channels: int = 1,
        img_size: int = 96,
    ) -> espdlx.Model:
        """EXPERIMENTAL centroid detector inspired by Edge Impulse FOMO (not affiliated).

        Single-class grayscale detector ending at a `grid x grid` head with
        per-cell `[bg_logit, obj_logit]` logits (softmax is applied at
        export/decode, never during training).

        Status: synthetic unit tests only (`tests/test_fomo.py`); best real-data
        run so far reached P=0.03/R=0.28 — not a working detector yet.
        See `espdlx.fomo` for the matching loss/encode/decode helpers.
        """
        strides: list[int] = []
        size = img_size // 2
        while size // 2 >= grid:
            strides.append(2)
            size //= 2
        while len(strides) < nblocks:
            strides.append(1)

        layers: list = [L.Conv2d(in_channels, base, 3, stride=2, padding=1), L.BatchNorm2d(base), L.ReLU6()]
        in_ch = base
        for st in strides:
            layers += [
                L.Conv2d(in_ch, expand, 1), L.BatchNorm2d(expand), L.ReLU6(),
                L.DepthwiseConv2d(expand, expand, 3, stride=st, padding=1), L.BatchNorm2d(expand), L.ReLU6(),
                L.Conv2d(expand, base, 1), L.BatchNorm2d(base), L.ReLU6(),
            ]
            in_ch = base

        layers += [L.Conv2d(base, 32, 1), L.ReLU6()]
        layers += [L.Conv2d(32, 2, 1)]
        name = f"mnv2_fomo_g{grid}"
        model = espdlx.Model(layers, name=name)
        model.validate_shapes((1, in_channels, img_size, img_size))
        return model


class MobileNetV1:
    """Depthwise-separable family (`DW3x3 + 1x1`, `BN + ReLU`)."""

    @staticmethod
    def Slim(num_classes: int = 10, in_channels: int = 3) -> espdlx.Model:
        """`micro_archs.v1_ds_mbv1` for STL10 (expects 96px RGB)."""
        ls: list = []
        ls += _cbr(in_channels, 24, s=2)
        ls += _ds(24, 48)
        ls += _ds(48, 96, s=2)
        ls += _ds(96, 96)
        ls += _ds(96, 192)
        ls += _ds(192, 192, s=2)
        ls += _ds(192, 192)
        ls += _ds(192, 384)
        ls += _ds(384, 384, s=2)
        ls += _gap_conv_head(384, num_classes)
        model = espdlx.Model(ls, name="mnv1_slim")
        model.num_classes = int(num_classes)
        return model


class ResNet:
    @staticmethod
    def BottleneckTiny(num_classes: int = 10, in_channels: int = 3) -> espdlx.Model:
        """`micro_archs.v2_bottleneck`: plain `1x1/3x3/1x1` blocks, no skips. Expects 96px RGB."""
        ls: list = []
        ls += _cbr(in_channels, 64, s=2)
        ls += [L.MaxPool2d(2)]
        ls += _btn(64, 48, 128)
        ls += _btn(128, 96, 256, s=2)
        ls += _btn(256, 96, 256)
        ls += _btn(256, 192, 512, s=2)
        ls += _btn(512, 192, 512)
        ls += _gap_conv_head(512, num_classes)
        model = espdlx.Model(ls, name="resnet_bottleneck_tiny")
        model.num_classes = int(num_classes)
        return model


class VGG:
    """Plain-`3x3` baselines (`BN + ReLU`). Stride/Wide heads assume 96px input (6x6)."""

    @staticmethod
    def Small(num_classes: int = 10, in_channels: int = 3) -> espdlx.Model:
        """`micro_archs.v3_vgg_s`: narrow `3x3` pairs + MaxPool."""
        ls: list = []
        ls += _cbr(in_channels, 16) + _cbr(16, 16) + [L.MaxPool2d(2)]
        ls += _cbr(16, 32) + _cbr(32, 32) + [L.MaxPool2d(2)]
        ls += _cbr(32, 64) + _cbr(64, 64) + [L.MaxPool2d(2)]
        ls += _cbr(64, 128) + _cbr(128, 128) + [L.MaxPool2d(2)]
        ls += _gap_conv_head(128, num_classes)
        model = espdlx.Model(ls, name="vgg_small")
        model.num_classes = int(num_classes)
        return model

    @staticmethod
    def Stride(num_classes: int = 10, in_channels: int = 3) -> espdlx.Model:
        """`micro_archs.v4_vgg_stride`: stride-2 convs, conv-collapse head (no GAP)."""
        ls: list = []
        ls += _cbr(in_channels, 24) + _cbr(24, 24, s=2) + _cbr(24, 24)
        ls += _cbr(24, 48) + _cbr(48, 48, s=2) + _cbr(48, 48)
        ls += _cbr(48, 96) + _cbr(96, 96, s=2) + _cbr(96, 96)
        ls += _cbr(96, 192) + _cbr(192, 192, s=2) + _cbr(192, 192)
        ls += _cbr(192, 64, s=2)
        ls += [L.Conv2d(64, num_classes, 3, padding=0), L.Flatten()]
        model = espdlx.Model(ls, name="vgg_stride")
        model.num_classes = int(num_classes)
        return model

    @staticmethod
    def Wide(num_classes: int = 10, in_channels: int = 3) -> espdlx.Model:
        """`micro_archs.v5_wide`: few wide stages, `AvgPool(6)` head."""
        ls: list = []
        ls += _cbr(in_channels, 48, s=2) + [L.MaxPool2d(2)]
        ls += _cbr(48, 96) + _cbr(96, 128)
        ls += [L.MaxPool2d(2)]
        ls += _cbr(128, 192) + _cbr(192, 256)
        ls += [L.MaxPool2d(2)]
        ls += _cbr(256, 320)
        ls += [L.AvgPool2d(6), L.Conv2d(320, num_classes, 1), L.Flatten()]
        model = espdlx.Model(ls, name="vgg_wide")
        model.num_classes = int(num_classes)
        return model

    @staticmethod
    def Base(num_classes: int = 10, in_channels: int = 3) -> espdlx.Model:
        """`micro_archs.v6_vgg_base`: `stl10_vgg` scaled to ~250M MACs."""
        ls: list = []
        ls += _cbr(in_channels, 24) + [L.MaxPool2d(2)]
        ls += _cbr(24, 24) + _cbr(24, 48) + _cbr(48, 48) + [L.MaxPool2d(2)]
        ls += _cbr(48, 96) + _cbr(96, 96) + [L.MaxPool2d(2)]
        ls += _cbr(96, 192) + _cbr(192, 192) + [L.MaxPool2d(2)]
        ls += _gap_conv_head(192, num_classes)
        model = espdlx.Model(ls, name="vgg_base")
        model.num_classes = int(num_classes)
        return model

    @staticmethod
    def Large(num_classes: int = 10, in_channels: int = 3) -> espdlx.Model:
        """Original `stl10_train.stl10_vgg` (256ch, 96px RGB)."""
        ls: list = [
            L.Conv2d(in_channels, 32, 3, padding=1), L.BatchNorm2d(32), L.ReLU(),
            L.Conv2d(32, 32, 3, padding=1), L.BatchNorm2d(32), L.ReLU(),
            L.MaxPool2d(2),
            L.Conv2d(32, 64, 3, padding=1), L.BatchNorm2d(64), L.ReLU(),
            L.Conv2d(64, 64, 3, padding=1), L.BatchNorm2d(64), L.ReLU(),
            L.MaxPool2d(2),
            L.Conv2d(64, 128, 3, padding=1), L.BatchNorm2d(128), L.ReLU(),
            L.Conv2d(128, 128, 3, padding=1), L.BatchNorm2d(128), L.ReLU(),
            L.MaxPool2d(2),
            L.Conv2d(128, 256, 3, padding=1), L.BatchNorm2d(256), L.ReLU(),
            L.Conv2d(256, 256, 3, padding=1), L.BatchNorm2d(256), L.ReLU(),
            L.MaxPool2d(2),
        ]
        ls += _gap_conv_head(256, num_classes)
        model = espdlx.Model(ls, name="vgg_large")
        model.num_classes = int(num_classes)
        return model


class DSCNN:
    """Keyword-spotting models on `1x40xF` log-mel input."""

    @staticmethod
    def Tiny(num_classes: int = 2, in_channels: int = 1) -> espdlx.Model:
        """`examples/wake_mnv2.py` (1x40x32 synth demo). No trailing softmax by design."""
        layers: list = [
            L.Conv2d(in_channels, 16, 3, padding=1), L.ReLU6(),
            L.Conv2d(16, 32, 1),
            L.Mean(), L.Flatten(),
            L.Linear(32, _pad8(num_classes)),
        ]
        model = espdlx.Model(layers, name="dscnn_tiny")
        model.num_classes = int(num_classes)
        return model

    @staticmethod
    def Small(num_classes: int = 2, in_channels: int = 1) -> espdlx.Model:
        """`scripts/train_kws_tinyml4all.py` (1x40x98 real audio)."""
        layers: list = [
            L.Conv2d(in_channels, 16, 3, padding=1), L.ReLU6(),
            L.DepthwiseConv2d(16, 16, 3, stride=2, padding=1), L.ReLU6(),
            L.Conv2d(16, 32, 1), L.ReLU6(),
            L.DepthwiseConv2d(32, 32, 3, stride=2, padding=1), L.ReLU6(),
            L.Conv2d(32, 48, 1), L.ReLU6(),
            L.Mean(), L.Flatten(),
            L.Linear(48, _pad8(num_classes)),
            L.Softmax(),
        ]
        model = espdlx.Model(layers, name="dscnn_small")
        model.num_classes = int(num_classes)
        return model


__all__ = ["MobileNetV2", "MobileNetV1", "ResNet", "VGG", "DSCNN"]
