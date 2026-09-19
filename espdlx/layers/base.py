"""Shared base for espdlx constrained blocks."""

from __future__ import annotations

import torch
from torch import nn


class Layer(nn.Module):
    def validate_shapes(self, in_shape: tuple) -> tuple:
        raise NotImplementedError
