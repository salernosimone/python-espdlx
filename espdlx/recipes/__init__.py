"""Training recipes: folder-of-audio -> ``.espdl`` in one call."""

from .kws import ARCHS, train_wakeword_detection

__all__ = ["ARCHS", "train_wakeword_detection"]
