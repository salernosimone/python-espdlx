"""espdlx: esp-dl friendly NN blocks for ESP32-S3 (PyTorch backend)."""

__version__ = "0.2.0"

from .graph import Model

from .recipes.kws import train_wakeword_detection

__all__ = ["Model", "__version__", "train_wakeword_detection"]
