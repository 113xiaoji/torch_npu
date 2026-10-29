"""Load the native CPU simulation bridge (never an NPU backend)."""
from .build_native import load_native

__all__ = ["load_native"]
