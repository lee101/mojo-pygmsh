"""Bridge to the optional GPU distance kernel."""

from __future__ import annotations

import ctypes
import os

import numpy as np

from ._lib import ROOT, addr

LIB = os.path.join(ROOT, "dist", "libmojo-pygmsh-gpu.so")
I = ctypes.c_int64
_handle = None
_available = None


def lib():
    global _handle
    if _handle is None:
        if not os.path.exists(LIB):
            return None
        try:
            _handle = ctypes.CDLL(LIB)
        except OSError:
            return None
        _handle.mpg_gpu_available.argtypes = []
        _handle.mpg_gpu_available.restype = I
        _handle.mpg_gpu_distance_field.argtypes = [I, I, I, I, I]
        _handle.mpg_gpu_distance_field.restype = I
    return _handle


def available():
    global _available
    if _available is None:
        handle = lib()
        _available = bool(handle is not None and handle.mpg_gpu_available())
    return _available


def distance(query: np.ndarray, sources: np.ndarray):
    if not available():
        return None
    result = np.empty(len(query), dtype=np.float64)
    rc = lib().mpg_gpu_distance_field(
        addr(query, np.float64), len(query), addr(sources, np.float64), len(sources),
        addr(result, np.float64),
    )
    return result if rc == 0 else None
