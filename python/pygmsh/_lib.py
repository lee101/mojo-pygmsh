"""ctypes bridge to the Mojo kernels; all output storage stays NumPy-owned."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYGMSH_LIB") or os.path.join(ROOT, "dist", "libmojo-pygmsh.so")
I = ctypes.c_int64
F = ctypes.c_double
_SIGNATURES = {
    "mpg_rectangle_mesh": ([F, F, F, F, I, I, I, I], None),
    "mpg_distance_field": ([I, I, I, I, I], None),
    "mpg_threshold_field": ([I, I, I, I, F, F, F, F, I], None),
    "mpg_min_field": ([I, I, I, I], None),
    "mpg_max_field": ([I, I, I, I], None),
}
_loaded = None


def build() -> str:
    if os.path.exists(LIB):
        return LIB
    command = shutil.which("mojo")
    if command is None:
        raise RuntimeError("Mojo is unavailable; run this project through pixi")
    proc = subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], text=True, capture_output=True)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib():
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = args, result
    return _loaded


def f64(value) -> np.ndarray:
    """Make a contiguous float64 array without silently losing precision."""
    array = np.asarray(value)
    if array.dtype.kind == "f":
        if array.dtype.itemsize > np.dtype(np.float64).itemsize:
            raise TypeError("float values wider than float64 are not supported")
    elif array.dtype.kind in "iu":
        # Integers beyond this range cannot be represented exactly by float64.
        limit = 1 << 53
        if array.size and (array.min() < -limit or array.max() > limit):
            raise TypeError("integer coordinates must be exactly representable as float64")
    else:
        raise TypeError("coordinates must be a real numeric array")
    return np.ascontiguousarray(array, dtype=np.float64)


def addr(value: np.ndarray, dtype: np.dtype | type | None = None) -> int:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise ValueError("FFI arrays must be contiguous NumPy arrays")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise TypeError(f"FFI array must have dtype {np.dtype(dtype)}")
    if value.size == 0:
        raise ValueError("empty arrays must not cross the FFI boundary")
    return int(value.ctypes.data)
