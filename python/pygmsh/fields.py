"""Composable, NumPy-vectorized mesh-size fields compatible with pygmsh's concepts."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class Distance:
    def __init__(self, points=None, nodes_list=None, **_):
        points = points if points is not None else nodes_list
        if points is None:
            raise ValueError("Distance needs points or nodes_list")
        if np.asarray(points).size == 0:
            raise ValueError("Distance needs at least one point")
        self.points = _points(points)

    def evaluate(self, x):
        x = _points(x)
        if not len(x):
            return np.empty(0, dtype=np.float64)
        d2 = ((x[:, None, :] - self.points[None, :, :]) ** 2).sum(axis=2)
        return np.sqrt(d2.min(axis=1))


class Threshold:
    def __init__(self, field=None, lcmin=None, lcmax=None, distmin=None, distmax=None,
                 in_field=None, **kwargs):
        self.field = field if field is not None else in_field
        self.lcmin = float(lcmin if lcmin is not None else kwargs.get("LcMin"))
        self.lcmax = float(lcmax if lcmax is not None else kwargs.get("LcMax"))
        self.distmin = float(distmin if distmin is not None else kwargs.get("DistMin"))
        self.distmax = float(distmax if distmax is not None else kwargs.get("DistMax"))
        if self.field is None:
            raise ValueError("Threshold needs a source field")
        if self.distmax <= self.distmin:
            raise ValueError("distmax must be greater than distmin")

    def evaluate(self, x):
        x = _points(x)
        if not len(x):
            return np.empty(0, dtype=np.float64)
        if isinstance(self.field, Distance):
            result = np.empty(len(x), dtype=np.float64)
            lib().mpg_threshold_field(addr(x, np.float64), len(x), addr(self.field.points, np.float64),
                                      len(self.field.points), self.lcmin, self.lcmax, self.distmin,
                                      self.distmax, addr(result, np.float64))
            return result
        d = f64(self.field.evaluate(x))
        return np.interp(d, [self.distmin, self.distmax], [self.lcmin, self.lcmax])


class Min:
    def __init__(self, fields, **_): self.fields = list(fields)
    def evaluate(self, x): return _combine(self.fields, x, "mpg_min_field")


class Max:
    def __init__(self, fields, **_): self.fields = list(fields)
    def evaluate(self, x): return _combine(self.fields, x, "mpg_max_field")


def _combine(fields, x, kernel):
    if not fields:
        raise ValueError("at least one field is required")
    points = _points(x)
    if not len(points):
        return np.empty(0, dtype=np.float64)
    values = f64([f.evaluate(points) for f in fields])
    result = np.empty(len(points), dtype=np.float64)
    getattr(lib(), kernel)(addr(values, np.float64), len(fields), len(result), addr(result, np.float64))
    return result


def _points(x):
    if hasattr(x, "x"):
        x = [x.x]
    elif isinstance(x, (list, tuple)) and x and hasattr(x[0], "x"):
        x = [point.x for point in x]
    x = f64(x)
    x = np.atleast_2d(x)
    if x.shape[1] == 2:
        x = np.column_stack((x, np.zeros(len(x))))
    if x.shape[1] != 3:
        raise ValueError("points must have two or three coordinates")
    return np.ascontiguousarray(x)
