"""The useful, portable 2-D subset of :mod:`pygmsh.geo`."""

from __future__ import annotations

from dataclasses import dataclass
import math

import meshio
import numpy as np

from ._lib import addr, f64, lib


@dataclass
class Point:
    x: tuple[float, float, float]
    mesh_size: float | None = None
    dim: int = 0


@dataclass
class Line:
    points: tuple[Point, Point]
    dim: int = 1


@dataclass
class CurveLoop:
    curves: list[Line]
    dim: int = 1


@dataclass
class PlaneSurface:
    curve_loop: CurveLoop
    holes: list | None = None
    dim: int = 2


class Polygon:
    dim = 2
    def __init__(self, points, mesh_size=None, holes=None, make_surface=True):
        points = _xyz(points)
        if len(points) < 3:
            raise ValueError("a polygon needs at least three points")
        sizes = mesh_size if isinstance(mesh_size, list) else [mesh_size] * len(points)
        if len(sizes) != len(points):
            raise ValueError("mesh_size list must match points")
        self.points = [Point(tuple(p), s) for p, s in zip(points, sizes)]
        self.curves = [Line((self.points[i], self.points[(i + 1) % len(self.points)])) for i in range(len(self.points))]
        self.lines = self.curves
        self.curve_loop = CurveLoop(self.curves)
        self.surface = PlaneSurface(self.curve_loop, holes or []) if make_surface else None
        self.mesh_size = mesh_size
        self.holes = holes or []


class Circle:
    dim = 2
    def __init__(self, x0, radius, mesh_size=None, num_sections=32, **_):
        self.x0, self.radius, self.mesh_size = tuple(_xyz([x0])[0]), float(radius), mesh_size
        if self.radius <= 0:
            raise ValueError("circle radius must be positive")
        self.num_sections = max(int(num_sections), 3)


class Geometry:
    def __init__(self, init_argv=None):
        self._surfaces = []
        self._points = []
        self._callback = None

    def __enter__(self): return self
    def __exit__(self, *_): return False

    def add_point(self, x, mesh_size=None):
        p = Point(tuple(_xyz([x])[0]), mesh_size)
        self._points.append(p)
        return p

    def add_line(self, p0, p1): return Line((p0, p1))
    def add_curve_loop(self, curves): return CurveLoop(list(curves))
    def add_plane_surface(self, curve_loop, holes=None):
        s = PlaneSurface(curve_loop, holes or [])
        self._surfaces.append(s)
        return s

    def add_polygon(self, points, mesh_size=None, holes=None, make_surface=True):
        poly = Polygon(points, mesh_size, holes, make_surface)
        self._points.extend(poly.points)
        if make_surface: self._surfaces.append(poly)
        return poly

    def add_rectangle(self, xmin, xmax, ymin, ymax, z, mesh_size=None, holes=None, make_surface=True):
        return self.add_polygon([[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]],
                                mesh_size, holes, make_surface)

    def add_circle(self, x0, radius, mesh_size=None, num_sections=32, make_surface=True, **kwargs):
        circle = Circle(x0, radius, mesh_size, num_sections, **kwargs)
        self._points.append(Point(circle.x0, mesh_size))
        if make_surface: self._surfaces.append(circle)
        return circle

    def set_mesh_size_callback(self, fun, ignore_other_mesh_sizes=True): self._callback = fun
    def set_background_mesh(self, fields, operator="Min"):
        fields = list(fields) if isinstance(fields, (list, tuple)) else [fields]
        if not fields:
            raise ValueError("at least one sizing field is required")
        chooser = min if operator.lower() == "min" else max
        self._callback = lambda dim, tag, x, y, z, lc: chooser(
            float(field.evaluate([[x, y, z]])[0]) for field in fields
        )
    def set_transfinite_curve(self, curve, num_nodes, mesh_type="Progression", coeff=1.0): return None
    def set_transfinite_surface(self, surface, arrangement="Left", corner_pts=None): return None
    def set_recombined_surfaces(self, surfaces): return None

    def generate_mesh(self, dim=3, order=None, algorithm=None, verbose=False):
        if dim != 2:
            raise NotImplementedError("mojo-pygmsh currently generates 2-D meshes")
        if len(self._surfaces) != 1:
            raise NotImplementedError("generate one 2-D surface per Geometry")
        surface = self._surfaces[0]
        if isinstance(surface, Circle): return _disk_mesh(surface, self._size(surface.mesh_size))
        vertices = surface.points if isinstance(surface, Polygon) else [line.points[0] for line in surface.curve_loop.curves]
        points = np.array([p.x for p in vertices], dtype=np.float64)
        mesh_size = surface.mesh_size if isinstance(surface, Polygon) else next(
            (point.mesh_size for point in vertices if point.mesh_size is not None), None
        )
        if len(points) == 4 and _axis_aligned_rectangle(points) and not surface.holes:
            return _rectangle_mesh(points, self._size(mesh_size))
        if surface.holes:
            raise NotImplementedError("polygon holes need a constrained triangulator")
        return _polygon_mesh(points)

    def _size(self, candidate):
        size = float(candidate) if candidate is not None else 0.1
        if self._callback is not None:
            sample = np.array([p.x for p in self._points], dtype=np.float64) if self._points else np.zeros((1, 3))
            if len(sample) > 1:
                sample = np.vstack((sample, sample.mean(axis=0)))
            values = np.asarray([self._callback_value(p, size) for p in sample], dtype=float)
            positive = values[values > 0]
            if len(positive): size = min(size, float(positive.min()))
        if size <= 0: raise ValueError("mesh size must be positive")
        return size

    def _callback_value(self, point, lc):
        try:
            return self._callback(2, 1, point[0], point[1], point[2], lc)
        except TypeError:
            return self._callback(*point)


def _rectangle_mesh(corners, size):
    xmin, xmax = corners[:, 0].min(), corners[:, 0].max()
    ymin, ymax = corners[:, 1].min(), corners[:, 1].max()
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("rectangle dimensions must be positive")
    nx, ny = max(1, math.ceil((xmax - xmin) / size)), max(1, math.ceil((ymax - ymin) / size))
    points = np.empty(((nx + 1) * (ny + 1), 3), dtype=np.float64)
    triangles = np.empty((2 * nx * ny, 3), dtype=np.int64)
    lib().mpg_rectangle_mesh(xmin, xmax, ymin, ymax, nx, ny,
                             addr(points, np.float64), addr(triangles, np.int64))
    return meshio.Mesh(points, [("triangle", triangles), ("line", _boundary(nx, ny))])


def _disk_mesh(circle, size):
    n = max(circle.num_sections, math.ceil(2 * math.pi * circle.radius / size))
    n = 4 * math.ceil(n / 4)
    angles = np.arange(n) * (2 * math.pi / n)
    rim = np.column_stack((circle.x0[0] + circle.radius * np.cos(angles), circle.x0[1] + circle.radius * np.sin(angles), np.full(n, circle.x0[2])))
    points = np.vstack((np.asarray(circle.x0), rim))
    triangles = np.column_stack((np.zeros(n, dtype=np.int64), np.arange(1, n + 1), np.roll(np.arange(1, n + 1), -1)))
    lines = np.column_stack((np.arange(1, n + 1), np.roll(np.arange(1, n + 1), -1)))
    return meshio.Mesh(points, [("triangle", triangles), ("line", lines)])


def _polygon_mesh(points):
    indices = list(range(len(points)))
    if _signed_area(points) < 0: indices.reverse()
    triangles = []
    while len(indices) > 3:
        for k, b in enumerate(indices):
            a, c = indices[k - 1], indices[(k + 1) % len(indices)]
            if _cross(points[a], points[b], points[c]) <= 0: continue
            if any(_inside(points[t], points[a], points[b], points[c]) for t in indices if t not in (a, b, c)): continue
            triangles.append((a, b, c)); indices.pop(k); break
        else: raise ValueError("polygon is self-intersecting or degenerate")
    triangles.append(tuple(indices))
    lines = np.column_stack((np.arange(len(points)), np.roll(np.arange(len(points)), -1)))
    return meshio.Mesh(points, [("triangle", np.asarray(triangles, dtype=np.int64)), ("line", lines)])


def _boundary(nx, ny):
    edges = []
    for i in range(nx): edges.append((i, i + 1)); edges.append((ny * (nx + 1) + i, ny * (nx + 1) + i + 1))
    for j in range(ny): edges.append((j * (nx + 1), (j + 1) * (nx + 1))); edges.append((j * (nx + 1) + nx, (j + 1) * (nx + 1) + nx))
    return np.asarray(edges, dtype=np.int64)


def _xyz(points):
    arr = f64(points)
    arr = np.atleast_2d(arr)
    if arr.shape[1] == 2: arr = np.column_stack((arr, np.zeros(len(arr))))
    if arr.shape[1] != 3: raise ValueError("points must be two- or three-dimensional")
    return arr


def _axis_aligned_rectangle(p): return np.unique(p[:, 0]).size == np.unique(p[:, 1]).size == 2
def _signed_area(p): return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
def _cross(a, b, c): return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
def _inside(p, a, b, c):
    signs = np.array([_cross(a, b, p), _cross(b, c, p), _cross(c, a, p)])
    return np.all(signs >= 0) or np.all(signs <= 0)
