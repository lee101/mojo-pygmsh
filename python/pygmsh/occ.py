"""OpenCASCADE-shaped constructors backed by the portable 2-D mesher."""

from __future__ import annotations

from .geo import Circle, Geometry as _Geometry


class Geometry(_Geometry):
    def add_rectangle(self, x0, a, b, corner_radius=None, mesh_size=None, **_):
        if corner_radius not in (None, 0.0):
            raise NotImplementedError("rounded OCC rectangles are not covered")
        x, y, z = x0
        return super().add_rectangle(x, x + a, y, y + b, z, mesh_size)

    def add_disk(self, x0, radius0, radius1=None, mesh_size=None, **kwargs):
        if radius1 not in (None, radius0):
            raise NotImplementedError("elliptical OCC disks are not covered")
        return self.add_circle(x0, radius0, mesh_size=mesh_size, **kwargs)
