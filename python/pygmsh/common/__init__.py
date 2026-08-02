"""Shared primitive classes for code that imports pygmsh.common explicitly."""

from ..geo import CurveLoop, Line, PlaneSurface, Point, Polygon

__all__ = ["Point", "Line", "CurveLoop", "PlaneSurface", "Polygon"]
