"""Portable 2-D constructive meshing, with pygmsh-compatible entry points."""

from . import fields, geo, occ
from .fields import Distance, Max, Min, Threshold

__all__ = ["geo", "occ", "fields", "Distance", "Threshold", "Min", "Max"]
__version__ = "0.1.0"
