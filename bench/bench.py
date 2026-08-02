"""Field and constructive-mesh benchmarks against NumPy and pygmsh/gmsh."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import pygmsh
from pygmsh.fields import Distance, Threshold


def best(fn, repeats=3):
    elapsed = math.inf
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        elapsed = min(elapsed, time.perf_counter() - started)
    return elapsed


def mesh_mojo():
    with pygmsh.geo.Geometry() as g:
        g.add_rectangle(0, 100, 0, 100, 0, mesh_size=.5)
        return g.generate_mesh(dim=2)


def mesh_gmsh():
    script = """import pygmsh
with pygmsh.geo.Geometry() as g:
    g.add_rectangle(0, 100, 0, 100, 0, mesh_size=.5)
    g.generate_mesh(dim=2)
"""
    return lambda: subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True)


def main():
    rng = np.random.default_rng(0)
    query = np.ascontiguousarray(rng.normal(size=(100_000, 3)))
    sources = np.ascontiguousarray(rng.normal(size=(32, 3)))
    field = Threshold(Distance(sources), .02, .5, .1, 2.)
    reference = lambda: np.interp(
        np.sqrt(((query[:, None] - sources[None]) ** 2).sum(axis=2).min(axis=1)),
        [.1, 2.], [.02, .5],
    )
    cases = [
        ("Threshold field (100k x 32)", lambda: field.evaluate(query), reference),
        ("Rectangle mesh (80,000 triangles)", mesh_mojo, mesh_gmsh()),
    ]
    print("| case | mojo-pygmsh | reference | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases:
        ours()
        a, b = best(ours), best(theirs)
        verdict = "faster" if a < b else "slower"
        print(f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {b / a:.2f}x {verdict} |")


if __name__ == "__main__":
    main()
