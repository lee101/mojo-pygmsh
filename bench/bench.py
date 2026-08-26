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
from pygmsh.fields import Distance, Max, Min, Threshold


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


def gpu_memory_free_mib():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        return int(result.stdout.split()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return 0


def main():
    rng = np.random.default_rng(0)
    query = np.ascontiguousarray(rng.normal(size=(100_000, 3)))
    sources = np.ascontiguousarray(rng.normal(size=(32, 3)))
    distance = Distance(sources)
    field = Threshold(distance, .02, .5, .1, 2.)
    distance_reference = lambda: np.sqrt(
        ((query[:, None] - sources[None]) ** 2).sum(axis=2).min(axis=1)
    )
    threshold_reference = lambda: np.interp(
        distance_reference(),
        [.1, 2.], [.02, .5],
    )
    grouped = [Distance(chunk) for chunk in np.split(sources, 4)]
    grouped_reference = lambda: [
        np.sqrt(((query[:, None] - chunk[None]) ** 2).sum(axis=2).min(axis=1))
        for chunk in np.split(sources, 4)
    ]
    cases = [
        ("Distance field (100k x 32)", lambda: distance.evaluate(query), distance_reference),
        ("Threshold field (100k x 32)", lambda: field.evaluate(query), threshold_reference),
        ("Min field (4 x 100k x 8)", lambda: Min(grouped).evaluate(query),
         lambda: np.minimum.reduce(grouped_reference())),
        ("Max field (4 x 100k x 8)", lambda: Max(grouped).evaluate(query),
         lambda: np.maximum.reduce(grouped_reference())),
        ("Rectangle mesh (80,000 triangles)", mesh_mojo, mesh_gmsh()),
    ]
    free_mib = gpu_memory_free_mib()
    if free_mib >= 4000:
        from pygmsh._gpu import available
        if available():
            gpu_query = np.ascontiguousarray(rng.normal(size=(500_000, 3)))
            gpu_sources = np.ascontiguousarray(rng.normal(size=(128, 3)))
            gpu_field = Distance(gpu_sources)
            cases.insert(-1, (
                "Distance GPU (500k x 128)",
                lambda: gpu_field.evaluate(gpu_query, device="gpu"),
                lambda: gpu_field.evaluate(gpu_query),
            ))
        else:
            print("GPU benchmark skipped: no usable device context")
    else:
        print(f"GPU benchmark skipped: only {free_mib} MiB free")
    print("| case | mojo-pygmsh | reference | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases:
        ours()
        a, b = best(ours), best(theirs)
        verdict = "faster" if a < b else "slower"
        print(f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {b / a:.2f}x {verdict} |")


if __name__ == "__main__":
    main()
