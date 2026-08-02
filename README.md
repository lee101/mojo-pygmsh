# mojo-pygmsh

`mojo-pygmsh` is a standalone, portable 2-D constructive mesher written around
Mojo kernels. It follows the familiar `pygmsh.geo.Geometry` and
`pygmsh.occ.Geometry` entry points, returning ordinary `meshio.Mesh` objects.
It is aimed at fast construction of rectangular triangle grids and evaluation
of distance-based mesh-size fields without requiring a Gmsh installation at
runtime.

## Covered subset

| API | Coverage |
| --- | --- |
| `geo.Geometry` | context manager, points/lines/curve loops/plane surfaces, `add_polygon`, `add_rectangle`, `add_circle`, `generate_mesh(dim=2)` |
| `occ.Geometry` | `add_rectangle(x0, a, b)`, `add_disk(x0, radius0)`, `generate_mesh(dim=2)` |
| sizing | upstream-compatible `set_mesh_size_callback(dim, tag, x, y, z, lc)`, `set_background_mesh` |
| field kernels | `Distance`, `Threshold`, `Min`, and `Max`, all with vectorized `evaluate(points)` |

Axis-aligned rectangles are filled by a structured, alternating-diagonal
triangle grid. Disks are triangle fans with a mesh-size-derived rim count.
Simple concave polygons use ear clipping. This is deliberately not a Gmsh
replacement: there is no 3-D meshing, CAD booleans, curved elements, polygon
holes, extrusion, or recombined/quadrilateral mesh support. Unsupported paths
raise `NotImplementedError` instead of silently producing a different mesh.

## Install and use

```bash
pixi install
pixi run build
```

```python
import numpy as np
import pygmsh
from pygmsh.fields import Distance, Threshold

near_origin = Threshold(Distance([[0.0, 0.0, 0.0]]), .05, .25, .2, 1.0)

with pygmsh.geo.Geometry() as geom:
    geom.add_rectangle(-1, 1, -1, 1, 0, mesh_size=.25)
    geom.set_background_mesh(near_origin)
    mesh = geom.generate_mesh(dim=2)

assert mesh.cells_dict["triangle"].shape[1] == 3
np.savetxt("points.txt", mesh.points)
```

The project tasks are all run through Pixi:

```bash
pixi run build
pixi run test
pixi run bench
```

## Performance

Measured with `pixi run bench` on Linux 6.8, Intel Xeon E5-2697 v4 (72 logical
CPUs), Mojo 1.0.0b3.dev2026072406, NumPy and pygmsh 7.1.17. Values are the
best of three runs.

| case | mojo-pygmsh | reference | ratio |
| --- | ---: | ---: | ---: |
| Threshold field (100k x 32) | 5.8 ms | NumPy 177.7 ms | 30.67x faster |
| Rectangle mesh (80,000 triangles) | 0.7 ms | pygmsh/Gmsh 2898.8 ms | 4078.01x faster |

The Gmsh number intentionally measures a fresh `pygmsh` process from geometry
creation through mesh extraction, because Gmsh owns a process-global model.
It therefore includes its initialization cost; it is a useful end-to-end
comparison, not a claim about an already-initialized Gmsh kernel alone.

The current kernels are memory-bound (well below 2 FLOP/byte), so no GPU path
is included: host-device transfer would make these workloads slower.

## How it works

The Python layer allocates contiguous NumPy `float64` point buffers and
`int64` connectivity buffers. `ctypes` passes their addresses as `Int` values
to one Mojo compilation unit (`src/capi.mojo`), whose exported C ABI rebuilds
typed pointers. Mojo never owns or allocates output memory. Rectangle filling,
nearest-source threshold evaluation, and pointwise field reduction therefore
run directly over the NumPy memory layout, with one FFI call per operation.

Tests compare rectangle and circle domains and areas with installed upstream
`pygmsh`/Gmsh in an isolated interpreter; field values also compare to a
NumPy reference implementation.

## License

MIT
