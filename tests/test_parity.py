from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import pygmsh
from pygmsh.fields import Distance, Max, Min, Threshold
from pygmsh._lib import f64


def _upstream(shape: str):
    code = """import json, numpy as np, pygmsh
with pygmsh.geo.Geometry() as g:
    SHAPE
    m = g.generate_mesh(dim=2)
t = m.cells_dict['triangle']
p = m.points
area = abs(np.cross(p[t[:,1]] - p[t[:,0]], p[t[:,2]] - p[t[:,0]])[:,2]).sum() / 2
print(json.dumps({'extent': [p[:,0].min(), p[:,0].max(), p[:,1].min(), p[:,1].max()], 'area': float(area)}))
""".replace("SHAPE", shape)
    return json.loads(subprocess.check_output([sys.executable, "-I", "-c", code], text=True))


def _area(mesh):
    tri = mesh.cells_dict["triangle"]
    p = mesh.points
    return abs(np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])[:, 2]).sum() / 2


def test_geo_rectangle_parity_with_upstream():
    with pygmsh.geo.Geometry() as g:
        rect = g.add_rectangle(-2, 3, 1, 4, 0, mesh_size=0.37)
        ours = g.generate_mesh(dim=2)
    ref = _upstream("g.add_rectangle(-2, 3, 1, 4, 0, mesh_size=0.37)")
    assert ours.points[:, :2].min(axis=0) == pytest.approx(ref["extent"][::2])
    assert ours.points[:, :2].max(axis=0) == pytest.approx(ref["extent"][1::2])
    assert _area(ours) == pytest.approx(ref["area"])
    assert rect.surface.dim == 2 and len(rect.lines) == 4


def test_geo_primitives_construct_a_plane_surface():
    with pygmsh.geo.Geometry() as g:
        points = [g.add_point(p, mesh_size=.5) for p in ((0, 0, 0), (1, 0, 0), (0, 1, 0))]
        loop = g.add_curve_loop([g.add_line(points[i], points[(i + 1) % 3]) for i in range(3)])
        g.add_plane_surface(loop)
        mesh = g.generate_mesh(dim=2)
    assert mesh.cells_dict["triangle"].shape == (1, 3)


def test_geo_circle_parity_with_upstream_domain():
    with pygmsh.geo.Geometry() as g:
        g.add_circle([1, -2, 0], 2.0, mesh_size=0.2, num_sections=48)
        ours = g.generate_mesh(dim=2)
    ref = _upstream("g.add_circle([1, -2, 0], 2.0, mesh_size=0.2, num_sections=48)")
    assert ours.points[:, :2].min(axis=0) == pytest.approx(ref["extent"][::2])
    assert ours.points[:, :2].max(axis=0) == pytest.approx(ref["extent"][1::2])
    assert _area(ours) == pytest.approx(ref["area"], rel=4e-3)


def test_concave_polygon_is_triangulated_without_leaving_domain():
    points = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]
    with pygmsh.geo.Geometry() as g:
        g.add_polygon(points)
        mesh = g.generate_mesh(dim=2)
    assert mesh.cells_dict["triangle"].shape == (4, 3)
    assert _area(mesh) == pytest.approx(5.0)


def test_threshold_distance_matches_numpy_reference():
    rng = np.random.default_rng(4)
    query, sources = rng.normal(size=(1003, 3)), rng.normal(size=(19, 3))
    field = Threshold(Distance(sources), lcmin=0.05, lcmax=0.7, distmin=0.2, distmax=1.4)
    got = field.evaluate(query)
    distance = np.sqrt(((query[:, None] - sources[None]) ** 2).sum(axis=2).min(axis=1))
    expected = np.interp(distance, [0.2, 1.4], [0.05, 0.7])
    assert got == pytest.approx(expected)


def test_distance_evaluate_matches_numpy_reference():
    query = np.array([[0., 0., 0.], [3., 4., 0.]])
    field = Distance([[0., 0., 0.], [10., 0., 0.]])
    assert field.evaluate(query) == pytest.approx([0., 5.])


@pytest.mark.parametrize("nquery", [1003, 262145])
def test_distance_serial_tail_and_parallel_threshold(nquery):
    rng = np.random.default_rng(nquery)
    query = rng.normal(size=(nquery, 3))
    sources = rng.normal(size=(5, 3))
    got = Distance(sources).evaluate(query)
    expected = np.sqrt(((query[:, None] - sources[None]) ** 2).sum(axis=2).min(axis=1))
    assert got == pytest.approx(expected)


def test_distance_gpu_path_or_cpu_fallback(monkeypatch):
    rng = np.random.default_rng(12)
    query = rng.normal(size=(1003, 3))
    field = Distance(rng.normal(size=(19, 3)))
    expected = field.evaluate(query)
    from pygmsh import _gpu
    if _gpu.available():
        assert field.evaluate(query, device="gpu") == pytest.approx(expected)
    monkeypatch.setattr(_gpu, "distance", lambda *_: None)
    assert field.evaluate(query, device="gpu") == pytest.approx(expected)


def test_composed_fields_use_pointwise_min_and_max():
    p = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    left = Threshold(Distance([[0, 0, 0]]), .1, .8, 0., 3.)
    right = Threshold(Distance([[3, 0, 0]]), .2, .6, 0., 3.)
    assert Min([left, right]).evaluate(p) == pytest.approx(np.minimum(left.evaluate(p), right.evaluate(p)))
    assert Max([left, right]).evaluate(p) == pytest.approx(np.maximum(left.evaluate(p), right.evaluate(p)))


def test_composed_fields_simd_body_and_scalar_tail():
    class Values:
        def __init__(self, values): self.values = values
        def evaluate(self, _): return self.values

    rng = np.random.default_rng(8)
    points = rng.normal(size=(1003, 3))
    values = rng.normal(size=(4, len(points)))
    fields = [Values(row) for row in values]
    assert Min(fields).evaluate(points) == pytest.approx(values.min(axis=0))
    assert Max(fields).evaluate(points) == pytest.approx(values.max(axis=0))


def test_gmsh_callback_signature_controls_mesh_size():
    with pygmsh.geo.Geometry() as g:
        g.add_rectangle(0, 1, 0, 1, 0, mesh_size=1.0)
        g.set_mesh_size_callback(lambda dim, tag, x, y, z, lc: .25)
        mesh = g.generate_mesh(dim=2)
    assert mesh.cells_dict["triangle"].shape == (32, 3)


def test_background_field_controls_mesh_size():
    with pygmsh.geo.Geometry() as g:
        g.add_rectangle(-1, 1, -1, 1, 0, mesh_size=1.0)
        g.set_background_mesh(Threshold(Distance([[0, 0, 0]]), .2, 1., 0., 2.))
        mesh = g.generate_mesh(dim=2)
    assert mesh.cells_dict["triangle"].shape == (200, 3)


def test_occ_rectangle_and_disk_have_upstream_signatures():
    with pygmsh.occ.Geometry() as g:
        g.add_rectangle((0, 0, 0), 2, 1, mesh_size=.25)
        rectangle = g.generate_mesh(dim=2)
    assert _area(rectangle) == pytest.approx(2.0)
    with pygmsh.occ.Geometry() as g:
        g.add_disk((0, 0, 0), 1, mesh_size=.1)
        disk = g.generate_mesh(dim=2)
    assert _area(disk) == pytest.approx(np.pi, rel=.01)


def test_explicit_unsupported_operations_fail_loudly():
    with pygmsh.geo.Geometry() as g:
        g.add_rectangle(0, 1, 0, 1, 0, holes=[object()])
        with pytest.raises(NotImplementedError):
            g.generate_mesh(dim=2)


def test_empty_fields_and_lossy_coordinate_dtypes_fail_loudly():
    with pytest.raises(ValueError, match="at least one"):
        Min([]).evaluate([[0, 0, 0]])
    with pytest.raises(ValueError, match="at least one"):
        Distance([])
    with pytest.raises(TypeError, match="wider"):
        f64(np.array([1], dtype=np.longdouble))
    with pytest.raises(TypeError, match="exactly"):
        f64(np.array([2**53 + 1], dtype=np.int64))
