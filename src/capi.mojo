"""Allocation-free C ABI for structured mesh construction and sizing fields."""

from max.algorithm import parallelize
from std.math import sqrt
from std.sys import simd_width_of as simdwidthof
from std.sys.info import num_physical_cores

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime DISTANCE_PARALLEL_THRESHOLD = 262144


def distance_range(
    query: Ptr, sources: Ptr, result: Ptr, nsources: Int, begin: Int, end: Int
):
    for i in range(begin, end):
        var nearest = 1.7976931348623157e308
        var x = query[3 * i]
        var y = query[3 * i + 1]
        var z = query[3 * i + 2]
        for j in range(nsources):
            var dx = x - sources[3 * j]
            var dy = y - sources[3 * j + 1]
            var dz = z - sources[3 * j + 2]
            var squared = dx * dx + dy * dy + dz * dz
            if squared < nearest:
                nearest = squared
        result[i] = sqrt(nearest)


@export("mpg_distance_field")
def mpg_distance_field(
    query_addr: Int, nquery: Int, sources_addr: Int, nsources: Int, result_addr: Int
) abi("C"):
    if nquery <= 0 or nsources <= 0 or query_addr == 0 or sources_addr == 0 or result_addr == 0:
        return
    var query = Ptr(unsafe_from_address=query_addr)
    var sources = Ptr(unsafe_from_address=sources_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    if nquery < DISTANCE_PARALLEL_THRESHOLD:
        distance_range(query, sources, result, nsources, 0, nquery)
        return
    var workers = min(nquery, num_physical_cores())

    @parameter
    def work(worker: Int):
        var begin = worker * nquery // workers
        var end = (worker + 1) * nquery // workers
        distance_range(query, sources, result, nsources, begin, end)

    parallelize[work](workers, workers)


@export("mpg_rectangle_mesh")
def mpg_rectangle_mesh(
    xmin: Float64, xmax: Float64, ymin: Float64, ymax: Float64,
    nx: Int, ny: Int, points_addr: Int, triangles_addr: Int
) abi("C"):
    # The Python bridge validates allocation sizes.  Keep the exported ABI safe
    # for accidental direct callers too: Mojo pointers cannot represent null.
    if nx <= 0 or ny <= 0 or points_addr == 0 or triangles_addr == 0:
        return
    var points = Ptr(unsafe_from_address=points_addr)
    var triangles = IPtr(unsafe_from_address=triangles_addr)
    var dx = (xmax - xmin) / Float64(nx)
    var dy = (ymax - ymin) / Float64(ny)
    for j in range(ny + 1):
        for i in range(nx + 1):
            var p = j * (nx + 1) + i
            points[3 * p] = xmin + Float64(i) * dx
            points[3 * p + 1] = ymin + Float64(j) * dy
            points[3 * p + 2] = 0.0
    for j in range(ny):
        for i in range(nx):
            var q = j * nx + i
            var a = j * (nx + 1) + i
            var b = a + 1
            var c = a + nx + 1
            var d = c + 1
            if (i + j) % 2 == 0:
                triangles[6 * q] = Int64(a)
                triangles[6 * q + 1] = Int64(b)
                triangles[6 * q + 2] = Int64(d)
                triangles[6 * q + 3] = Int64(a)
                triangles[6 * q + 4] = Int64(d)
                triangles[6 * q + 5] = Int64(c)
            else:
                triangles[6 * q] = Int64(a)
                triangles[6 * q + 1] = Int64(b)
                triangles[6 * q + 2] = Int64(c)
                triangles[6 * q + 3] = Int64(b)
                triangles[6 * q + 4] = Int64(d)
                triangles[6 * q + 5] = Int64(c)


@export("mpg_threshold_field")
def mpg_threshold_field(
    query_addr: Int, nquery: Int, sources_addr: Int, nsources: Int,
    lcmin: Float64, lcmax: Float64, distmin: Float64, distmax: Float64, result_addr: Int
) abi("C"):
    if nquery <= 0 or nsources <= 0 or query_addr == 0 or sources_addr == 0 or result_addr == 0:
        return
    var query = Ptr(unsafe_from_address=query_addr)
    var sources = Ptr(unsafe_from_address=sources_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    for i in range(nquery):
        var nearest = 1.7976931348623157e308
        var x = query[3 * i]
        var y = query[3 * i + 1]
        var z = query[3 * i + 2]
        for j in range(nsources):
            var dx = x - sources[3 * j]
            var dy = y - sources[3 * j + 1]
            var dz = z - sources[3 * j + 2]
            var squared = dx * dx + dy * dy + dz * dz
            if squared < nearest:
                nearest = squared
        var distance = sqrt(nearest)
        if distance <= distmin:
            result[i] = lcmin
        elif distance >= distmax:
            result[i] = lcmax
        elif distmax <= distmin:
            result[i] = lcmax
        else:
            result[i] = lcmin + (lcmax - lcmin) * (distance - distmin) / (distmax - distmin)


@export("mpg_min_field")
def mpg_min_field(values_addr: Int, nfields: Int, npoints: Int, result_addr: Int) abi("C"):
    if nfields <= 0 or npoints <= 0 or values_addr == 0 or result_addr == 0:
        return
    var values = Ptr(unsafe_from_address=values_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    comptime W = simdwidthof[DType.float64]()
    var vector_end = npoints - npoints % W
    var i = 0
    while i < vector_end:
        result.store(i, values.load[width=W](i))
        i += W
    while i < npoints:
        result[i] = values[i]
        i += 1
    for j in range(1, nfields):
        i = 0
        var offset = j * npoints
        while i < vector_end:
            result.store(i, min(result.load[width=W](i), values.load[width=W](offset + i)))
            i += W
        while i < npoints:
            if values[offset + i] < result[i]:
                result[i] = values[offset + i]
            i += 1


@export("mpg_max_field")
def mpg_max_field(values_addr: Int, nfields: Int, npoints: Int, result_addr: Int) abi("C"):
    if nfields <= 0 or npoints <= 0 or values_addr == 0 or result_addr == 0:
        return
    var values = Ptr(unsafe_from_address=values_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    comptime W = simdwidthof[DType.float64]()
    var vector_end = npoints - npoints % W
    var i = 0
    while i < vector_end:
        result.store(i, values.load[width=W](i))
        i += W
    while i < npoints:
        result[i] = values[i]
        i += 1
    for j in range(1, nfields):
        i = 0
        var offset = j * npoints
        while i < vector_end:
            result.store(i, max(result.load[width=W](i), values.load[width=W](offset + i)))
            i += W
        while i < npoints:
            if values[offset + i] > result[i]:
                result[i] = values[offset + i]
            i += 1
