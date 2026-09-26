"""Optional GPU distance-field kernel with CPU fallback handled by Python."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import sqrt

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime BLOCK = 256


def distance_kernel(
    query: UnsafePointer[Float64, AnyOrigin[mut=True]],
    sources: UnsafePointer[Float64, AnyOrigin[mut=True]],
    result: UnsafePointer[Float64, AnyOrigin[mut=True]],
    nquery: Int64,
    nsources: Int64,
):
    var i = Int(global_idx.x)
    if Int64(i) >= nquery:
        return
    var nearest = 1.7976931348623157e308
    var x = query[3 * i]
    var y = query[3 * i + 1]
    var z = query[3 * i + 2]
    for j in range(Int(nsources)):
        var dx = x - sources[3 * j]
        var dy = y - sources[3 * j + 1]
        var dz = z - sources[3 * j + 2]
        var squared = dx * dx + dy * dy + dz * dz
        if squared < nearest:
            nearest = squared
    result[i] = sqrt(nearest)


@export("mpg_gpu_available")
def mpg_gpu_available() abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        _ = ctx.name()
        return 1
    except:
        return 0


@export("mpg_gpu_distance_field")
def mpg_gpu_distance_field(
    query_addr: Int, nquery: Int, sources_addr: Int, nsources: Int, result_addr: Int
) abi("C") -> Int:
    if nquery <= 0 or nsources <= 0 or query_addr == 0 or sources_addr == 0 or result_addr == 0:
        return -2
    try:
        var ctx = DeviceContext()
        var query_device = ctx.enqueue_create_buffer[DType.float64](3 * nquery)
        var sources_device = ctx.enqueue_create_buffer[DType.float64](3 * nsources)
        var result_device = ctx.enqueue_create_buffer[DType.float64](nquery)
        ctx.enqueue_copy(query_device, Ptr(unsafe_from_address=query_addr))
        ctx.enqueue_copy(sources_device, Ptr(unsafe_from_address=sources_addr))
        ctx.enqueue_function[distance_kernel](
            query_device,
            sources_device,
            result_device,
            Int64(nquery),
            Int64(nsources),
            grid_dim=(nquery + BLOCK - 1) // BLOCK,
            block_dim=BLOCK,
        )
        ctx.enqueue_copy(Ptr(unsafe_from_address=result_addr), result_device)
        ctx.synchronize()
        return 0
    except:
        return -1
