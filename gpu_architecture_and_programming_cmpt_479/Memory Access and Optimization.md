# Memory Access and Optimization

Source: `03_Memory_Perf_Optimizations.pdf`, slides 26-43.

## Global-memory coalescing

A warp's neighboring lanes can share memory transactions when they request nearby addresses. For row-major matrix `B`, the expression `B[k * width + col]` has the same `k` and consecutive columns for neighboring output-column threads, so the addresses are adjacent. For column-major `B`, `B[col * height + k]` spaces those requests `height` elements apart. A tiled kernel can reorder cooperative loads to suit the layout, then place values in shared memory where the computation expects them; the lecture calls this corner turning.

Tiling reduces repeated requests across outputs. Coalescing reduces transactions needed to serve the remaining requests. Sustained bandwidth also needs enough independent requests across DRAM channels and banks. Ready warps can execute while others wait. Vector loads and stores can move several adjacent values per instruction when alignment and layout permit.

## Shared-memory bank conflicts

Shared memory has banks that can serve different addresses in parallel. In the lecture's 32-bank, 4-byte-per-bank example, a warp writing `a[threadIdx.x][threadIdx.y]` into `float a[32][32]` addresses rows 32 floats apart when `threadIdx.y` is fixed. Distinct addresses then hit the same bank. Padding to `float a[32][33]` shifts rows onto different banks. Accesses to the same address can have broadcast behavior, so the distinct-address case is the relevant conflict example.

## Optimize the measured bottleneck

1. Establish correctness and measure representative inputs on the target GPU.
2. Estimate FLOPs and global-memory bytes to find the likely roofline ceiling.
3. Inspect redundant loads, adjacent-lane addresses, and shared-memory bank use.
4. Check register and shared-memory use against block and warp residency.
5. Make one meaningful change and re-measure; the bottleneck may move.

Other techniques in the lecture include **thread coarsening** (more work per thread), **loop unrolling** (fewer loop branches and more independent instructions), and **double buffering** (separate storage for current and next tiles). Each may improve throughput, but each can also increase register or shared-memory demand or leave too little parallel work.
