# CUDA Grids, GPU Architecture, and Scheduling

Source: [[raw_files/02_Arch_Scheduling.pdf]]

## Grid, block, and thread coordinates

All threads in a grid execute the same kernel. Each thread uses its coordinates to select its data. `blockIdx` identifies a block; `threadIdx` identifies a thread within that block. The launch configuration sets `gridDim` and `blockDim`.

```cpp
dim3 grid(32, 1, 1);
dim3 block(128, 1, 1);
kernel<<<grid, block>>>(...);
```

Grid and block dimensions may be multidimensional. For a 2D image, a common mapping is:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Choose dimensions that match the data layout when that makes indexing and memory access straightforward. Always check that coordinates are within the input bounds, especially for dimensions that are not multiples of the block dimensions.

The slides state a maximum of 1024 threads per block, with the product of its x, y, and z dimensions limited to 1024. Grid dimension limits depend on the CUDA device and compute capability; consult the current CUDA documentation for target-specific limits.

## Example workloads

- **Vector addition:** one thread computes one output element.
- **Color to grayscale:** map a thread to an image pixel, account for the input pixel representation, and guard edge threads.
- **Image blur:** each output pixel depends on a neighborhood of input pixels; edge handling must define what happens where the neighborhood crosses the image boundary.
- **Matrix multiplication:** a simple version assigns one thread to each output element and computes its dot product. This is easy to understand but can repeatedly load the same input values from global memory.

## GPU organization

The slides describe GPUs as groups of GPU Processing Clusters (GPCs), containing Streaming Multiprocessors (SMs), with CUDA cores and other specialized units. SMs provide execution resources and on-chip storage such as registers and shared memory. Exact resource counts differ by GPU generation and model.

## Blocks, warps, and scheduling

The GPU schedules blocks onto available SMs. A block executes on one SM, and its threads are partitioned into warps (32 threads in the CUDA model described in the slides). The SM schedules ready warps, allowing it to run other work while one warp waits on memory or another dependency.

**Occupancy** is the fraction of an SM’s supported warps that are active or resident. Registers, shared memory, block slots, and thread limits constrain how many blocks and warps fit on an SM. More occupancy can help hide latency, but it is not by itself a guarantee of higher performance.

## Synchronization and divergence

Threads in a block can coordinate through shared memory and block synchronization. A barrier is safe only when all participating threads in that block reach it along compatible control flow. A thread must not exit or skip a required barrier while other threads wait at it.

When threads in one warp take different branches, the warp has **control divergence**. The hardware may execute each branch path with only the threads for that path active, reducing useful parallel work. Organizing work so neighboring threads follow the same control path can reduce divergence.

