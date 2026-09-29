# GPU Execution and Occupancy

Source: `02_Arch_Scheduling.pdf`, slides 22-38; shared-memory occupancy examples in `03_Memory_Perf_Optimizations.pdf`, slides 23-25.

## From blocks to warps

Hardware assigns each block to one streaming multiprocessor (SM). Several blocks may reside on an SM when resources allow; ordinary blocks can run in any order. An SM has execution units and on-chip registers and shared memory. The lecture shows SMs grouped in GPU Processing Clusters and gives model-specific counts that vary by device.

A resident block is partitioned into 32-thread warps in the CUDA model used here. A 256-thread block has eight warps. For multidimensional blocks, threads are linearized with x fastest, then y, then z:

`linearThread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)`.

Thus a `32 × 8` block can put one row of 32 adjacent x coordinates in each warp, while a `16 × 16` block can put two rows in one warp. Threads are written as scalar programs, but a warp's active lanes execute instructions together: single instruction, multiple threads (SIMT).

## Scheduling and occupancy

If a warp waits for memory or another dependency, the SM can issue work from a different ready warp. A grid larger than simultaneous block capacity executes in waves. Occupancy is resident warps per SM divided by the maximum supported resident warps per SM. Threads, blocks, registers, and shared memory all constrain residency.

For the lecture's illustrative H100 SM with 2048 thread slots and 32 block slots, 32-thread blocks can fill at most 1024 thread slots: 50% occupancy before other limits. Two 768-thread blocks fill 1536 slots: 75%. More occupancy can hide latency, but it does not guarantee better throughput. Extra registers or shared memory can reduce residency while improving useful work per thread.

The resource tradeoff is especially direct when a kernel stages data in [[CUDA Memory and Tiling|shared-memory tiles]]. Query device properties and profile the target GPU before choosing a block size.

**Next:** [[Synchronization and Divergence]] explains waiting at barriers and inactive warp lanes.
