# CUDA Memory and Tiling

Source: `03_Memory_Perf_Optimizations.pdf`, slides 11-25.

## Storage scope and cost

| Storage | Typical scope | Main consideration |
| --- | --- | --- |
| Registers | One thread | Fast operands; finite capacity limits resident threads. |
| Local memory | One thread logically | May reside in device memory; spills and some arrays use it. |
| Shared memory | One block | On-chip scratchpad for cooperation; allocated per resident block. |
| Global memory | Device allocation | Large capacity and higher latency; bandwidth and access pattern matter. |
| Constant memory | Read-only data | Useful for suitable shared constants. |
| L1/L2 caches | Hardware managed | Can capture reuse without explicit staging. |

“Local” describes visibility, not necessarily a fast physical location. Automatic scalars are often stored in registers, but compiler allocation and spills determine actual storage.

## Tile matrix multiplication

The basic matrix kernel requests the same inputs across many output threads. Assign a `T × T` output tile to one block. In each phase, threads load a `T × T` tile of each input into shared memory, synchronize, and accumulate `T` products per output. Repeat for `ceil(N / T)` phases.

1. Load assigned input values; write zero for out-of-bounds tile elements.
2. All block threads reach `__syncthreads()` before reading the tile.
3. Each thread accumulates its partial dot product.
4. All block threads reach another barrier before the arrays are reused.
5. In-bounds threads store their outputs.

The first barrier protects reads after writes; the second protects writes after reads. Do not let boundary checks skip either barrier.

Two FP32 `T × T` tiles use `8T²` bytes of shared memory per block. Larger tiles can cut global input traffic but also increase block size and shared-memory use, leaving fewer blocks resident. The lecture's `T = 32` example raises simplified arithmetic intensity from 0.25 to about 8 FLOP/B, though actual traffic depends on caches and access patterns.

**Next:** [[Memory Access and Optimization]] addresses the remaining memory transactions and tuning choices.
