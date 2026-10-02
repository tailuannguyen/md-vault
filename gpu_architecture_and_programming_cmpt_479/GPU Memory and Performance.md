---
course: CMPT 479/982
lecture: 3
aliases:
  - GPU Memory Architecture and Performance Optimizations
  - CUDA Memory Optimization
tags:
  - cmpt479
  - gpu/memory
  - gpu/roofline
  - cuda/optimization
  - study/exam-review
source: "[[raw_files/03_Memory_Perf_Optimizations.pdf]]"
---

# GPU Memory and Performance

> [!abstract] Core idea
> Performance depends on how much useful computation the GPU performs per byte transferred, how efficiently threads access memory, and whether enough independent work is ready. Use the roofline to identify a limit, then improve reuse, access patterns, or scheduling according to the actual bottleneck.

**Source:** [[raw_files/03_Memory_Perf_Optimizations.pdf|Lecture 3 slides]]. References use PDF page numbers. Worked answers, code adaptations, and checklists are synthesized review aids. Code has not been compiled on a CUDA device. Numerical hardware limits are the lecture's examples.

**Prerequisites:** [[Introduction and Parallel Programming#CUDA execution and data movement|Host/device execution]] and [[CUDA Grids, GPU Architecture, and Scheduling#Occupancy and resource limits|warps, barriers, and occupancy]].

## Review route

- Before an exam: derive [[#Roofline model]], explain the two [[#Tiled matrix multiplication|tile barriers]], compute [[#Shared memory bank conflicts]], then attempt [[#Retrieval practice]].
- Cumulative review: connect [[Introduction and Parallel Programming#Performance and Amdahl's law|application speedup]] → kernel roofline → [[CUDA Grids, GPU Architecture, and Scheduling#Occupancy and resource limits|resource limits]].
- At work: follow [[#Optimization workflow]] and use [[#Optimization decision table]] to select one change at a time.

## Roofline model

*Source: pp. 6-10.*

Let $C_{peak}$ be peak compute throughput, $BW_{peak}$ peak bandwidth, and $I$ arithmetic intensity:

$$
I=\frac{FLOPs}{bytes\ transferred},\qquad
P\le\min(C_{peak},BW_{peak}I),\qquad
I_{ridge}=\frac{C_{peak}}{BW_{peak}}.
$$

The roofline plot has intensity on the horizontal axis and compute throughput on the vertical axis. The sloped roof is bandwidth × intensity; the horizontal roof is peak compute throughput.

- If $I<I_{ridge}$, the model gives a bandwidth-limited roof.
- If $I>I_{ridge}$, the model gives a compute-limited roof.
- A measured kernel can fall far below either roof because of latency, insufficient work, divergence, instruction overhead, or other limits.

Use a consistent memory level: DRAM bandwidth requires intensity based on DRAM traffic. Source-level loads, cache traffic, and actual DRAM transactions are different quantities. The slides' simple analyses count global-access bytes and ignore cache effects; they are explanatory bounds rather than complete performance predictions.

For the lecture's H100 example, $C_{peak}\approx67$ TFLOP/s and $BW_{peak}=3.35$ TB/s, so $I_{ridge}=20$ FLOP/B. Throughput also depends on precision and operation type; do not substitute a tensor-core peak for an ordinary FP32 kernel without justification.

## Roofline examples

### Vector addition

*Source: p. 8; kernel in [[Introduction and Parallel Programming#Vector addition]].*

One FP32 output performs one addition, two 4-byte loads, and one 4-byte store:

$$
I=\frac{1}{12}\approx0.0833\text{ FLOP/B}.
$$

It has a low bandwidth roof. For $10^9$ elements, nominal device traffic is 12 GB. If the kernel takes 4 ms:

$$
BW_{effective}=\frac{12\text{ GB}}{0.004\text{ s}}=3\text{ TB/s},\qquad
utilization=\frac{3}{3.35}\approx89.6\%.
$$

The ideal bandwidth time is $12/3350\approx0.003582$ s, or 3.58 ms. Moving from 4 ms to that bound is a **10.4% time reduction** or **11.7% throughput increase**; these percentages are different. The bound excludes host transfers and allocation, so it does not describe end-to-end application time.

### Matrix multiplication: capacity is not traffic

*Source: pp. 9-10.*

For square FP32 matrices of width $W$, approximate work is $2W^3$ FLOPs, counting a multiply-add as two FLOPs.

If each input element is fetched once and each output stored once, ideal traffic is $12W^2$ bytes:

$$
I_{ideal}=\frac{2W^3}{12W^2}=\frac{W}{6}.
$$

At $W=1024$, this is approximately 170.7 FLOP/B. This is an ideal reuse estimate, not the intensity of any arbitrary GEMM implementation. Page 9 rounds the coefficient to 0.167 and describes about 167 FLOP/B; $W/6$ is the consistent value.

The [[CUDA Grids, GPU Architecture, and Scheduling#Naive matrix multiplication|naive kernel]] fetches two floats for each multiply-add step. Including its final output stores, nominal traffic is $8W^3+4W^2$ bytes:

$$
I_{naive}=\frac{2W^3}{8W^3+4W^2}\longrightarrow0.25\text{ FLOP/B}.
$$

The lecture's simplified bandwidth roof is $3.35\cdot0.25=0.8375$ TFLOP/s. The difference from $W/6$ comes from repeated input requests: the array's storage capacity does not count how often values are fetched. Caches may satisfy some requests without DRAM access.

## GPU memory spaces

*Source: pp. 11-14.*

| Space | Typical physical storage | Who can use the data? | Lifetime and typical use |
| --- | --- | --- | --- |
| Registers | On-chip register file | One thread | Thread execution; private scalar values and accumulators |
| Local memory | Device memory, with caching | One thread | Thread execution; spills and private arrays not kept in registers |
| Shared memory | On-chip scratchpad | Threads in a block | Block execution; explicit cooperation and reuse |
| Global memory | Device DRAM, with caching | Threads across blocks; host through CUDA operations | Allocation/symbol lifetime; persistent datasets |
| Constant memory | Device storage with a read-only access path/cache | Device threads read it | Device symbol lifetime; common read-only values |

“Local” specifies thread-private scope, not on-chip placement. A local array can create device-memory traffic. The compiler may keep small constant-indexed arrays in registers, but that is an optimization rather than a guarantee.

Register accesses do not consume global-memory bandwidth. Shared memory requires load/store instructions but provides fast block-local exchange. The lecture introduces distributed shared memory for cooperating blocks within a supported thread block cluster; ordinary blocks do not automatically share their scratchpads.

**Counting copies:** automatic variables have one logical copy per thread; a shared declaration has one per block; a device global or constant symbol is shared rather than replicated per thread. Resident storage and total copies created across an entire launch are different counts.

## Tiled matrix multiplication

*Source: pp. 15-22.*

Let tile width be $T$, with a $T\times T$ block computing one output tile. Each thread retains one output accumulator in registers.

For each phase:

1. Threads cooperatively load a $T\times T$ tile from each input into shared memory.
2. A block barrier ensures all input tile values have been written before consumption: a **read-after-write** dependence.
3. Each thread accumulates a length-$T$ partial dot product.
4. A second barrier ensures everyone finishes reading before any thread overwrites shared storage for the next phase: a **write-after-read** dependence.

For width $W$ divisible by $T$, there are $W/T$ phases. For arbitrary $W$, use $\lceil W/T\rceil$ phases and load zeros for invalid input coordinates.

### Boundary-safe teaching kernel

This adaptation makes phase bounds and barrier participation explicit. It multiplies square, row-major matrices into a separate output, with positive `width`, block shape `(T,T)`, and grid shape `(ceil(width/T),ceil(width/T))`.

```cpp
template<int T>
__global__ void tiledMatMul(const float* M, const float* N,
                            float* P, int width) {
    __shared__ float Ms[T][T];
    __shared__ float Ns[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    float sum = 0.0f;

    for (int ph = 0; ph < (width + T - 1) / T; ++ph) {
        int mCol = ph * T + tx;
        int nRow = ph * T + ty;
        Ms[ty][tx] = (row < width && mCol < width)
                  ? M[row * width + mCol] : 0.0f;
        Ns[ty][tx] = (nRow < width && col < width)
                  ? N[nRow * width + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < T; ++k)
            sum += Ms[ty][k] * Ns[k][tx];
        __syncthreads();
    }
    if (row < width && col < width)
        P[row * width + col] = sum;
}
```

Every block thread executes every phase and barrier, including those without a valid output. Such threads may still load input values needed by valid output threads. Avoid wrapping the whole phase loop in an output bounds check.

### Reuse and arithmetic intensity

For divisible dimensions, one phase does approximately $2T^3$ FLOPs while loading $2T^2$ floats, or $8T^2$ bytes. Ignoring the final output store:

$$
I_{tiled}\approx\frac{T}{4}.
$$

Including stores over the complete matrix:

$$
I_{tiled}=\frac{2W^3}{8W^3/T+4W^2}.
$$

For $T=32$ and large $W$, intensity approaches 8 FLOP/B; input requests fall by a factor of 32 compared with the naive model. The lecture's H100 bandwidth roof becomes $3.35\cdot8=26.8$ TFLOP/s, still below the 67 TFLOP/s compute roof. This is a model improvement, not a guaranteed 32-fold measured speedup.

Each input element is nominally requested $W$ times without tiling and $W/T$ times with $T\times T$ tiles, assuming divisibility. Boundary padding and caches change exact traffic.

## Shared memory and occupancy

*Source: pp. 23-25; general method in [[CUDA Grids, GPU Architecture, and Scheduling#Occupancy and resource limits]].*

Two FP32 tiles use $s=8T^2$ shared bytes per block. With $T^2$ threads, that is 8 bytes/thread. At $T=32$, storage is 8192 bytes/block and the block has 1024 threads. Shared-memory capacity need not be its binding constraint; threads, registers, block slots, and per-block restrictions still matter.

For the lecture's H100 example, a kernel with 256 threads/block and 38 KiB shared memory/block fits six blocks into 228 KiB, giving 1536 resident threads and 75% occupancy, assuming no other constraint. Whole-block allocation is essential: an average bytes/thread shortcut is useful only when it agrees with the integer block calculation.

### Dynamic shared memory

The third launch configuration argument gives **bytes per block**. A single dynamically allocated region can be partitioned into two arrays:

```cpp
// Inside a kernel with known tile width T:
extern __shared__ float storage[];
float* Ms = storage;
float* Ns = storage + T * T;  // Pointer offset is in float elements.
// Access as Ms[ty*T + tx] and Ns[ty*T + tx].

// Host launch fragment:
size_t sharedBytes = 2 * T * T * sizeof(float);
// dynamicKernel<<<grid, block, sharedBytes>>>(...);
```

Page 25's displayed declaration is malformed and mixes byte and element offset conventions. The fragment above keeps those units explicit. Account for static plus dynamic shared storage, and verify both per-block and per-SM limits. The correct query fields are explained in [[CUDA Grids, GPU Architecture, and Scheduling#Querying actual resources|the device-query note]], based on [NVIDIA's API reference](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaDeviceProp.html).

## Global memory coalescing

*Source: pp. 27-30.*

Coalescing combines memory requests from lanes executing the **same warp instruction** into fewer transactions when their addresses fit the hardware's transaction layout. Think across lanes at one instruction, rather than only across successive accesses by one thread.

- Adjacent FP32 elements across consecutive lanes are a favorable pattern.
- Large strides across lanes generally require more transactions and waste transferred bytes.
- Alignment and transaction boundaries matter; consecutive addresses alone do not specify an exact transaction count.
- Caching can reduce DRAM requests independently of coalescing.

For row-major `N[k*W + col]`, consecutive `threadIdx.x` values map to consecutive columns. For column-major `N[col*W + k]`, the same mapping makes adjacent lanes stride by $W$ elements.

**Corner turning:** load column-major input in the order that is contiguous in global memory, then arrange it in shared memory for the later dot products. Swapping the load roles of `threadIdx.x` and `threadIdx.y` must be accompanied by a consistent shared-memory placement. Simply changing a global address without checking the logical tile contents can produce the wrong matrix.

For $16\times16$ blocks, each warp spans two rows. Analyze both groups of 16 lanes; do not assume every warp corresponds to one entire matrix row. See [[CUDA Grids, GPU Architecture, and Scheduling#Warps and SIMT|warp linearization]].

**Distinction:** tiling reduces repeated bytes through reuse; coalescing reduces transaction waste for requested bytes. A kernel may need both.

## DRAM latency and bandwidth utilization

*Source: pp. 31-34.*

Multiple channels and banks allow memory requests to overlap. While one bank waits through an access cycle, other banks can use the channel. Interleaving consecutive address ranges across channels/banks spreads traffic.

For a simplified double-data-rate channel:

$$
BW=(bytes\ per\ transfer)\cdot2\cdot(clock\ frequency).
$$

A 64-bit channel at 3.2 GHz gives $8\cdot2\cdot3.2=51.2$ GB/s, the lecture's DDR5-6400 example. Peak bus bandwidth is not necessarily the bandwidth achieved by a kernel.

The lecture's simplified bank-overlap model needs at least $R+1$ banks when $R$ is the ratio of array access latency to transfer time. This is an explanatory model, not a universal bank-count formula for every memory controller. [[CUDA Grids, GPU Architecture, and Scheduling#Latency hiding|Ready warp scheduling]] and parallel memory requests both help tolerate latency; neither increases the physical peak bandwidth.

## Shared memory bank conflicts

*Source: pp. 36-37, 47.*

Under the lecture's 32-bank, 4-byte-word model:

$$
bank(address)=\left\lfloor\frac{byte\ address}{4}\right\rfloor\bmod32.
$$

For FP32 array `a`, assuming `a[0]` starts in bank 0, the bank of `a[i]` is $i\bmod32$. Different words requested from the same bank by lanes of one warp must be served in separate conflict-free accesses.

**Same-word read exception:** multiple lanes reading the same shared word can receive a broadcast; that is not a different-word bank conflict. This clarifies why `Ms[ty][k]` can be efficient when lanes share `ty`. See [NVIDIA's shared-memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks). Concurrent writes to the same word should not be treated as a safe reduction.

### Padding a transposed tile

With `float a[32][32]`, lanes reading or writing `a[tx][fixed_ty]` address words $32tx+fixed\_ty$. Every lane hits the same bank, causing a 32-way conflict for distinct words.

Changing the declaration to `float a[32][33]` gives word index $33tx+fixed\_ty$, whose bank is $(tx+fixed\_ty)\bmod32$. The lanes spread across all banks, at a cost of 128 additional bytes for this array.

### Stride exercise

For a full warp accessing **distinct** words `a[lane * stride]`, bank sequence is $(lane\cdot stride)\bmod32$. For nonzero integer stride, the conflict degree is $\gcd(stride,32)$ and the number of distinct banks is $32/\gcd(stride,32)$.

| Stride | Banks visited | Distinct banks | Access multiplicity per bank |
| ---: | --- | ---: | ---: |
| 32 | 0 only | 1 | 32-way |
| 31 | 0, 31, 30, …, 1 | 32 | 1: no conflict |
| 24 | 0, 24, 16, 8, repeated | 4 | 8-way |
| 16 | 0, 16, repeated | 2 | 16-way |
| 12 | 0, 12, 24, 4, 16, 28, 8, 20, repeated | 8 | 4-way |

“8-way conflict” means eight different requests compete for each used bank; state this convention instead of using an ambiguous “number of conflicts.” The formula does not apply unchanged to same-word broadcasts or other word sizes.

## Other optimization techniques

*Source: pp. 35, 38-42.*

### Vector loads and stores

`float4` represents four FP32 elements, or 16 bytes. A thread can load two vectors, perform four additions, and write one output vector. This reduces the number of load/store instructions relative to four separate scalar accesses; it does not change vector addition's useful 1 FLOP per 12 bytes.

Use correctly aligned full vector groups and separately handle any scalar tail. Page 35's figure uses `blockDim.x * blockIdx.x + threadIdx.x` as a **vector index**; convert that to scalar offset `4*i` when working with scalar pointers. Avoid assuming that any arbitrary offset into a float allocation remains aligned for `float4`.

### Thread coarsening

Give each thread multiple outputs or more work. It can reduce redundant work and overhead, but can also reduce available blocks, increase registers, or change coalescing. Connect the decision to [[CUDA Grids, GPU Architecture, and Scheduling#Waves and transparent scalability|wave utilization]] and [[CUDA Grids, GPU Architecture, and Scheduling#Occupancy and resource limits|occupancy]].

### Loop unrolling

Replicate loop bodies, for example with `#pragma unroll 4`, to reduce loop-control instructions and expose independent instructions. Extra register demand and code size can offset the benefit; handle trip counts consistently. Unrolling does not inherently reduce the number of useful arithmetic operations or bytes required.

### Double buffering

Use separate input and output scratchpads so threads can read the old state while writing the next state. After a common barrier, swap their roles. This removes an overwrite hazard that otherwise requires a read-completion barrier before writing into the same buffer.

The p. 41 example reduces two barriers per iteration to one by removing a write-after-read dependence. It still needs initial data preparation and a synchronization point before new outputs are read, and it uses more storage. Double buffering alone does not guarantee asynchronous overlap of memory transfer and computation.

The checklist on p. 42 also mentions register tiling, warp-level cooperation, and privatization. Its privatization entry is explicitly marked as covered later; treat these as follow-on topics rather than fully taught algorithms in this lecture.

## Optimization decision table

*Synthesis from pp. 27-43.*

| Observation to investigate | Candidate change | Benefit to test | Tradeoff to inspect |
| --- | --- | --- | --- |
| Repeated reads of the same input | Shared-memory or register reuse | Less traffic at the relevant memory level | Capacity, barriers, registers |
| Adjacent lanes access widely separated addresses | Remap threads, change layout, or corner turn | Fewer wasted memory transactions | Correct logical indexing |
| Shared memory serializes distinct-word requests | Pad or rearrange the shared tile | Fewer bank conflicts | Extra storage |
| Few ready warps and long dependency stalls | Tune block/resource use or expose independent work | More useful issue while other work waits | Occupancy versus reuse |
| Many lanes inactive on expensive paths | Reorganize work to group similar paths | Higher useful lane utilization | Reordering cost and locality |
| Excess scalar memory instructions | Aligned vector accesses | Fewer memory instructions | Tail handling and parallelism |
| Loop-control overhead or short dependency chains | Unroll selected loops | Lower overhead or more independent instructions | Registers and code size |
| Overwrite barriers dominate | Separate old/new buffers | Fewer false-dependence barriers | More shared storage |
| Fine-grained work repeats overhead | Modest thread coarsening | Amortized overhead | Fewer blocks, larger per-thread state |

## Optimization workflow

*Source: p. 43, with practical review steps added.*

1. **Establish correctness.** Compare with a reference using appropriate numerical tolerance. Include partial tiles and small sizes.
2. **Define the measurement.** Record GPU, data type, problem size, and whether time includes allocation and host transfers. Ensure timing covers completed device execution.
3. **Estimate the limit.** Count operations and traffic; calculate intensity and a roofline bound. State assumptions about reuse and caches.
4. **Locate the bottleneck.** Inspect achieved bandwidth, compute activity, ready work, lane utilization, and memory behavior. A theoretical occupancy number alone does not identify it.
5. **Make one targeted change.** Predict which traffic, instruction count, or stall should decrease, and what resource might increase.
6. **Recheck correctness and time.** Keep the change only if it benefits representative workloads.
7. **Repeat with the new bottleneck.** Stop when additional complexity has little useful benefit near the relevant limit.

Remember [[Introduction and Parallel Programming#Performance and Amdahl's law|Amdahl's law]]: a kernel improvement must matter at application scope. Keep a brief record of baseline, change, assumptions, measured result, and reason to retain it.

## Retrieval practice

1. A thread performs 36 FLOPs and seven 32-bit global accesses. Compute intensity and classify the roofline on (a) 200 GFLOP/s, 100 GB/s; (b) 300 GFLOP/s, 250 GB/s.
2. Explain why GEMM's ideal $W/6$ intensity does not describe the naive kernel.
3. For $T=32$, compute shared bytes/block, approximate tiled intensity, and the input-load reduction factor.
4. Why are both tile barriers necessary? Why must threads outside output bounds still participate?
5. A launch has 1000 blocks of 512 threads. How many logical private-variable copies and shared-variable copies exist across its execution?
6. Under 2048 threads/SM, 32 blocks/SM, 65536 registers/SM, and 96 KiB shared memory/SM, find occupancy for (a) 64 threads/block, 27 registers/thread, 4 KiB shared/block; (b) 256 threads/block, 31 registers/thread, 8 KiB shared/block.
7. How do tiling, coalescing, and padding solve different problems?
8. For `a[lane*24]`, which banks are accessed and what is the conflict degree?
9. Does vectorizing vector addition improve its arithmetic intensity? Does doubling occupancy guarantee double throughput?
10. In p. 46's example, eight 128-thread blocks each declare private `i` and `float x[4]`, plus shared `float y_s` and `float b_s[128]`. Count copies and shared storage. If each thread loads four input floats and one old output float, then stores one output after five multiplications and five additions, compute intensity.

> [!success]- Answer key
> 1. $I=36/(7\cdot4)=9/7\approx1.286$ FLOP/B. Ridge (a) is 2, so the bandwidth roof binds at about 128.6 GFLOP/s. Ridge (b) is 1.2, so the compute roof binds at 300 GFLOP/s. Actual performance can be lower.
> 2. $W/6$ assumes each input read once. Naive threads repeatedly request rows/columns, giving about 0.25 FLOP/B in the lecture's access-count model.
> 3. $8(32^2)=8192$ bytes; approximately 8 FLOP/B; 32-fold fewer input requests under the divisible-dimension model.
> 4. First: wait for tile producers before reading. Second: wait for readers before overwriting. Invalid-output threads may supply needed tile inputs and must reach the common barriers.
> 5. 512000 private copies and 1000 shared copies, not necessarily all resident at once. Shared arrays count as one array per block.
> 6. (a) Shared storage permits 24 blocks, giving 1536 threads and 75% occupancy. (b) Eight blocks fit, giving 2048 threads and 100% occupancy in the simplified model. Page 45 says “shared memory/SM” for the per-kernel amounts; these answers explicitly interpret those amounts as **per block**. If they instead mean total shared usage per SM, that occupancy constraint cannot be inferred the same way.
> 7. Tiling reduces repeated requests; coalescing reduces transaction waste across warp lanes; padding reduces distinct-word contention in shared banks.
> 8. Banks 0, 24, 16, and 8; eight distinct-word requests per bank, an 8-way conflict.
> 9. No: useful work and bytes remain 1 FLOP per 12 bytes. No: ready work and other bottlenecks determine whether extra occupancy helps.
> 10. 1024 copies of `i` and 1024 private arrays `x` (4096 float elements total); eight copies of `y_s` and eight shared arrays `b_s`. Shared bytes/block: $4+128\cdot4=516$. Global traffic/thread: $6\cdot4=24$ bytes; intensity $10/24\approx0.417$ FLOP/B, counting multiply-add as two operations and excluding shared traffic.

## Last-pass review

- [ ] I can distinguish data size from transfer traffic and nominal requests from measured DRAM traffic.
- [ ] I can compute a roofline ridge with consistent units and operation types.
- [ ] I can explain why local memory is not necessarily fast.
- [ ] I can derive tiled loads, phase count, zero padding, and both barriers.
- [ ] I can compute whole-block occupancy with shared memory included.
- [ ] I can inspect addresses across the actual lanes of a warp.
- [ ] I can derive banks using modulo arithmetic and distinguish broadcasts from conflicts.
- [ ] I can name the resource cost of coarsening, unrolling, and double buffering.
- [ ] I can justify an optimization with measured application and kernel results.

**Previous:** [[CUDA Grids, GPU Architecture, and Scheduling]]. **Return to foundations:** [[Introduction and Parallel Programming]].
