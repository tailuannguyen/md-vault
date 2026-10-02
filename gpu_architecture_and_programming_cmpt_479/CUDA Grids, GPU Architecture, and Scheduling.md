---
course: CMPT 479/982
lecture: 2
aliases:
  - GPU Architecture and Thread Scheduling
  - CUDA Grids and Warps
tags:
  - cmpt479
  - cuda/thread-mapping
  - gpu/scheduling
  - gpu/occupancy
  - study/exam-review
source: "[[raw_files/02_Arch_Scheduling.pdf]]"
---

# CUDA Grids, GPU Architecture, and Scheduling

> [!abstract] Core idea
> Thread coordinates define which data a thread owns. Blocks define cooperation and resource allocation. Warps define scheduling and instruction execution. Resource limits determine how much work can reside on an SM, while ready warps allow the GPU to hide latency.

**Source:** [[raw_files/02_Arch_Scheduling.pdf|Lecture 2 slides]]. References use PDF page numbers. Explanations and worked answers are synthesized review material. Code fragments are illustrative and have not been compiled on a CUDA device. Hardware values below are the lecture's examples, not specifications for every GPU.

**Prerequisite:** [[Introduction and Parallel Programming#Vector addition|CUDA indexing and vector addition]]. **Next:** [[GPU Memory and Performance|Memory reuse and optimization]].

## Review route

- Early exam review: derive [[#Multidimensional mapping]], trace [[#Example kernels]], and explain [[#Synchronization and block independence]].
- Cumulative review: compute [[#Warps and SIMT]], [[#Control divergence]], and [[#Occupancy and resource limits]], then combine occupancy with [[GPU Memory and Performance#Shared memory and occupancy|shared-memory requirements]].
- Work reference: use [[#Practical review checklist]] to inspect a launch or diagnose insufficient parallel work.

## Multidimensional mapping

*Source: pp. 5-13, 21; overlaps Lecture 1 pp. 69-78.*

`dim3(x,y,z)` specifies dimensions in **x, y, z** order. For a 2D image use $x$ for columns, $y$ for rows, and $z=1$.

```cpp
// Host launch geometry for an image of width columns and height rows.
dim3 block(16, 16, 1);
dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
// imageKernel<<<grid, block>>>(...);

// Inside the kernel:
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
// Use row < height && col < width before accessing an output pixel.
```

The grid does not store the image; it organizes threads that compute image coordinates. The memory layout is a separate decision.

### Linear memory addresses

For a matrix with height $H$, width $W$, and zero-based indices:

$$
\text{row-major index}=row\cdot W+col,\qquad
\text{column-major index}=col\cdot H+row.
$$

For a tensor stored with $x$ varying fastest, then $y$, then $z$:

$$
index=zHW+yW+x.
$$

**Worked examples (p. 21):** $W=400$, $H=500$, row 20, column 10 gives row-major index 8010 and column-major index 5020. With depth 300, $(x,y,z)=(10,20,5)$ gives $5(500)(400)+20(400)+10=1008010$. These are element indices; multiply by the element size for byte offsets.

### Image boundary example

For 62 rows and 76 columns, $16\times16$ blocks require grid `(5,4,1)`:

- 20 blocks, each with 256 threads: 5120 launched threads.
- 4712 valid pixels: 408 threads fail the output bounds check.
- The bottom-right block has 14 valid rows and 12 valid columns: 168 useful threads.

These extra threads are intentional. Avoiding out-of-bounds accesses matters more than filling every lane at the boundary.

### Launch limits

Page 6 gives a maximum of 1024 threads per block, with the **product** $B_xB_yB_z$ constrained. `(16,32,2)` has 1024 threads; `(32,32,2)` has 2048 and is invalid under that limit. Per-axis limits also apply; a legal product alone is not sufficient.

The lecture gives grid limits $G_x\le2^{31}-1$ and $G_y,G_z\le65535$. Query the actual device for production use. A block's shape does not determine how many CUDA cores it permanently owns.

## Example kernels

*Source: pp. 10-20.*

### Color to grayscale

One thread reads one RGB pixel and writes one grayscale pixel. For row-major storage, `pixel = row * width + col`, and RGB components occupy `3*pixel`, `3*pixel+1`, and `3*pixel+2`. Guard both row and column. There is no communication between output threads.

### Image blur

A radius $r$ uses a nominal $(2r+1)\times(2r+1)$ neighborhood. Each thread owns one output but reads overlapping neighborhoods from the input.

For the lecture's truncated-neighborhood average:

1. Verify that the output pixel is valid.
2. Visit neighbor offsets from $-r$ through $r$.
3. Accumulate only neighbors within the image and count them.
4. Divide by the number of valid neighbors, then write the output.

For a $3\times3$ neighborhood on an image at least $3\times3$, the interior uses 9 pixels, a non-corner edge uses 6, and a corner uses 4. Dividing every output by 9 would darken the edges under this boundary convention.

Use separate input and output storage: writing into the input could change what other threads read. Overlapping **reads** are safe when the input is immutable; overlapping writes or read/write dependencies require more care. Neighborhood reuse motivates [[GPU Memory and Performance#Tiled matrix multiplication|shared-memory tiling]], although blur tiles also need halo pixels beyond the output tile.

### Naive matrix multiplication

For square matrices of width $W$:

$$
P[row,col]=\sum_{k=0}^{W-1} M[row,k]N[k,col].
$$

One thread computes one output element, retaining a private accumulator. A row-major implementation reads `M[row*W+k]` and `N[k*W+col]` and writes `P[row*W+col]` after the loop.

```cpp
// Kernel-body fragment; M, N, P are distinct row-major device matrices.
if (row < width && col < width) {
    float sum = 0.0f;
    for (int k = 0; k < width; ++k)
        sum += M[row * width + k] * N[k * width + col];
    P[row * width + col] = sum;
}
```

A $4\times4$ output with $2\times2$ blocks uses four blocks and 16 threads. Each thread executes four multiply-add steps. No block barrier is needed because each output uses only immutable inputs and private state.

**Alternative mapping:** one thread per output row needs only $W$ threads, each doing $W$ dot products. It reduces the number of threads but increases serial work per thread and changes the warp's memory access pattern. One thread per output column is another possible design. The exercise on p. 21 repeats “row” twice; treat it as a duplicated prompt rather than assuming a hidden second design.

**Connection:** algorithmic independence makes this kernel correct, but repeated input reads make it inefficient. See [[GPU Memory and Performance#Roofline examples|naive GEMM intensity]] and [[GPU Memory and Performance#Tiled matrix multiplication|tiled GEMM]].

## GPU hardware hierarchy

*Source: pp. 22-27, 33.*

The lecture's hierarchy is GPU → GPU Processing Clusters (GPCs) → Streaming Multiprocessors (SMs) → execution units, including Streaming Processors (SPs), also called CUDA cores. Other units handle tensor operations, ray tracing, loads/stores, texture work, and special functions.

**A CUDA thread is a software execution context, not a CUDA core.** Many threads can reside on an SM while its execution units issue work from selected warps over time.

| Lecture architecture example | SM organization and examples |
| --- | --- |
| Ada AD102, pp. 24 | 12 GPCs × 12 SMs = 144 SMs; 128 SPs per SM |
| Blackwell GB202, pp. 25-26 | 12 GPCs × 8 TPCs × 2 SMs = 192 SMs; 128 SPs per SM |
| H100 example, p. 33 | 128 SPs per SM split into four processing blocks |

These are architecture-level lecture illustrations, not promises about the enabled units of a particular retail device. Use them to understand hierarchy; use device queries for the machine running your code.

## Synchronization and block independence

*Source: pp. 27-30.*

A block is assigned as a unit to one SM, and multiple blocks may reside on that SM. Threads within a block can cooperate through shared memory and a block barrier:

```cpp
__syncthreads();
```

A barrier is required when later work consumes values produced by other participating threads. Follow the lecture's safe rule: all threads in the block reach the **same barrier in the same phase**. Placing one barrier in the even-thread branch and a different one in the odd-thread branch is not equivalent to a common barrier after both branches.

```cpp
// Safe structure for an edge tile: every thread participates.
// tile is shared; each thread writes its own slot.
tile[threadIdx.x] = valid ? input[index] : 0.0f;
__syncthreads();
// Now consume shared values, with output access guarded as needed.
```

Do not let boundary threads return before a barrier that other block threads need. [[GPU Memory and Performance#Tiled matrix multiplication|Boundary-safe tiled GEMM]] uses zero-filled loads and unconditional barriers.

Ordinary blocks must support execution in arbitrary order. `__syncthreads()` does not synchronize a grid. A scheme in which one block spins waiting for another can prevent progress if the needed block has not been scheduled. The lecture introduces cluster and grid synchronization through Cooperative Groups; these require the appropriate execution model and launch support, rather than replacing a block barrier casually.

## Waves and transparent scalability

*Source: p. 30.*

If a kernel grid has $G$ blocks and the GPU can concurrently accommodate $C$ blocks for that kernel, the simplified wave count is:

$$
waves=\left\lceil\frac{G}{C}\right\rceil.
$$

For 660 blocks and capacity 264, there are three waves with 264, 264, and 132 blocks. The final wave fills half the capacity. Actual blocks finish at different times, so waves are a capacity model, not necessarily synchronized batches.

More blocks provide opportunities for load balancing. A partial final wave can limit utilization, but changing problem size just to make an integer number of waves is not always possible or useful.

**Transparent scalability:** independent blocks let the same kernel execute on GPUs with different SM counts; scheduling adjusts how many blocks run concurrently.

## Warps and SIMT

*Source: pp. 31-33, 36.*

A warp contains 32 thread lanes in the lecture's model. A block with $T$ threads has $\lceil T/32\rceil$ warps; the final warp can contain inactive lanes. Warps are formed within each block, so compute grid warp count as **blocks × warps per block**, rather than combining leftover threads across blocks.

For multidimensional blocks, linearize with $x$ changing fastest:

$$
t=t_x+B_x(t_y+B_y t_z),\qquad
warpID=\lfloor t/32\rfloor,\qquad laneID=t\bmod32.
$$

**Example:** a $16\times16$ block contains eight warps. Warp 0 spans rows `threadIdx.y=0` and 1, with 16 lanes in each row. A $32\times8$ block also contains eight warps, but each warp spans one full row. Equal thread counts can therefore produce different memory access behavior.

**SIMT** lets programmers write scalar thread code; hardware groups those threads into warps and shares instruction issue. Each thread retains its own state and can take its own logical branch path.

### Latency hiding

When a warp waits for an instruction result, the scheduler can select another **ready resident warp**. The lecture calls this zero-overhead scheduling because resident state remains in hardware; it does not require an operating-system save/restore on each switch.

Having more resident warps provides more opportunities to issue useful work while others wait. It does not reduce the underlying latency of a memory request. A large grid alone cannot guarantee many resident warps if registers or shared memory restrict each SM.

## Control divergence

*Source: pp. 34-35, 39.*

Divergence occurs when threads **within a warp** follow different paths. The hardware executes path instructions with subsets of lanes active, lowering useful lane utilization. Warps taking different uniform paths do not create this kind of within-warp divergence.

- A branch on `threadIdx.x % 2` splits every full 1D warp into even and odd lanes.
- A branch on `threadIdx.x < 32` in a 64-thread block separates two whole warps; neither warp is internally split by that condition.
- Different loop trip counts can cause divergence after some lanes finish.
- A vector bounds check can split the final warp; this is necessary for correctness.

For one instruction, useful lane efficiency can be estimated as $active\ lanes/32$. This is distinct from [[#Occupancy and resource limits|occupancy]], which counts resident warps.

Since Volta, independent thread scheduling permits more flexible progress among threads. It does not eliminate the cost of executing divergent paths. The lecture stresses explicit warp synchronization (`__syncwarp()`) when cooperating lanes must complete a phase before another begins; do not rely on implicit lockstep for communication correctness.

**Boundary example:** vector length 1003 with 64 threads per block produces 32 warps. Only the final warp mixes valid and invalid output threads: 11 valid lanes and 21 excluded lanes.

## Occupancy and resource limits

*Source: pp. 37-40. Shared-memory extension: [[GPU Memory and Performance#Shared memory and occupancy]].*

$$
occupancy=\frac{resident\ warps\ per\ SM}{maximum\ supported\ warps\ per\ SM}.
$$

| Lecture example | Threads/SM | Warps/SM | Blocks/SM | Registers/SM |
| --- | ---: | ---: | ---: | ---: |
| H100 | 2048 | 64 | 32 | 65536 |
| RTX2000 | 1536 | 48 | 24 | 65536 |

For hand calculations with $T$ threads/block, $r$ registers/thread, and $s$ shared bytes/block:

$$
B_{resident}=\min\left(
B_{max},
\left\lfloor\frac{T_{max}}{T}\right\rfloor,
\left\lfloor\frac{W_{max}}{\lceil T/32\rceil}\right\rfloor,
\left\lfloor\frac{R_{SM}}{rT}\right\rfloor,
\left\lfloor\frac{S_{SM}}{s}\right\rfloor
\right).
$$

Omit a resource term if the kernel does not use it. Then $W_{resident}=B_{resident}\lceil T/32\rceil$. Also verify that a single block is legal. Real allocations have granularity and architecture-specific restrictions; this is the simplified lecture calculation.

### Worked occupancy examples

Using the H100 lecture limits and assuming no other limit:

| Threads/block | Registers/thread | Resident blocks | Occupancy | Binding constraint |
| ---: | ---: | ---: | ---: | --- |
| 32 | 0 for this calculation | 32 | 50% | Block slots: 32 blocks contain only 1024 threads |
| 768 | 0 for this calculation | 2 | 75% | Thread slots plus whole-block allocation |
| 512 | 32 | 4 | 100% | Threads and registers exactly fill capacity |
| 512 | 33 | 3 | 75% | $\lfloor65536/(512\cdot33)\rfloor=3$ |
| 256 | 64 | 4 | 50% | Register capacity |

“0 for this calculation” means ignoring register use to isolate another limit; actual kernels use registers. Small increases in resource demand can drop occupancy in whole-block steps.

> [!tip] Occupancy is an opportunity, not a performance score
> Resident warps can hide latency, but performance also depends on ready instructions, useful lanes, traffic, and reuse. An optimization that uses more registers may still run faster even if occupancy falls. Measure the result.

### Querying actual resources

Page 38 introduces `cudaGetDeviceCount` and `cudaGetDeviceProperties`. Useful fields include `warpSize`, `maxThreadsPerBlock`, `maxThreadsDim`, `maxGridSize`, `multiProcessorCount`, and `maxThreadsPerMultiProcessor`.

**API correction:** use `regsPerMultiprocessor` and `sharedMemPerMultiprocessor` for per-SM capacities. `regsPerBlock` and `sharedMemPerBlock` refer to per-block limits; the per-SM labels in Lecture 2 p. 38 and Lecture 3 p. 24 mix these up. See the [NVIDIA device-property reference](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaDeviceProp.html).

## Retrieval practice

1. Flatten `(row=20,col=10)` in a $500\times400$ matrix in both layouts. Flatten `(x=10,y=20,z=5)` in a width-400, height-500 tensor.
2. How many launched and useful threads are in the $62\times76$ image example? How many warps does each $16\times16$ block contain?
3. Why do blur boundary pixels need a neighbor count? Why should input and output be distinct?
4. Why is a barrier under `if (threadIdx.x % 2 == 0)` unsafe?
5. For 128-thread blocks, analyze the predicate `threadIdx.x < 40 || threadIdx.x >= 104`. Which warps execute its body, and which are divergent? (Adapted from p. 39.)
6. Under the same launch, what happens for an `i % 2 == 0` branch and a loop with trip count `5 - (i % 3)`?
7. With 2048 threads/SM, 32 blocks/SM, and 65536 registers/SM, determine occupancy for `(T,r)=(128,30)`, `(32,29)`, and `(256,34)`.
8. A barrier's eight threads arrive at 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9 microseconds. What fraction of aggregate thread time is waiting?

> [!success]- Answer key
> 1. 8010 row-major; 5020 column-major; 1008010 tensor index.
> 2. 5120 launched, 4712 useful; eight warps per block.
> 3. The truncated neighborhood contains fewer valid pixels. In-place writes could alter other threads' input reads.
> 4. Only a subset reaches that barrier. Put the shared barrier outside the divergent condition.
> 5. Warp 0 has 32 active lanes; warp 1 has 8; warp 2 has none; warp 3 has 24. Warps 1 and 3 diverge. For the p. 39 grid of eight blocks, 24 warps execute the body and 16 are divergent. Per-instruction efficiencies are 100%, 25%, and 75% for warps 0, 1, and 3.
> 6. Every full warp has 16 even lanes, so all 32 grid warps execute the branch body and diverge, at 50% lane efficiency for that statement. The loop has three iterations where every lane participates, then two where only subsets remain.
> 7. Respectively 16 blocks and 100%; 32 blocks and 50% (block slots); seven blocks and 87.5% (registers: eight would require 69632).
> 8. The last arrival is at 3.0. Total available thread time is $8\cdot3=24$; useful time is 19.9; waiting is 4.1. Waiting fraction $4.1/24\approx17.1\%$.

## Practical review checklist

- [ ] Derive coordinates and addresses separately; check width/height and element/byte units.
- [ ] Linearize the block to identify actual warp membership.
- [ ] Check bounds for outputs and each input neighborhood.
- [ ] Keep block barriers common to all participating threads and phases.
- [ ] Avoid assumptions about execution order between ordinary blocks.
- [ ] Compute resident blocks from every relevant resource limit.
- [ ] Distinguish occupancy, active-lane efficiency, and achieved throughput.
- [ ] Check whether the grid provides enough blocks and whether the last wave is small.
- [ ] Follow [[GPU Memory and Performance#Optimization workflow|the memory optimization workflow]] after establishing correctness.

**Previous:** [[Introduction and Parallel Programming]]. **Next:** [[GPU Memory and Performance]].
