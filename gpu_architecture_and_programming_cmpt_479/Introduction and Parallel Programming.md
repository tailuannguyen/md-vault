---
course: CMPT 479/982
lecture: 1
aliases:
  - GPU Introduction
  - Parallel Programming Foundations
tags:
  - cmpt479
  - gpu/foundations
  - cuda/programming
  - study/exam-review
source: "[[raw_files/01_Introduction.pdf]]"
---

# Introduction and Parallel Programming

> [!abstract] Core idea
> A GPU accelerates work that can be divided into many independent tasks. The CPU manages execution and data movement; CUDA maps the parallel work to a grid of threads. A faster kernel helps only to the extent that it improves the whole application's execution time.

**Source:** [[raw_files/01_Introduction.pdf|Lecture 1 slides]]. Page references below use PDF page numbers. These are synthesized study notes; worked solutions and practical checklists are review aids derived from the lecture, rather than instructor-provided answers. Code is illustrative and has not been compiled on a CUDA device.

**Knowledge links:** [[CUDA Grids, GPU Architecture, and Scheduling|Thread mapping and scheduling]] explains how the grid executes. [[GPU Memory and Performance|Memory and performance]] explains why many threads alone do not guarantee speed.

## Review route

- First pass: [[#Parallel computing foundations]] → [[#Performance and Amdahl's law]] → [[#CUDA execution and data movement]] → [[#Vector addition]].
- Before an exam: derive the formulas and kernel from memory, then attempt [[#Retrieval practice]] before expanding the answers.
- Cumulative review: connect Amdahl's application limit to [[GPU Memory and Performance#Roofline model|the kernel's roofline limit]].
- At work: start with [[#Practical review checklist]], then follow the scheduling and memory links.

> [!info] Assessment context from the uploaded slides
> Page 8 describes three in-class exams, rather than a separate midterm and final. The notes still support both early and cumulative exam review. They do not establish the scope of any particular exam; use the instructor's announcements for that.

## Parallel computing foundations

*Source: pp. 14-27, 43-48.*

### Instructions, data, and memory

Flynn's taxonomy counts **instruction streams** and **data streams**:

| Model | Instruction streams | Data streams | Interpretation |
| --- | --- | --- | --- |
| SISD | One | One | Sequential execution |
| SIMD | One | Multiple | The same instruction operates on multiple data elements |
| MISD | Multiple | One | Several instruction streams process one data stream; uncommon |
| MIMD | Multiple | Multiple | Independent processors can execute different instructions on different data |

The lecture places modern parallel machines, including GPUs as a whole, under MIMD. Within a GPU, warp execution uses SIMD hardware through the SIMT programming model. These describe different levels of the machine; see [[CUDA Grids, GPU Architecture, and Scheduling#Warps and SIMT|Warps and SIMT]].

| Memory architecture | Addressing and communication                                  | Locality implication                                |
| ------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| Shared memory       | Processors can access a common address space                  | Shared addressing still requires synchronization    |
| UMA                 | Access time is uniform across processors                      | Location has less effect on access time             |
| NUMA                | A common address space, but access time depends on location   | Prefer memory near the processor using it           |
| Distributed memory  | Separate local address spaces; communicate explicitly         | The program must move data between tasks            |
| Hybrid              | Shared memory within a node, distributed memory between nodes | Use both local sharing and inter-node communication |

**Cache coherence** concerns consistent cached copies. It does not make a multi-step operation atomic. Two withdrawals that both read an old balance can still produce an incorrect result, even on a coherent machine.

> [!warning] Two meanings of shared memory
> A **shared-memory architecture** is a general machine organization. CUDA **shared memory** is a specific on-chip, block-scoped scratchpad. See [[GPU Memory and Performance#GPU memory spaces|GPU memory spaces]].

### Programming models and decomposition

Hardware architecture and programming model are separate choices. Message passing can run on a shared-memory machine; a shared-memory abstraction can be implemented over distributed hardware.

| Model | How work communicates |
| --- | --- |
| Shared memory | Reads and writes in a shared address space, coordinated with synchronization |
| Threads | Concurrent execution paths within a process, with private state and shared resources |
| Message passing | Tasks explicitly send and receive data, for example with MPI |
| Data parallel | Tasks apply similar operations to different partitions of a data structure |
| Hybrid | Combines models to match different parts of the application |

**Domain decomposition** partitions data, such as image regions. **Functional decomposition** partitions operations, such as different pipeline stages. CUDA vector addition is domain decomposition: thread $i$ owns output element $i$.

### What makes parallelization difficult?

- **Dependencies:** execution order can affect the result. `A[i] = 2*A[i-1]` has a loop-carried dependence, so assigning iterations to arbitrary threads does not preserve the original algorithm.
- **Communication:** independent pixel inversion needs little communication; neighborhood operations need shared input data. Communication has both latency and bandwidth costs.
- **Synchronization:** a barrier waits for a group; a lock protects a critical section. A barrier does not by itself make a shared update atomic.
- **Load imbalance:** the slowest participant delays progress at a barrier.
- **Granularity:** small tasks make balancing easier but can spend a large fraction of time communicating or synchronizing. Larger tasks amortize overhead but may leave too little parallel work.

These tradeoffs reappear as [[CUDA Grids, GPU Architecture, and Scheduling#Synchronization and block independence|block barriers]], [[CUDA Grids, GPU Architecture, and Scheduling#Waves and transparent scalability|waves]], and [[GPU Memory and Performance#Other optimization techniques|thread coarsening]].

## CPU and GPU design tradeoffs

*Source: pp. 28-33.*

| CPU emphasis | GPU emphasis |
| --- | --- |
| Low latency for a sequential instruction stream | High aggregate throughput for parallel work |
| Complex control, speculation, and out-of-order execution | Many execution units sharing control within warps |
| Large caches and mechanisms that speed up individual tasks | High memory bandwidth and many resident threads to tolerate latency |
| Sequential and irregular orchestration | Large amounts of similar, parallel computation |

A heterogeneous application uses both. Moving a small task to a GPU can lose time if allocation, transfers, or launch overhead dominate its computation. Hardware details depend on the architecture; the table captures the lecture's design principles.

CUDA provides a general compute programming interface through language extensions and runtime functions. Historically, general-purpose GPU computation often had to be expressed using graphics APIs.

## Performance and Amdahl's law

*Source: pp. 34-41.*

**Latency** is elapsed time for a job, including communication and synchronization. **Throughput** is completed work per unit time. **Strong scaling** fixes total problem size while adding processors; **weak scaling** fixes problem size per processor while increasing total work.

Let $T_1$ be sequential time and $T_p$ parallel time with $p$ processors:

$$
S_p=\frac{T_1}{T_p},\qquad E_p=\frac{S_p}{p},\qquad C_p=pT_p.
$$

For a fixed workload with parallelizable fraction $F$, assuming ideal parallelization and no extra overhead:

$$
T_p=T_1\left((1-F)+\frac{F}{p}\right),\qquad
S_p=\frac{1}{(1-F)+F/p},\qquad
S_\infty=\frac{1}{1-F}.
$$

**Worked example:** $F=0.90$, $p=8$, and $T_1=100$ s:

$$
T_8=100(0.10+0.90/8)=21.25\text{ s},\quad
S_8\approx4.71,\quad E_8\approx58.8\%,\quad S_\infty=10.
$$

For GPU offloading, a useful extension is to substitute a measured acceleration factor $s$ for the accelerated portion:

$$
T_{new}=T_{old}\left((1-F)+\frac{F}{s}\right)+T_{extra}.
$$

Here $T_{extra}$ accounts for additional transfers, launches, or other overhead. The fraction $F$ is a fraction of **baseline runtime**, not a fraction of source-code lines. Keep baseline and optimized measurement boundaries consistent.

**Connection:** Amdahl limits the application benefit; [[GPU Memory and Performance#Roofline model|roofline]] limits a kernel's compute throughput for its memory traffic. Both can constrain the same program.

## CUDA execution and data movement

*Source: pp. 49-65.*

The **host** is the CPU side; the **device** is the GPU side. In the lecture's explicit-memory model, allocating host arrays does not allocate their device counterparts.

The basic lifecycle is:

1. Prepare input and output storage on the host.
2. Allocate device buffers with `cudaMalloc`.
3. Copy inputs with `cudaMemcpy(..., cudaMemcpyHostToDevice)`.
4. Launch a kernel with `kernel<<<grid, block>>>(...)`.
5. Copy the output back with `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.
6. Free device buffers with `cudaFree`.

`cudaMemcpy` takes **destination first, source second**. Sizes are in bytes: an array of $n$ floats needs `n * sizeof(float)` bytes. Output data need not be copied to the device first if the kernel overwrites every valid output.

| Function qualifier | Execution location | Meaning |
| --- | --- | --- |
| `__host__` | CPU | Ordinary host function; default |
| `__global__` | GPU | Kernel; a host launch creates a grid of threads |
| `__device__` | GPU | Device helper function; an ordinary call does not create a new grid |

The slides also mention device-side kernel launches, but the examples here use host launches. `nvcc` separates host and device compilation; PTX is an intermediate device representation that can be compiled into machine code for a GPU. A typical source file is compiled with `nvcc example.cu -o example`.

Check CUDA API return values against `cudaSuccess` and obtain the message with `cudaGetErrorString`. The error-handling slide on p. 58 declares `err` but tests `error`; use the same variable name consistently.

## Vector addition

*Source: pp. 54-68.*

The operation is $C[i]=A[i]+B[i]$. Outputs are independent, so one thread can compute each element without a block barrier.

### Thread hierarchy and indexing

| CUDA object | Relevant built-in | Meaning |
| --- | --- | --- |
| Grid | `gridDim` | Number of blocks along each axis |
| Block | `blockIdx` | This block's coordinates in the grid |
| Block shape | `blockDim` | Threads along each block axis |
| Thread | `threadIdx` | This thread's coordinates within its block |

For a 1D launch:

$$
i=\texttt{blockIdx.x}\,\texttt{blockDim.x}+\texttt{threadIdx.x},\qquad
G=\left\lceil\frac{n}{B}\right\rceil.
$$

Here $B$ is threads per block and $G$ is blocks in the grid. A grid of $G$ blocks launches $GB$ threads; only the first $n$ should access the arrays.

```cpp
// Teaching fragment: device pointers, positive n, and a valid launch assumed.
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Host-side launch; allocation, copies, and error checks omitted.
int threads = 256;
int blocks = (n + threads - 1) / threads;
vectorAdd<<<blocks, threads>>>(a_d, b_d, c_d, n);
```

The ceiling formula assumes the addition fits the integer type. For a practical implementation, handle empty inputs and use types suitable for the largest supported problem.

**Worked example:** $n=1003$, $B=64$. There are 16 blocks and 1024 launched threads. The bounds check prevents the last 21 from accessing the arrays. Thread `(blockIdx.x=15, threadIdx.x=42)` handles index 1002, the last valid element.

### Giving a thread more work

If each thread handles **two adjacent elements**, first compute its logical thread ID $t$, then use $i_0=2t$, $i_1=2t+1$. Launch $\lceil n/(2B)\rceil$ blocks and check each element separately. Adjacent lanes now have stride two for their first scalar load. Alternative mappings and vector loads can change memory behavior; see [[GPU Memory and Performance#Global memory coalescing|coalescing]] and [[GPU Memory and Performance#Other optimization techniques|coarsening]].

## Preview of multidimensional mapping

*Source: pp. 69-78; developed in [[CUDA Grids, GPU Architecture, and Scheduling#Multidimensional mapping|Lecture 2]].*

Use $x$ for columns and $y$ for rows:

$$
col=b_xB_x+t_x,\quad row=b_yB_y+t_y,\quad
index=row\cdot width+col.
$$

For row-major RGB input, the pixel begins at `3 * index`; grayscale output uses `index`. The conceptual luminance expression on p. 50 is $0.299R+0.587G+0.114B$.

> [!warning] Slide consistency notes
> Page 75's labels for rows and columns conflict with its launch expression. Derive grid $x$ from **width** and grid $y$ from **height**. Page 77's code figure uses different grayscale weights than p. 50, and its cast placement can truncate partial results. Choose one specified weighting convention and convert to the output type after evaluating the complete weighted sum.

## Retrieval practice

These priorities are suggested study targets, not a statement of exam coverage.

1. Why does cache coherence fail to prevent the bank-withdrawal race?
2. What is the difference between domain and functional decomposition?
3. For $F=0.95$ and $p=20$, compute speedup, efficiency, and maximum speedup.
4. A kernel uses $n=200000$ and $B=128$ (p. 68). How many blocks and threads launch? How many threads compute an output?
5. Why does vector addition need a bounds check but no `__syncthreads()`?
6. Derive the first index and launch size when each thread computes two adjacent outputs.
7. Explain why a kernel-only speedup of $10\times$ might barely change application time.

> [!success]- Answer key
> 1. Coherence does not serialize the full read/check/write sequence. Both tasks can read the same old balance.
> 2. Domain splits data; functional splits operations or roles.
> 3. $S=1/(0.05+0.95/20)\approx10.26$; efficiency $\approx51.3\%$; maximum speedup $20$.
> 4. $G=1563$ blocks; $GB=200064$ threads execute the index calculation; 200000 execute the guarded output statement.
> 5. Ceiling division launches extra threads, but valid outputs do not depend on results from other threads.
> 6. $i_0=2(b_xB+t_x)$; $G=\lceil n/(2B)\rceil$; guard both $i_0$ and $i_0+1$.
> 7. The kernel may be a small fraction of baseline time, or new transfer/launch costs may consume the saved time.

## Practical review checklist

- [ ] Identify the independent output and the input data it requires.
- [ ] Check dependencies before replacing a loop with threads.
- [ ] Decide whether the dataset is large enough to amortize GPU overhead.
- [ ] Trace every buffer's allocation, copy direction, size, and lifetime.
- [ ] Derive indexing and test sizes that are not multiples of the block size.
- [ ] Check errors and compare results against a trusted sequential reference.
- [ ] Measure total application time as well as device computation time.

**Next:** [[CUDA Grids, GPU Architecture, and Scheduling]]. **Performance connection:** [[GPU Memory and Performance#Roofline examples|Vector addition's bandwidth limit]].
