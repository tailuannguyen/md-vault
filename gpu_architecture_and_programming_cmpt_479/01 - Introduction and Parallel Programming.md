# Introduction and Parallel Programming

Source: [[raw_files/01_Introduction.pdf]]

## Course focus

CMPT 479/982 studies GPU compute and memory architecture and CUDA programming. The course combines architectural concepts with implementation and optimization of data parallel programs. Topics include threads, warps, synchronization, SIMD hardware, scheduling, control divergence, memory hierarchy, and algorithms such as convolution, reduction, sorting, and matrix multiplication.

## Parallel architecture

Parallel execution performs independent work at the same time across multiple processing elements. Flynn’s taxonomy classifies systems by instruction and data streams; GPUs are commonly used for data parallel work, where many data elements receive similar operations.

Memory organization affects how processors communicate and share data:

- **Shared memory systems:** processors access a common address space.
- **Distributed memory systems:** each processor has local memory and communicates by exchanging data.
- **Hybrid systems:** combine shared memory within nodes and distributed memory across nodes.

Programming models describe how parallel work and communication are expressed. Common models include shared memory threads, message passing, and data parallel programming. CUDA uses a heterogeneous model: CPU code manages the application and launches kernels, while GPU kernels process data in parallel.

## Heterogeneous CPU/GPU execution

The CPU (host) and GPU (device) have distinct roles and memory. A typical workflow is:

1. Allocate and initialize host data.
2. Allocate device memory and copy inputs to the GPU.
3. Launch a kernel with a grid of thread blocks.
4. Copy results back when the CPU needs them.
5. Release host and device resources.

Keeping data on the device across multiple kernels can avoid unnecessary transfer overhead.

## CUDA kernel model

A kernel is a function executed by many GPU threads. Threads are grouped into blocks; blocks form a grid. Threads use built-in indices to identify their work. A common one-dimensional index is:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

For a vector operation, each valid thread can process one element. The launch configuration controls the number of blocks and threads per block; a bounds check is needed when the total number of launched threads exceeds the data length.

## Course logistics reference

The source slides contain course schedule, assessment, policies, and contact details. Check the current course site for changes; the slides say that its information may be updated.

