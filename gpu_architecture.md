# GPU architecture and programming

CMPT 479/982

## Course overview

GPU architecture and programming studies the compute and memory principles of graphics processing units (GPUs) and how to use them effectively.

The course connects GPU compute and memory architecture with heterogeneous data-parallel programming in CUDA. It examines how threads, warps, scheduling, synchronization, and memory access influence program behavior and performance.

## Course topics

- Modern GPU architecture: threads, warps, synchronization, SIMD hardware, thread-block and warp scheduling, and control divergence.
- GPU memory hierarchy and architecture.
- GPU performance optimizations.
- Heterogeneous data-parallel computing using CUDA.
- CUDA programs and optimizations for convolution, reduction, sorting, and matrix multiplication.
- Extensions: graph operations, convolutional neural networks, and attention in large language models.

## Educational goals

- Understand the basics of GPU compute and memory architecture.
- Write GPU programs using CUDA.
- Read and analyze GPU architecture research.

## Note outline

- [Introduction and parallel programming](gpu_architecture_and_programming_cmpt_479/introduction_and_parallel_programming.md): parallel architectures and programming models, dependencies, CPU/GPU tradeoffs, Amdahl's law, host/device execution, CUDA indexing, and vector addition.
- [CUDA grids, GPU architecture, and scheduling](gpu_architecture_and_programming_cmpt_479/cuda_grids_gpu_architecture_and_scheduling.md): multidimensional mapping, image and matrix kernels, GPU hardware hierarchy, block synchronization, waves, warps, divergence, occupancy, and device queries.
- [GPU memory and performance](gpu_architecture_and_programming_cmpt_479/gpu_memory_and_performance.md): the roofline model, memory spaces, tiled matrix multiplication, shared-memory occupancy, coalescing, bank conflicts, vector accesses, coarsening, unrolling, and double buffering.
