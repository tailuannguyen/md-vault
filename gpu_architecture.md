# GPU architecture and programming

CMPT 479/982

## Course overview

The calendar description covers current topics in computing systems, selected according to faculty and student interest. This offering focuses on the architecture and programming principles of graphics processing units (GPUs).

The course connects GPU compute and memory architecture with heterogeneous data-parallel programming in CUDA. It examines how threads, warps, scheduling, synchronization, and memory access influence program behavior and performance.

Alongside textbook reading, students read original research papers, complete homework assignments, and collaborate on a large programming project. The course is cross-listed with the graduate course CMPT 982 G400. The course description identifies this as its first offering, with possible changes to topics and grading guidelines.

## Course topics

- Modern GPU architecture: threads, warps, synchronization, SIMD hardware, thread-block and warp scheduling, and control divergence.
- GPU memory hierarchy and architecture.
- GPU performance optimizations.
- Heterogeneous data-parallel computing using CUDA.
- CUDA programs and optimizations for convolution, reduction, sorting, and matrix multiplication.
- Time permitting: graph operations, convolutional neural networks, and attention in large language models.

## Educational goals

- Understand the basics of GPU compute and memory architecture.
- Write GPU programs using CUDA.
- Read, review, and present research papers.
- Collaborate on a large programming project using CUDA.

## Note outline

- [Introduction and parallel programming](gpu_architecture_and_programming_cmpt_479/introduction_and_parallel_programming.md): parallel architectures and programming models, dependencies, CPU/GPU tradeoffs, Amdahl's law, host/device execution, CUDA indexing, and vector addition.
- [CUDA grids, GPU architecture, and scheduling](gpu_architecture_and_programming_cmpt_479/cuda_grids_gpu_architecture_and_scheduling.md): multidimensional mapping, image and matrix kernels, GPU hardware hierarchy, block synchronization, waves, warps, divergence, occupancy, and device queries.
- [GPU memory and performance](gpu_architecture_and_programming_cmpt_479/gpu_memory_and_performance.md): the roofline model, memory spaces, tiled matrix multiplication, shared-memory occupancy, coalescing, bank conflicts, vector accesses, coarsening, unrolling, and double buffering.
