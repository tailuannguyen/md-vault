# Parallel Computing Foundations

Source: `01_Introduction.pdf`, slides 14-48.

## Independent work and dependencies

Parallel execution overlaps work that can proceed independently. If two tasks update the same account balance, an unsynchronized read and write can lose an update. A parallel design therefore identifies independent tasks and the communication needed between dependent tasks.

Flynn's taxonomy describes instruction and data streams. SIMD applies one instruction to multiple data elements; MIMD permits different instruction streams. A GPU has several levels of execution, so the taxonomy should not be confused with the CUDA warp model used later.

## Memory and programming models

| Architecture | Data access | Coordination |
| --- | --- | --- |
| Shared memory | Processors access a common address space. | Synchronize conflicting reads and writes. |
| Distributed memory | Each processor owns local memory. | Exchange data explicitly. |
| Hybrid | Shared memory within nodes, messages across nodes. | Use both forms at their respective scopes. |

UMA and NUMA are shared-memory designs with uniform and nonuniform access times. Location can affect performance even when an address is reachable.

Shared-memory threads, message passing, and data-parallel programming are programming models rather than one-to-one hardware categories. In data parallelism, tasks perform similar operations on different elements. Grayscale conversion is an example: each input pixel can compute one output luminance. A blur also assigns one output per task, but neighboring tasks read overlapping input patches.

Fine-grained tasks can balance uneven work but may coordinate frequently. Coarser tasks do more between coordination points, though too few tasks can leave hardware idle.

**Next:** [[CUDA Execution and Data Movement]] turns data-parallel tasks into CUDA threads.
