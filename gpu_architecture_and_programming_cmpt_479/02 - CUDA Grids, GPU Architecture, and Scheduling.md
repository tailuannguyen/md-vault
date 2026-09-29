# 02 - CUDA Grids, GPU Architecture, and Scheduling

Source: [[raw_files/02_Arch_Scheduling.pdf]]

This stream connects a program's grid to the GPU hardware that runs it:

**Start:** [[CUDA Data Mapping and Kernels]] → GPU execution and occupancy → synchronization and divergence.

Thread coordinates choose output elements. Blocks group threads that can cooperate; the GPU assigns each block to a streaming multiprocessor (SM). The SM divides threads into warps, schedules ready warps, and shares finite resources among resident blocks. These hardware details explain why a correct launch layout can still have different performance on different GPUs.
