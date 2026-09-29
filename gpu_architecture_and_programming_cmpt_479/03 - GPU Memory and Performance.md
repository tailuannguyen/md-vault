# 03 - GPU Memory and Performance

Source: [[raw_files/03_Memory_Perf_Optimizations.pdf]]

This stream starts with a performance model, then changes how kernels move and reuse data:

**Start:** [[Roofline and Arithmetic Intensity]] → CUDA memory and tiling → memory access and optimization.

The roofline model helps identify whether arithmetic throughput or memory bandwidth is the tighter ceiling. Storage choices, tiling, and access patterns then explain how a kernel can approach that ceiling. For definitions and formulas at a glance, see [[gpu_memory_architecture|GPU Memory Architecture - Quick Reference]].
