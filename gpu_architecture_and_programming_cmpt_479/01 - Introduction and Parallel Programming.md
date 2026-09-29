# 01 - Introduction and Parallel Programming

Source: [[raw_files/01_Introduction.pdf]]

This stream moves from general parallel computing to a first CUDA program:

**Start:** [[Parallel Computing Foundations]] → CUDA execution and data movement.

Data parallelism divides a collection into similar, mostly independent tasks. CUDA expresses those tasks as threads in a grid: the CPU prepares data and launches the grid, and each GPU thread selects its own element. The vector-add example puts this model into code.

The source slides also contain Fall 2026 logistics and policies. Check the current course site for changes.
