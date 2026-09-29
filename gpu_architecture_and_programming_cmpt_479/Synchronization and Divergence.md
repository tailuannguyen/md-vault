# Synchronization and Divergence

Source: `02_Arch_Scheduling.pdf`, slides 27-35.

## Block barriers

`__syncthreads()` is a block-wide barrier: all participating block threads must arrive before any continues. It is useful after a cooperative shared-memory load, so no thread reads an incomplete tile. A second barrier is needed before shared storage is overwritten while another thread may still be reading it.

Do not put a block barrier behind a condition that only some block threads satisfy. At image or matrix boundaries, out-of-bounds threads can load zero into a shared tile and still participate in the barriers; only the final output store needs to be conditional. Ordinary blocks do not have an implicit grid-wide barrier. Supported cooperative APIs provide broader synchronization when explicitly used.

## Control divergence

When lanes in one warp take different branches, the warp executes the required paths with only matching lanes active. This can happen in loops as well as `if` statements. The last block's bounds check is a common, usually small case. A branch that separates entire warps does not diverge within a warp.

Newer GPU generations support independent thread scheduling, so code should not rely on implicit communication between lanes merely because they share a warp. Use an appropriate warp synchronization primitive when a warp-level exchange requires one.

Barriers can leave early-arriving threads waiting, while divergence can reduce useful work per issue. Adding resident warps may hide some waiting but cannot fix an incorrect barrier or eliminate divergent work.
