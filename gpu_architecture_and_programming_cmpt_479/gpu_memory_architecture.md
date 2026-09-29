# GPU Memory Architecture - Quick Reference

Source: `03_Memory_Perf_Optimizations.pdf`.

| Question | Concept |
| --- | --- |
| How much arithmetic can the GPU perform? | Peak compute throughput, in FLOP/s. |
| How fast can global memory supply data? | Peak bandwidth, in B/s. |
| How much work is done per byte? | Arithmetic intensity = FLOPs / global-memory bytes, in FLOP/B. |
| Where can inputs be reused? | Registers, shared memory, and caches. |
| Can neighboring lanes share transactions? | Global-memory coalescing. |

The roofline bound is $P \leq \min(P_{\mathrm{peak}}, B_{\mathrm{peak}}I)$. The ridge point is $I_{\mathrm{ridge}} = P_{\mathrm{peak}} / B_{\mathrm{peak}}$. These are ceilings; measured throughput can be lower.
