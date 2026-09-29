# Roofline and Arithmetic Intensity

Source: `03_Memory_Perf_Optimizations.pdf`, slides 5-10 and 20.

## Compute and bandwidth ceilings

Peak compute throughput measures arithmetic per second; peak memory bandwidth measures bytes per second. Arithmetic intensity is `I = FLOPs / bytes transferred from global memory`, in FLOP/B. The roofline bound is:

$$P \leq \min(P_{\mathrm{peak}}, B_{\mathrm{peak}} I).$$

The ridge point is `I_ridge = P_peak / B_peak`. Using the lecture's illustrative H100 values, 67 TFLOP/s ÷ 3.35 TB/s = 20 FLOP/B. Below that intensity, the bandwidth ceiling is lower; above it, the compute ceiling is lower. These are upper bounds, not measured performance.

## Vector addition

One FP32 addition reads two 4-byte inputs and writes one 4-byte output: `1 FLOP / 12 B ≈ 0.083 FLOP/B`, assuming the stated global-memory traffic. At a 3.35 TB/s bandwidth ceiling, its roofline limit is about 0.28 TFLOP/s. The lecture's one-billion-element example moves 12 GB; at 3.35 TB/s, the minimum transfer time is about 3.6 ms under this model. Whole-program time may also include host-device transfers.

## Matrix multiplication depends on reuse

For square width `N`, matrix multiplication does roughly `2N³` FLOPs. Counting each of the three `N²` matrices only once gives an idealized `2N³ / (12N²) = N/6 FLOP/B` for FP32. That assumes nearly perfect reuse. A basic one-output-per-thread kernel instead requests two 4-byte inputs for roughly two FLOPs in each inner-loop step: `2 / 8 = 0.25 FLOP/B` if those requests reach global memory. Caches can change actual DRAM traffic.

In the lecture's simplified model, a `T × T` shared-memory tile cuts input loads by about `T`, raising intensity to about `T/4 FLOP/B` when output traffic and other effects are ignored. For `T = 32`, that is about 8 FLOP/B.

**Next:** [[CUDA Memory and Tiling]] explains where that reuse occurs.
