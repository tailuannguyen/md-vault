# CUDA Data Mapping and Kernels

Source: `02_Arch_Scheduling.pdf`, slides 4-20. The grayscale formula also appears in `01_Introduction.pdf`.

## Coordinates and linear storage

`blockIdx` identifies a block, `threadIdx` a thread inside it, and `blockDim` and `gridDim` describe the launch. For a row-major image or matrix:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < height && col < width) {
    int offset = row * width + col;
}
```

A column-major array instead maps `(row, col)` to `col * height + row`. A 3D row-major tensor with width `W` and height `H` maps `(z, y, x)` to `(z * H + y) * W + x`. Coordinates describe work; the linear address describes where data lives.

For a `height × width` image and `16 × 16` blocks, use grid dimensions `((width + 15) / 16, (height + 15) / 16)`. The lecture's 62-row, 76-column image needs a `5 × 4` grid. The final blocks include threads outside the image, so guard both coordinates. The slides use a 1024-thread maximum per block; query the target device for its actual limits.

## Three one-output-per-thread examples

- **Grayscale:** for interleaved RGB, the input pixel starts at `3 * (row * width + col)`. Its output is at `row * width + col`. A common luminance approximation is `0.299 R + 0.587 G + 0.114 B`.
- **Blur:** a radius-one output uses a 3 × 3 input patch. Boundary pixels need a defined policy, such as averaging only valid neighbors or padding the input. Adjacent outputs read overlapping patches.
- **Matrix multiplication:** for square row-major matrices of width `N`, thread `(row, col)` computes `C[row * N + col] = sum_k A[row * N + k] * B[k * N + col]`. Guard the output coordinates. Neighboring outputs repeatedly request some of the same inputs, although caches may serve some requests.

The matrix example directly motivates [[Roofline and Arithmetic Intensity|the comparison between requested memory traffic and ideal reuse]].

**Next:** [[GPU Execution and Occupancy]] shows how these logical threads become resident blocks and warps.
