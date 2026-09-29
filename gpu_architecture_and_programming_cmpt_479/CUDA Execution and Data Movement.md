# CUDA Execution and Data Movement

Source: `01_Introduction.pdf`, slides 49-68.

## Host and device

The CPU runs host code and launches GPU kernels. A kernel is device code run by many threads over a grid of blocks. In the basic model shown in the slides, host and device allocations are distinct: `cudaMalloc` allocates device global memory, `cudaMemcpy` transfers data, and `cudaFree` releases device memory.

For vector addition, allocate and initialize host arrays; allocate device arrays; copy inputs to the GPU; launch the kernel; copy the output back when needed; and release resources. Keeping data on the device across several kernels avoids extra transfers.

## One thread per element

```cpp
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int threadsPerBlock = 256;
int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocks, threadsPerBlock>>>(a_d, b_d, c_d, n);
```

The grid replaces a serial loop: thread `i` owns `c[i]`. Ceiling division launches enough blocks; the bounds check protects extra threads in the last block. `__global__` marks a kernel entry point, while `__device__` marks a function executed on the GPU and called from device code.

Check CUDA API and kernel errors when debugging. Kernel launches are normally asynchronous with respect to the host, so synchronize before reading results or timing completed GPU work; a blocking device-to-host copy in the same stream also waits for preceding work.

**Next stream:** [[CUDA Data Mapping and Kernels]] extends the same index formula to images and matrices.
