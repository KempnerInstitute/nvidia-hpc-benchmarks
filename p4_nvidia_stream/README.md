# NVIDIA STREAM Benchmark

The original STREAM benchmark is a simple synthetic benchmark program that measures sustainable memory bandwidth (in MB/s) and the corresponding computation rate for simple vector kernels. Here are resources to read more about the STREAM benchmark:

- [STREAM Benchmark Overview](http://www.cs.virginia.edu/stream/ref.html)
- [NVIDIA GRACE CPU Benchmarking Guide](https://nvidia.github.io/grace-cpu-benchmarking-guide/foundations/STREAM/index.html)

STREAM-GPU provides cuda kernels that explicitly mimics COPY, SCALE, ADD, and TRIAD operations on the GPU. The STREAM-GPU benchmark is designed to measure the memory bandwidth of the GPU and the CPU in a hybrid system. In other words, it is CUDA implementation of the original STREAM benchmark.

Here are the list of tasks that we will run in this section:

- COPY: $a[i] = b[i]$
- SCALE: $a[i] = scaler * b[i]$
- ADD: $a[i] = b[i] + c[i]$
- TRIAD: $a[i] = b[i] + scaler * c[i]$

You can specify the test with `--t` option, e.g., `--t COPY` to run the COPY test. If no test is specified, it will run all tests by default.

> [!NOTE]
> The STREAM benchmark measures the memory bandwidth of a single device’s local memory (for GPUs) or main memory (for CPUs). It does not measure inter- or intra-node communication bandwidth. For that, you can use the NVIDIA NCCL tests.

# Results

The following table summarizes the results of the STREAM benchmark on the Kempner AI Cluster using NVIDIA Hopper GPUs (H100). The results are presented in terms of memory bandwidth in MB/s for both single-precision (FP32) and double-precision (FP64) floating-point operations.

| Function    | FP32 Bandwidth (MB/s)  | % Peak |   FP64 Bandwidth (MB/s)  | % Peak | ECC  |
|-------------|------------------------|--------|--------------------------|--------|------|
| COPY        | 3065453.4495           |  91.44 |  3071120.9703            | 91.61  | Off  |
| SCALE       | 3065829.6062           |  91.45 |  3059058.0968            | 91.25  | Off  |
| ADD         | 3119722.3842           |  93.06 |  3125260.4738            | 93.22  | Off  |
| TRIAD       | 3121150.5004           |  93.10 |  3127058.6779            | 93.28  | Off  |

The Peak Memory Bandwidth for the H100 GPU is approximately 3.35 TB/s (read more [here](https://lenovopress.lenovo.com/lp1613-thinksystem-sd665-n-v3-server)). This value is also computed internally in the STREAM report, mostly by collecting device properies using `cudaGetDeviceProperties`. The H100 GPUs tested in this benchmark provides excellent memory bandwidth performance, achieving over 90% of the theoretical peak bandwidth for both single-precision and double-precision operations across all STREAM tests. The results indicate that the H100 GPUs are well-suited for memory-intensive applications, providing high throughput for data movement operations.

