# NVIDIA HPCG Benchmark

The High Performance Conjugate Gradients (HPCG) Benchmark is designed to provide an alternative metric for ranking high-performance computing (HPC) systems. While the High Performance LINPACK (HPL) benchmark — which drives the TOP500 list — remains representative of certain large-scale applications, it does not reflect the performance characteristics of many modern, real-world workloads.

HPCG complements HPL by stressing computational and memory access patterns common in a broader range of applications, encouraging system architects to optimize hardware features that matter for these workloads.

HPCG is a self-contained benchmark that measures the performance of fundamental operations frequently used in scientific and engineering codes, including:

- Sparse matrix-vector multiplication
- Vector updates
- Global dot products
- Local symmetric Gauss-Seidel smoothing
- Sparse triangular solves (as part of the Gauss-Seidel step)

Read more about the HPCG benchmark in the following resources:
- [HPCG Benchmark Overview](https://www.hpcg-benchmark.org/)
- [Innovative Computing Laboratory - HPCG](https://icl.utk.edu/files/print/2016/hpcg-sc16.pdf)


# Results

The following table summarizes the results of the HPCG benchmark runs on the Kempner AI Cluster using NVIDIA Hopper GPUs (H100). The results are presented in terms of performance in GFLOPS (Giga Floating Point Operations per Second). 

Common parameters for all runs:
- Problem size: $256 \times 256 \times 256$
- GPU type: NVIDIA H100 80GB HBM3
- Run time: 30 minutes


| Compute Config | HPCG GFLOP/s  |
|----------------|---------------|
| 1 N 1 GPU      | 515.269       |
| 1 N 2 GPUs     | 991.491       |
| 1 N 4 GPUs     | 1969.54       |
| 2 N 4 GPUs     | 3762.01       |

