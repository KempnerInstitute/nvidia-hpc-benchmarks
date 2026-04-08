# NVIDIA HPL-MxP Benchmark

The HPL-MxP (High-Performance Linpack - Mixed Precision) benchmark is an advanced version of the traditional HPL benchmark that utilizes NVIDIA’s Tensor Core acceleration for mixed-precision matrix operations. It is designed to deliver significantly higher performance on supported GPUs by combining mixed precision arithmetic with iterative refinement to solve dense linear systems faster while maintaining double-precision accuracy. The HPL-MxP benchmark seeks to highlight the emerging convergence of high-performance computing (HPC) and artificial intelligence (AI) workloads, where mixed precision is increasingly used to accelerate training and inference tasks.

- [HPL-MxP](https://hpl-mxp.org/)


The purpose of the benchmark is to solve a system of linear equations to full 64-bit floating-point accuracy by first performing a mixed-precision factorization of the matrix (LU decomposition) to obtain an approximate solution in lower precision. Then, an iterative solver (such as GMRES) runs in 64-bit precision to refine this approximate solution. The low-precision LU factors act as a preconditioner for the iterative method, allowing the final result to reach the same level of accuracy as a full 64-bit LU decomposition, but with significantly improved performance. Read more about the rules [here](https://hpl-mxp.org/rules.md).


# Results

These benchmark suites support `FP8` and `FP16` mixed precision modes for LU factorization, as well as `FP64` for the iterative refinement step. The precision mode is controlled via the `--sloppy-type` flag:

| `--sloppy-type` | Precision | Description                              |
|-----------------|-----------|------------------------------------------|
| 0               | FP64      | Full double-precision (emulated on TC)   |
| 1               | FP8       | FP8 (E4M3) LU factorization             |
| 2               | FP16      | FP16 LU factorization (default in runs)  |

We present the results for `FP16` mixed precision mode (`--sloppy-type 2`) in this section. To run in `FP8` mode, add `--sloppy-type 1` to the command line.

> **Note:** HPL-MxP is a *Linpack-style* benchmark — it measures how fast a dense linear system can be solved using mixed-precision LU factorization with iterative refinement to FP64 accuracy. It is **not** a tensor-core saturation microbenchmark. For raw tensor-core throughput measurement, see [`p5_tensor_gemm/`](../p5_tensor_gemm/README.md).


| Metric               | 1 N 1 GPU  | 1 N 2 GPUs | 1 N 4 GPUs | 2 N 8 GPUs |
|----------------------|------------|------------|------------|------------|
| Problem Size (N)     | 92160      | 136192     | 190464     | 264192     |
| Block Size (NB)      | 1024       | 1024       | 1024       | 1024       |
| Grid Size (P x Q)    | 1 x 1      | 2 x 1      | 2 x 2      | 4 x 2      |
| GFLOPS               | 6.1567e+04 | 6.6512e+04 | 1.9768e+05 | 8.1169e+05 |
| GFLOPS (Per GPU)     | 61566.51   | 33255.97   | 49420.46   | 101461.07  |
| LU GFLOPS            | 2.3786e+05 | 4.4344e+05 | 8.7595e+05 | 1.9651e+06 |
| LU GFLOPS (Per GPU)  | 237855.39  | 221721.54  | 218987.25  | 245636.39  |

The final results demonstrate that scaling the number of GPUs significantly increases total performance for both the base computation (GFLOPS) and the LU factorization workload (LU GFLOPS). As the problem size grows proportionally with the number of GPUs and nodes, total GFLOPS improves markedly — from about 61.6 TFLOPS on one GPU to over 811.7 TFLOPS on 2 nodes with 8 GPUs. Notably, the LU GFLOPS (which reflects the core linear algebra workload) scales nearly linearly with GPU count, showing strong parallel efficiency, reaching nearly 2 PFLOPS on the largest configuration.

## Parameterized Launcher

Instead of maintaining separate scripts for each GPU/precision configuration, you can use the parameterized launcher [`run_hpl_mxp.sh`](run_hpl_mxp.sh) which supports:

- **Explicit precision control**: Set `SLOPPY_TYPE=FP16` or `SLOPPY_TYPE=FP8` (mapped to `--sloppy-type 2` or `1` respectively).
- **Architecture gating**: FP4 is automatically blocked on Hopper (requires Blackwell SM ≥ 100).
- **Flexible scaling**: Set `NODES`, `GPUS_PER_NODE`, `N`, `NB`, `P`, `Q` as environment variables.

Example:
```bash
# FP16 mode on 4 GPUs
NODES=1 GPUS_PER_NODE=4 SLOPPY_TYPE=FP16 N=190464 P=2 Q=2 sbatch run_hpl_mxp.sh

# FP8 mode on 8 GPUs across 2 nodes
NODES=2 GPUS_PER_NODE=4 SLOPPY_TYPE=FP8 N=264192 P=4 Q=2 sbatch run_hpl_mxp.sh
```

The original per-configuration scripts are preserved in the [`runs/`](runs/) directory for reference.