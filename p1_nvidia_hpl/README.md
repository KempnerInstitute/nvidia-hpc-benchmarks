# NVIDIA HPL Benchmark

HPL (High-Performance Linpack) is the industry-standard benchmark for measuring a system’s floating-point computing performance. It solves a dense system of linear equations ($Ax = b$) using LU factorization with partial pivoting. The primary output is GFLOPS (Giga Floating Point Operations per Second) — a practical measure of how much raw numerical throughput the system can deliver.

NVIDIA provides an HPL container through the NGC registry, pre-optimized to run on NVIDIA GPUs using CUDA libraries, NVSHMEM, and NCCL for distributed computing.

## Problem Definition

At its core, HPL (High Performance Linpack) solves a dense system of linear equations:

$$Ax = b$$

where:
- $A$ is a large matrix of size $N \times N$.
- $x$ is the vector of unknowns.
- $b$ is the right-hand side vector.

The goal is to find the vector $x$ that satisfies this equation.

HPL uses $LU$ factorization with partial pivoting to decompose the matrix $A$ into a product of a lower triangular matrix $L$ and an upper triangular matrix $U$. The solution is then obtained in two steps:
1. **Forward substitution** to solve $Ly = b$ for $y$.
2. **Backward substitution** to solve $Ux = y$ for $x$. 

The factorization requires $O(\frac{2}{3} N^3)$ operations, while the forward and backward substitutions require $O(N^2)$ operations each. Thus, the overall complexity is dominated by the factorization step.

## How is it parallelized?

HPL:
- Divides $A$ into blocks ($NB$)
- Distributes these blocks over a 2D process grid ($P \times Q$)
- Uses block-cyclic distribution to balance the workload.
- Each process does local matrix operations and then communicates the partial results using MPI or NCCL/NVSHMEM for GPU acceleration.

## How is correctness checked?

HPL checks correctenss by computing the residual:

$$ Residual = \frac{||Ax -b||_\infty}{\epsilon{||A||_\infty}||x||_\infty +||b||_\infty)N}$$

Where:
- $\epsilon$ is the machine precision. (for double, $~10^{-16}$)
- A low residual (typically less than TODO) indicates a correct solution.

## How is performance measured?

HPL measures performance in GFLOPS (Giga Floating Point Operations per Second):

$$ GFLOPS = \frac{\frac{2}{3} \cdot N^3}{t \cdot 10^9} $$

Where:
- $N$ is the size of the matrix.
- $t$ is the time taken to solve the system (in seconds).


## Run Results

We tried the run with 1, 2, 4 GPUs on a single node, and 8 GPUs on two nodes. The results are as follows:

| # Nodes | # GPUs | N      | NB   | P | Q | Time (s) | GFLOPS (Per. GPU GFLOPS) |
|---------|--------|--------|------|---|---|----------|--------------------------|
| 1       | 1      | 92160  | 1024 | 1 | 1 | 12.40    | 4.208e+04 ( 4.208e+04)   |
| 1       | 2      | 136192 | 1024 | 2 | 1 | 20.28    | 8.304e+04 ( 4.152e+04)   |
| 1       | 4      | 190464 | 1024 | 2 | 2 | 27.35    | 1.684e+05 ( 4.210e+04)   |
| 2       | 4      | 264192 | 1024 | 4 | 2 | 37.49    | 3.279e+05 ( 4.099e+04)   |  



