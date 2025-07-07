# Benchmarking NVIDIA HPC Workloads on the Kempner AI Cluster


# Overview 

This benchmarking study aims to evaluate the floating-point compute performance, memory bandwidth, and system scalability of the Kempner AI Cluster using standard NVIDIA HPC Benchmark containers. The benchmarks cover four key workloads: HPL (High-Performance Linpack), HPL-MxP (mixed-precision Linpack), HPCG (High-Performance Conjugate Gradient), and STREAM (memory bandwidth test). Tests are conducted on NVIDIA Hopper GPUs (H100) in configurations ranging from single-GPU up to multi-GPU (1–8 GPUs) and multi-node scenarios. Results will provide a robust baseline for performance tuning, user expectations, and future hardware procurement decisions. Additional tests using Grace Hopper (H200) nodes may be added to compare next-generation CPU-GPU memory architectures.

# Prerequisites

The following prerequisites must be met before preparing for benchmarking:

- Access to the Kempner AI Cluster with valid user credentials and appropriate SLURM job submission permissions.

- At least one allocated NVIDIA Hopper (H100) GPU per test; multi-GPU and multi-node runs require correct GPU partition and Infiniband network access.

- Podman installed locally (or on a build node) to pull official NVIDIA HPC benchmark container images.

- Singularity (or Apptainer) installed and configured for container execution in the cluster environment.

- Basic familiarity with SLURM job submission (srun, sbatch), resource allocation flags (--gres, --ntasks), and container runtime options.

- Sufficient scratch storage or a working directory to store benchmark input files, result logs, and Singularity image files (.sif).

- Access to NVIDIA NGC account for pulling the most up-to-date benchmark containers. (A copy of the container images can also be provided by the cluster administrators.)

- Recommended: Familiarity with HPL benchmarking concepts, including problem size (N), process grid (P x Q), and performance metrics (GFLOPS, efficiency).


# NVIDIA HPC Container

For this benchmarking effort, we use the official NVIDIA HPC Benchmark container, which conveniently packages industry-standard workloads,  including HPL, HPL-MxP, HPCG, and STREAM, with tuned libraries and drivers to deliver consistent, reproducible performance. This container ensures compatibility with the latest NVIDIA architectures and provides an easy way to run demanding HPC tests without building complex software stacks from scratch. See the link below for details on how we pull, convert, and deploy this container on the Kempner AI Cluster.

- [NVIDIA HPC Benchmark](https://docs.nvidia.com/nvidia-hpc-benchmarks/overview.html)
- [Convert NVIDIA HPC Benchmark Container to Singularity](p0_ngc_to_singularity/README.md)

# NVIDIA HPL Benchmark

HPL (High-Performance Linpack) is a widely used benchmark for measuring the floating-point compute performance of supercomputers. It solves a dense system of linear equations and is the basis for the TOP500 list of supercomputers. In this section, we will run the HPL benchmark using the NVIDIA HPC container on the Kempner AI Cluster.

- [NVIDIA HPL Benchmark on Kempner AI Cluster](p1_nvidia_hpl/README.md)

Here is the summary of the HPL benchmark runs on the Kempner AI Cluster:

| # Nodes | # GPUs | N      | NB   | P | Q | Time (s) | GFLOPS (Per. GPU GFLOPS) |
|---------|--------|--------|------|---|---|----------|--------------------------|
| 1       | 1      | 92160  | 1024 | 1 | 1 | 12.40    | 4.208e+04 ( 4.208e+04)   |
| 1       | 2      | 136192 | 1024 | 2 | 1 | 20.28    | 8.304e+04 ( 4.152e+04)   |
| 1       | 4      | 190464 | 1024 | 2 | 2 | 27.35    | 1.684e+05 ( 4.210e+04)   |
| 2       | 4      | 264192 | 1024 | 4 | 2 | 37.49    | 3.279e+05 ( 4.099e+04)   |  


# NVIDIA HPL-MxP Benchmark

HPL-MxP (High-Performance Linpack - Mixed Precision) is an enhanced version of the traditional HPL benchmark that leverages NVIDIA’s Tensor Core acceleration for mixed-precision matrix operations, delivering significantly higher performance on supported GPUs. HPL-MxP demonstrates how modern hardware can use mixed precision arithmetic combined with iterative refinement to solve dense linear systems faster while maintaining double-precision accuracy. In this section, we run the HPL-MxP benchmark using the NVIDIA HPC container on the Kempner AI Cluster.

- [NVIDIA HPL-MxP Benchmark on Kempner AI Cluster](p2_nvidia_hpl_mxp/README.md)

Here is a summary of MxP benchmark (using `FP16` for LU factorization) runs on the Kempner AI Cluster using NVIDIA Hopper GPUs (H100):

| Metric               | 1 N 1 GPU  | 1 N 2 GPUs | 1 N 4 GPUs | 2 N 4 GPUs |
|----------------------|------------|------------|------------|------------|
| Problem Size (N)     | 92160      | 136192     | 190464     | 264192     |
| Block Size (NB)      | 1024       | 1024       | 1024       | 1024       |
| Grid Size (P x Q)    | 1 x 1      | 2 x 1      | 2 x 2      | 4 x 2      |
| GFLOPS               | 6.1567e+04 | 6.6512e+04 | 1.9768e+05 | 8.1169e+05 |
| GFLOPS (Per GPU)     | 61566.51   | 33255.97   | 49420.46   | 101461.07  |
| LU GFLOPS            | 2.3786e+05 | 4.4344e+05 | 8.7595e+05 | 1.9651e+06 |
| LU GFLOPS (Per GPU)  | 237855.39  | 221721.54  | 218987.25  | 245636.39  |


# NVIDIA HPCG Benchmark

HPCG (High Performance Conjugate Gradients) is designed to complement HPL by providing a more realistic benchmark for modern HPC systems. Unlike HPL, which is compute-bound, HPCG stresses memory system performance, data movement, and interconnect efficiency, resembling the sparse linear algebra workloads found in many real-world scientific applications.

- [NVIDIA HPCG Benchmark on Kempner AI Cluster](p3_nvidia_hpcg/README.md)

Here is a summary of the HPCG benchmark runs on the Kempner AI Cluster using NVIDIA Hopper GPUs (H100):

| Compute Config | HPCG GFLOP/s  |
|----------------|---------------|
| 1 N 1 GPU      | 515.269       |
| 1 N 2 GPUs     | 991.491       |
| 1 N 4 GPUs     | 1969.54       |
| 2 N 4 GPUs     | 3762.01       |


# NVIDIA STREAM Benchmark

The STREAM benchmark is a simple yet powerful memory bandwidth test originally designed to measure sustainable memory throughput for CPUs. NVIDIA’s GPU-accelerated version extends this to modern GPUs, measuring the achievable bandwidth of on-device memory (HBM) by performing simple vector operations like COPY, SCALE, ADD, and TRIAD. It does not include PCIe or interconnect transfers — it focuses purely on how fast the GPU can read and write data within its own high-bandwidth memory. For more details on the STREAM benchmark, see the following resources:

- [NVIDIA STREAM Benchmark on Kempner AI Cluster](p4_nvidia_stream/README.md)

Here is the summary of the STREAM benchmark runs on the Kempner AI Cluster using NVIDIA Hopper GPUs (H100):

| Function    | FP32 Bandwidth (MB/s)  | % Peak |   FP64 Bandwidth (MB/s)  | % Peak | ECC  |
|-------------|------------------------|--------|--------------------------|--------|------|
| COPY        | 3065453.4495           |  91.44 |  3071120.9703            | 91.61  | Off  |
| SCALE       | 3065829.6062           |  91.45 |  3059058.0968            | 91.25  | Off  |
| ADD         | 3119722.3842           |  93.06 |  3125260.4738            | 93.22  | Off  |
| TRIAD       | 3121150.5004           |  93.10 |  3127058.6779            | 93.28  | Off  |


Done!
