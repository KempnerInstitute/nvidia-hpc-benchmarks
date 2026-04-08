# NVIDIA Tensor-Core GEMM Saturation Benchmark

This benchmark measures raw tensor-core throughput by running dense GEMM operations (C = αAB + βC) via cuBLASLt. It answers the question:

> "How close can my GPU get to its theoretical peak FP16 / FP8 tensor-core TFLOPS?"

This is NOT a Linpack benchmark. For Linpack-like mixed-precision benchmarking (solving Ax=b with iterative refinement), use [HPL-MxP](../p2_nvidia_hpl_mxp/README.md) instead.

## Why This Benchmark Exists

The NVIDIA HPC Benchmarks container includes a GEMM microbenchmark, but that tool targets FP64 / FP64-emulation workloads — it does not saturate Hopper FP16/FP8 tensor cores. This cuBLASLt-based benchmark fills that gap by directly exercising:

- **FP16 Tensor Cores** (HMMA instructions) on Hopper
- **FP8 (E4M3) Tensor Cores** on Hopper
- **FP4 Tensor Cores** on Blackwell (future — gated, not available on H100/H200)

## Theoretical Peak Throughput

| GPU | Architecture | FP16 TC (dense) | FP8 TC (dense) | FP4 TC (dense) |
|-----|-------------|-----------------|----------------|----------------|
| H100 SXM | Hopper (SM 90) | 989.4 TFLOPS | 1978.9 TFLOPS | N/A |
| H200 SXM | Hopper (SM 90) | 989.4 TFLOPS | 1978.9 TFLOPS | N/A |
| B200 | Blackwell (SM 100) | TBD | TBD | TBD |

> **Note:** H100 and H200 have identical compute SMs. H200's advantage is more HBM3e capacity/bandwidth, not higher tensor-core throughput. With sparsity (2:4 structured), throughput doubles, but this benchmark measures dense GEMMs.

## Precision Support by Architecture

| Precision | Hopper (H100/H200) | Blackwell | RTX Blackwell |
|-----------|:------------------:|:---------:|:-------------:|
| FP16 | ✓ | ✓ | ✓ |
| FP8 (E4M3) | ✓ | ✓ | ✓ |
| FP4 / NVFP4 | ✗ | ✓ | ✓ |

FP4 is hard-gated: the benchmark will refuse to run FP4 on Hopper GPUs.

## Benchmark Modes

| Mode | Description | Shapes |
|------|-------------|--------|
| `smoke` | Quick sanity check | Single 8192×8192×8192 |
| `saturation` | Find near-peak throughput | Sweep [4096, 8192, 16384, 32768]³ |
| `transformer` | ML-relevant projections | QKV, FFN, attention shapes |
| `custom` | User-specified | Set `GEMM_M`, `GEMM_N`, `GEMM_K` |

## Quick Start

```bash
# FP16 smoke test (submit via SLURM):
sbatch p5_tensor_gemm/examples/h100-fp16-smoke.sh

# FP8 saturation sweep:
sbatch p5_tensor_gemm/examples/h100-fp8-saturation.sh

# Or use the parameterized launcher directly:
DTYPE=fp16 MODE=saturation sbatch p5_tensor_gemm/run_tensor_gemm.sh

# Custom shapes:
DTYPE=fp8 MODE=custom GEMM_M=4096,8192 GEMM_N=4096,8192 GEMM_K=12288 \
  sbatch p5_tensor_gemm/run_tensor_gemm.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DTYPE` | `fp16` | `fp16`, `fp8`, `fp8_e4m3`, `fp4` (Blackwell only) |
| `MODE` | `smoke` | `smoke`, `saturation`, `transformer`, `custom` |
| `GEMM_M` | `8192` | Comma-separated M dimensions (custom mode) |
| `GEMM_N` | `8192` | Comma-separated N dimensions (custom mode) |
| `GEMM_K` | `8192` | Comma-separated K dimensions (custom mode) |
| `WARMUP` | `10` | Warmup iterations before timing |
| `ITERS` | `100` | Timed iterations |
| `BENCH_BIN_DIR` | `/tmp` | Where to cache the compiled binary |

Plus all shared variables from `common/config.sh` (partition, account, container path, etc.).

## How It Works

1. **Compilation**: On first run, `run_tensor_gemm.sh` compiles `tensor_gemm_bench.cu` inside the Singularity container (which has `nvcc` and cuBLAS). The binary is cached in `BENCH_BIN_DIR`.

2. **Execution**: The binary uses `cublasLtMatmul()` with heuristic algorithm selection to run dense GEMM at the specified precision.

3. **Output**: Per-shape results (M, N, K, time, TFLOPS) plus best-achieved throughput and percentage of theoretical peak.

## Sample Output

```
GPU: NVIDIA H100 80GB HBM3 (SM 90, 132 SMs, 79.1 GB, 1980 MHz)

dtype        M        N        K    time_ms     TFLOPS
──────────── ──────── ──────── ──────── ────────── ──────────
fp16             8192     8192     8192      1.128      974.3

Best achieved: 974.3 TFLOPS (fp16)
H100 SXM theoretical peak (fp16 dense): 989.4 TFLOPS
Achieved 98.5% of theoretical peak
```

## Interpreting Results

- **> 90% of peak**: Excellent — your GPU is well-saturated at this shape.
- **70–90% of peak**: Good — typical for medium shapes or when memory bandwidth starts to matter.
- **< 70% of peak**: The GEMM is likely too small or too skinny to saturate tensor cores. Try larger square matrices.
- **FP8 vs FP16**: FP8 should achieve roughly 2× the TFLOPS of FP16 at the same shape, because Hopper FP8 tensor cores have 2× the throughput of FP16 tensor cores.

## Building Manually

If you want to compile outside the SLURM flow:

```bash
singularity exec --nv /path/to/nvidia-hpc-benchmarks-25-04.sif \
  nvcc -O3 -std=c++17 \
  -gencode arch=compute_90,code=sm_90 \
  -lcublasLt -lcublas \
  -o tensor_gemm_bench tensor_gemm_bench.cu

# Then run directly:
./tensor_gemm_bench --dtype fp16 --m 8192,16384 --n 8192,16384 --k 8192,16384
```

## Benchmark Results (Kempner AI Cluster)

All results from single-GPU runs using cuBLASLt heuristic algorithm selection.

### H100 SXM 80GB (holygpu8a17604)

**GPU clocks: 1980 MHz**

| Precision | Best Shape (M×N×K) | Time (ms) | TFLOPS | % of Peak |
|-----------|-------------------|-----------|--------|-----------|
| FP16 | 16384³ | 9.478 | 928.4 | 93.8% |
| FP8 (E4M3) | 32768³ | 42.59 | 1654.8 | 83.6% |

### H200 SXM 141GB (holygpu8a10302)

**GPU clocks: 1980 MHz**

| Precision | Best Shape (M×N×K) | Time (ms) | TFLOPS | % of Peak |
|-----------|-------------------|-----------|--------|-----------|
| FP16 | 4096×4096×16384 | 0.621 | 885.5 | 89.5% |
| FP8 (E4M3) | 4096×4096×8192 | 0.172 | 1595.7 | 80.6% |
