#!/usr/bin/env bash
# p5_tensor_gemm/run_tensor_gemm.sh — Tensor-core GEMM saturation benchmark launcher.
#
# This benchmark answers: "How close can I get to peak FP16 / FP8 tensor-core
# throughput on my GPU?" — it is NOT a Linpack solve. For Linpack-like
# mixed-precision benchmarking, use HPL-MxP (p2).
#
# ── Modes ────────────────────────────────────────────────────────────────────
#   MODE=smoke        Quick test: single large square GEMM (8192³)
#   MODE=saturation   Full sweep: multiple sizes to find near-peak throughput
#   MODE=transformer  ML-relevant shapes (attention/FFN projections)
#   MODE=custom       Use GEMM_M, GEMM_N, GEMM_K directly
#
# ── Environment variables ────────────────────────────────────────────────────
#   DTYPE            — fp16, fp8, fp4            (default: fp16)
#   MODE             — smoke, saturation, transformer, custom  (default: smoke)
#   GEMM_M           — comma-separated M dims    (for custom mode)
#   GEMM_N           — comma-separated N dims    (for custom mode)
#   GEMM_K           — comma-separated K dims    (for custom mode)
#   WARMUP           — warmup iterations          (default: 10)
#   ITERS            — timed iterations            (default: 100)
#   BENCH_BIN_DIR    — where to cache the compiled binary (default: /tmp)
#
# ── Usage examples ───────────────────────────────────────────────────────────
#   # FP16 smoke test:
#   sbatch p5_tensor_gemm/run_tensor_gemm.sh
#
#   # FP8 saturation sweep:
#   DTYPE=fp8 MODE=saturation sbatch p5_tensor_gemm/run_tensor_gemm.sh
#
#   # Custom shapes:
#   DTYPE=fp16 MODE=custom GEMM_M=4096,8192 GEMM_N=4096,8192 GEMM_K=12288 \
#     sbatch p5_tensor_gemm/run_tensor_gemm.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="TensorGEMM_run"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/config.sh"
source "${SCRIPT_DIR}/../common/detect_arch.sh"

# ── Apply SBATCH overrides (auto-resubmit with proper flags) ─────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch \
        --export=ALL \
        --partition="${BENCH_PARTITION}" \
        --account="${BENCH_ACCOUNT}" \
        ${BENCH_CONSTRAINT:+--constraint="${BENCH_CONSTRAINT}"} \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${BENCH_CPUS_PER_TASK}" \
        --mem="${BENCH_MEM:-250G}" \
        --gres=gpu:1 \
        --time="${BENCH_WALLTIME:-01:00:00}" \
        "$0"
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
DTYPE="${DTYPE:-fp16}"
MODE="${MODE:-smoke}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-100}"
BENCH_BIN_DIR="${BENCH_BIN_DIR:-/tmp}"

# Custom mode dims
GEMM_M="${GEMM_M:-8192}"
GEMM_N="${GEMM_N:-8192}"
GEMM_K="${GEMM_K:-8192}"

BENCH_WALLTIME="${BENCH_WALLTIME:-01:00:00}"

# ── Source file ──────────────────────────────────────────────────────────────
SRC="${SCRIPT_DIR}/tensor_gemm_bench.cu"
BIN="${BENCH_BIN_DIR}/tensor_gemm_bench"

# ── Main ─────────────────────────────────────────────────────────────────────
bench_load_modules
bench_job_header

detect_gpu_arch 2>/dev/null || true
print_gpu_info 2>/dev/null || true

# ── Precision gating ─────────────────────────────────────────────────────────
DTYPE_LOWER="${DTYPE,,}"
if [[ "${DTYPE_LOWER}" == "fp4" || "${DTYPE_LOWER}" == "nvfp4" ]]; then
    if [[ -n "${GPU_ARCH:-}" ]] && ! arch_supports_precision fp4; then
        echo "ERROR: FP4 is NOT supported on ${GPU_ARCH} (SM ${GPU_SM})." >&2
        echo "  FP4 tensor-core operations require Blackwell-class GPUs (SM ≥ 100)." >&2
        echo "  H100/H200 (Hopper, SM 90) support FP16 and FP8 only." >&2
        exit 1
    fi
fi

# ── Compile if needed ────────────────────────────────────────────────────────
CONT="$(bench_container_path)"

compile_bench() {
    echo "Compiling tensor_gemm_bench.cu ..."
    singularity exec --nv "${CONT}" \
        nvcc -O3 -std=c++17 \
        -gencode arch=compute_90,code=sm_90 \
        -lcublasLt -lcublas \
        -o "${BIN}" "${SRC}"
    echo "Binary cached at: ${BIN}"
}

if [[ ! -x "${BIN}" ]] || [[ "${SRC}" -nt "${BIN}" ]]; then
    compile_bench
else
    echo "Using cached binary: ${BIN}"
fi

# ── Define shapes per mode ───────────────────────────────────────────────────
case "${MODE}" in
    smoke)
        M_LIST="8192"
        N_LIST="8192"
        K_LIST="8192"
        echo "Mode: smoke — quick single-shape test"
        ;;
    saturation)
        M_LIST="4096,8192,16384,32768"
        N_LIST="4096,8192,16384,32768"
        K_LIST="4096,8192,16384,32768"
        echo "Mode: saturation — sweeping square GEMMs to find peak"
        ;;
    transformer)
        # Representative transformer shapes:
        #   QKV projection:     batch*seq  × hidden × 3*hidden   e.g. 4096 × 4096 × 12288
        #   FFN up-project:     batch*seq  × hidden × 4*hidden   e.g. 4096 × 4096 × 16384
        #   Attention matmul:   batch*heads × seq × seq           e.g. 4096 × 4096 × 4096
        #   Large FFN:          batch*seq  × hidden × ffn_dim     e.g. 2048 × 12288 × 49152
        M_LIST="4096,4096,4096,2048,8192"
        N_LIST="4096,4096,4096,12288,8192"
        K_LIST="12288,16384,4096,49152,8192"
        echo "Mode: transformer — ML-relevant projection / attention shapes"
        ;;
    custom)
        M_LIST="${GEMM_M}"
        N_LIST="${GEMM_N}"
        K_LIST="${GEMM_K}"
        echo "Mode: custom — user-specified shapes"
        ;;
    *)
        echo "ERROR: Unknown MODE '${MODE}'. Use: smoke, saturation, transformer, custom" >&2
        exit 1
        ;;
esac

echo "Dtype: ${DTYPE_LOWER}"
echo "Warmup: ${WARMUP}  Iters: ${ITERS}"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────────
# For saturation mode, we run all combos (cartesian product via the binary).
# For transformer mode, we want paired shapes, so run each pair separately.

if [[ "${MODE}" == "transformer" ]]; then
    # Paired shapes — split into arrays and iterate
    IFS=',' read -ra M_ARR <<< "${M_LIST}"
    IFS=',' read -ra N_ARR <<< "${N_LIST}"
    IFS=',' read -ra K_ARR <<< "${K_LIST}"

    for i in "${!M_ARR[@]}"; do
        singularity exec --nv "${CONT}" \
            "${BIN}" \
            --dtype "${DTYPE_LOWER}" \
            --m "${M_ARR[$i]}" \
            --n "${N_ARR[$i]}" \
            --k "${K_ARR[$i]}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}"
        echo ""
    done
else
    singularity exec --nv "${CONT}" \
        "${BIN}" \
        --dtype "${DTYPE_LOWER}" \
        --m "${M_LIST}" \
        --n "${N_LIST}" \
        --k "${K_LIST}" \
        --warmup "${WARMUP}" \
        --iters "${ITERS}"
fi

echo ""
echo "Job finished at: $(bench_timestamp)"
