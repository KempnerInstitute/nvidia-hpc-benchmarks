#!/usr/bin/env bash
# p4_nvidia_stream/run_stream.sh — Parameterized STREAM (memory bandwidth) launcher.
#
# ── Environment variables (or use defaults) ──────────────────────────────────
#   DEVICE           — GPU device index                (default: 0)
#   STREAM_N         — number of elements per array    (default: 1000000000)
#   STREAM_DTYPE     — fp32 or fp64                    (default: fp64)
#   STREAM_TESTS     — test selection string           (default: CSAT = all)
#
# ── Usage examples ───────────────────────────────────────────────────────────
#   sbatch p4_nvidia_stream/run_stream.sh
#   STREAM_DTYPE=fp32 sbatch p4_nvidia_stream/run_stream.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="STREAM_GPU_run"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/config.sh"

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
        --time="${BENCH_WALLTIME:-00:30:00}" \
        "$0"
fi

DEVICE="${DEVICE:-0}"
STREAM_N="${STREAM_N:-1000000000}"
STREAM_DTYPE="${STREAM_DTYPE:-fp64}"
STREAM_TESTS="${STREAM_TESTS:-CSAT}"

# STREAM uses less memory than HPL
BENCH_MEM="${BENCH_MEM:-250G}"
BENCH_WALLTIME="${BENCH_WALLTIME:-00:30:00}"

bench_load_modules
bench_job_header

echo ""
echo "STREAM (Memory Bandwidth) Benchmark"
echo "Device=${DEVICE}  N=${STREAM_N}  Dtype=${STREAM_DTYPE}  Tests=${STREAM_TESTS}"
echo ""

CONT="$(bench_container_path)"

DT_FLAG=""
if [[ "${STREAM_DTYPE,,}" == "fp32" ]]; then
    DT_FLAG="--dt fp32"
fi

srun --gres=gpu:1 \
     singularity exec --nv "${CONT}" \
     bash /workspace/stream-gpu-test.sh \
     --d "${DEVICE}" \
     --n "${STREAM_N}" \
     ${DT_FLAG} \
     --t "${STREAM_TESTS}"

echo ""
echo "Job finished at: $(bench_timestamp)"
