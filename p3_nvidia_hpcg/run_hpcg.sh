#!/usr/bin/env bash
# p3_nvidia_hpcg/run_hpcg.sh — Parameterized HPCG launcher.
#
# ── Environment variables (or use defaults) ──────────────────────────────────
#   NODES            — number of SLURM nodes           (default: 1)
#   GPUS_PER_NODE    — GPUs per node                   (default: 4)
#   NX / NY / NZ     — local problem dimensions        (default: 256 each)
#   RT               — run time in seconds             (default: 1800 = 30 min)
#
# ── Usage examples ───────────────────────────────────────────────────────────
#   sbatch p3_nvidia_hpcg/run_hpcg.sh
#   NODES=2 GPUS_PER_NODE=4 RT=60 sbatch p3_nvidia_hpcg/run_hpcg.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="HPCG_run"
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
        --nodes="${NODES:-1}" \
        --ntasks="$(( ${NODES:-1} * ${GPUS_PER_NODE:-4} ))" \
        --cpus-per-task="${BENCH_CPUS_PER_TASK}" \
        --mem="${BENCH_MEM}" \
        --gres="gpu:${GPUS_PER_NODE:-4}" \
        --time="${BENCH_WALLTIME}" \
        "$0"
fi

NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NX="${NX:-256}"
NY="${NY:-256}"
NZ="${NZ:-256}"
RT="${RT:-1800}"

TOTAL_GPUS=$(( NODES * GPUS_PER_NODE ))
NTASKS="${TOTAL_GPUS}"

bench_load_modules
bench_job_header

echo ""
echo "HPCG (Sparse / System) Benchmark"
echo "Nodes=${NODES}  GPUs/node=${GPUS_PER_NODE}  Total GPUs=${TOTAL_GPUS}"
echo "NX=${NX}  NY=${NY}  NZ=${NZ}  RT=${RT}s"
echo ""

CONT="$(bench_container_path)"

srun --nodes="${NODES}" \
     --ntasks="${NTASKS}" \
     --ntasks-per-node="${GPUS_PER_NODE}" \
     --gpus-per-node="${GPUS_PER_NODE}" \
     --mpi="${BENCH_MPI}" \
     singularity exec --nv "${CONT}" \
     bash /workspace/hpcg.sh \
     --nx "${NX}" \
     --ny "${NY}" \
     --nz "${NZ}" \
     --rt "${RT}"

echo ""
echo "Job finished at: $(bench_timestamp)"
