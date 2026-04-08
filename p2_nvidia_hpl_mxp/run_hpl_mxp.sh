#!/usr/bin/env bash
# p2_nvidia_hpl_mxp/run_hpl_mxp.sh — Parameterized HPL-MxP (mixed-precision Linpack) launcher.
#
# HPL-MxP performs LU factorization in low precision (FP16 / FP8 / FP4) and
# refines the solution to FP64 accuracy via iterative refinement (GMRES).
# This is a Linpack-like benchmark, NOT a raw tensor-core saturation test.
#
# ── Precision modes ──────────────────────────────────────────────────────────
#   SLOPPY_TYPE   Container flag   Arch requirement
#   FP16          --sloppy-type 2  Hopper (SM ≥ 90)
#   FP8           --sloppy-type 1  Hopper (SM ≥ 90)
#   FP4           --sloppy-type 3  Blackwell (SM ≥ 100) + container ≥ 25.06
#
# ── Required environment variables (or use defaults) ─────────────────────────
#   NODES            — number of SLURM nodes           (default: 1)
#   GPUS_PER_NODE    — GPUs per node                   (default: 4)
#   N                — matrix size                     (default: 190464)
#   NB               — block size                      (default: 1024)
#   NPROW            — process-grid rows               (default: 2)
#   NPCOL            — process-grid cols               (default: 2)
#   SLOPPY_TYPE      — precision: FP16, FP8, or FP4   (default: FP16)
#   NPORDER          — process ordering                (default: row)
#   GPU_AFFINITY     — colon-separated GPU IDs         (default: auto)
#
# ── Optional ─────────────────────────────────────────────────────────────────
#   CPU_AFFINITY     — CPU affinity string for hpl-mxp.sh
#   MEM_AFFINITY     — memory affinity string
#   UCX_AFFINITY     — UCX network device affinity
#
# ── Usage examples ───────────────────────────────────────────────────────────
#   # FP16 on 1 node, 4 GPUs (default):
#   sbatch p2_nvidia_hpl_mxp/run_hpl_mxp.sh
#
#   # FP8 on 1 node, 4 GPUs:
#   SLOPPY_TYPE=FP8 sbatch p2_nvidia_hpl_mxp/run_hpl_mxp.sh
#
#   # FP16 on 2 nodes, 8 GPUs total:
#   NODES=2 GPUS_PER_NODE=4 N=264192 NPROW=4 NPCOL=2 sbatch p2_nvidia_hpl_mxp/run_hpl_mxp.sh
#
#   # FP8 on 1 node, 1 GPU:
#   NODES=1 GPUS_PER_NODE=1 N=92160 NPROW=1 NPCOL=1 SLOPPY_TYPE=FP8 \
#     sbatch p2_nvidia_hpl_mxp/run_hpl_mxp.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="HPL_MxP_run"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Load shared configuration ────────────────────────────────────────────────
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
        --nodes="${NODES:-1}" \
        --ntasks="$(( ${NODES:-1} * ${GPUS_PER_NODE:-4} ))" \
        --cpus-per-task="${BENCH_CPUS_PER_TASK}" \
        --mem="${BENCH_MEM}" \
        --gres="gpu:${GPUS_PER_NODE:-4}" \
        --time="${BENCH_WALLTIME}" \
        "$0"
fi

# ── HPL-MxP-specific defaults ───────────────────────────────────────────────
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
N="${N:-190464}"
NB="${NB:-1024}"
NPROW="${NPROW:-2}"
NPCOL="${NPCOL:-2}"
SLOPPY_TYPE="${SLOPPY_TYPE:-FP16}"
NPORDER="${NPORDER:-row}"

# Optional affinity (empty = let hpl-mxp.sh pick defaults)
GPU_AFFINITY="${GPU_AFFINITY:-}"
CPU_AFFINITY="${CPU_AFFINITY:-}"
MEM_AFFINITY="${MEM_AFFINITY:-}"
UCX_AFFINITY="${UCX_AFFINITY:-}"

TOTAL_GPUS=$(( NODES * GPUS_PER_NODE ))
NTASKS="${TOTAL_GPUS}"

# Minimum container version required for FP4 support
FP4_MIN_CONTAINER="25.06"

# ── Map user-friendly precision name to --sloppy-type numeric value ──────────
map_sloppy_type() {
    local prec="${1^^}"  # uppercase
    case "${prec}" in
        FP16)  echo 2 ;;
        FP8)   echo 1 ;;
        FP4)   echo 3 ;;
        *)
            echo "ERROR: Unknown SLOPPY_TYPE '${1}'." >&2
            echo "  Supported values: FP16, FP8, FP4" >&2
            exit 1
            ;;
    esac
}

# ── Auto-generate GPU affinity if not provided ──────────────────────────────
auto_gpu_affinity() {
    local gpus_per_node="$1" nodes="$2"
    local aff=""
    for ((node=0; node<nodes; node++)); do
        for ((g=0; g<gpus_per_node; g++)); do
            if [[ -n "${aff}" ]]; then aff+=":"; fi
            aff+="${g}"
        done
    done
    echo "${aff}"
}

# ── Main ─────────────────────────────────────────────────────────────────────
bench_load_modules
bench_job_header

# Detect architecture (will work on compute nodes; warns on login nodes)
detect_gpu_arch 2>/dev/null || true
print_gpu_info 2>/dev/null || true

# Resolve sloppy-type
SLOPPY_NUM=$(map_sloppy_type "${SLOPPY_TYPE}")
echo ""
echo "HPL-MxP (Mixed-Precision Linpack) Benchmark"
echo "Precision      : ${SLOPPY_TYPE} (--sloppy-type ${SLOPPY_NUM})"
echo "Nodes          : ${NODES}"
echo "GPUs/node      : ${GPUS_PER_NODE}"
echo "Total GPUs     : ${TOTAL_GPUS}"
echo "N=${N}  NB=${NB}  NPROW=${NPROW}  NPCOL=${NPCOL}  NPORDER=${NPORDER}"
echo ""

# ── Precision gating ─────────────────────────────────────────────────────────
PREC_LOWER="${SLOPPY_TYPE,,}"

if [[ "${PREC_LOWER}" == "fp4" ]]; then
    # Gate 1: Architecture must support FP4
    if [[ -n "${GPU_ARCH:-}" ]] && ! arch_supports_precision fp4; then
        echo "ERROR: FP4 is NOT supported on ${GPU_ARCH} (SM ${GPU_SM})." >&2
        echo "  FP4 tensor-core operations require Blackwell-class GPUs (SM ≥ 100)." >&2
        echo "  H100/H200 (Hopper, SM 90) support FP16 and FP8 only." >&2
        exit 1
    fi

    # Gate 2: Container version must be recent enough
    if ! bench_version_ge "${BENCH_CONTAINER_TAG}" "${FP4_MIN_CONTAINER}"; then
        echo "ERROR: FP4 HPL-MxP requires container version ≥ ${FP4_MIN_CONTAINER}." >&2
        echo "  Current container tag: ${BENCH_CONTAINER_TAG}" >&2
        echo "  Update BENCH_SIF_FILE and BENCH_CONTAINER_TAG to a newer image." >&2
        exit 1
    fi

    echo "NOTE: FP4 mode enabled. Ensure your container (${BENCH_CONTAINER_TAG}) actually supports --sloppy-type 3."
fi

# ── Build GPU affinity string ────────────────────────────────────────────────
if [[ -z "${GPU_AFFINITY}" ]]; then
    GPU_AFFINITY=$(auto_gpu_affinity "${GPUS_PER_NODE}" "${NODES}")
fi

# ── Construct optional arguments ─────────────────────────────────────────────
EXTRA_ARGS=""
if [[ -n "${CPU_AFFINITY}" ]]; then
    EXTRA_ARGS+=" --cpu-affinity ${CPU_AFFINITY}"
fi
if [[ -n "${MEM_AFFINITY}" ]]; then
    EXTRA_ARGS+=" --mem-affinity ${MEM_AFFINITY}"
fi
if [[ -n "${UCX_AFFINITY}" ]]; then
    EXTRA_ARGS+=" --ucx-affinity ${UCX_AFFINITY}"
fi

# ── Run ──────────────────────────────────────────────────────────────────────
CONT="$(bench_container_path)"

srun --nodes="${NODES}" \
     --ntasks="${NTASKS}" \
     --ntasks-per-node="${GPUS_PER_NODE}" \
     --gpus-per-node="${GPUS_PER_NODE}" \
     --mpi="${BENCH_MPI}" \
     singularity exec --nv "${CONT}" \
     bash /workspace/hpl-mxp.sh \
     --n "${N}" \
     --nb "${NB}" \
     --nprow "${NPROW}" \
     --npcol "${NPCOL}" \
     --nporder "${NPORDER}" \
     --gpu-affinity "${GPU_AFFINITY}" \
     --sloppy-type "${SLOPPY_NUM}" \
     ${EXTRA_ARGS}

echo ""
echo "Job finished at: $(bench_timestamp)"
