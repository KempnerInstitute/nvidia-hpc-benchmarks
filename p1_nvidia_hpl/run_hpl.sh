#!/usr/bin/env bash
# p1_nvidia_hpl/run_hpl.sh — Parameterized HPL (FP64 Linpack) launcher.
#
# This script can be submitted directly via sbatch OR sourced/called by a
# wrapper that sets the environment variables below.
#
# ── Required environment variables (or use defaults) ─────────────────────────
#   NODES            — number of SLURM nodes           (default: 1)
#   GPUS_PER_NODE    — GPUs per node                   (default: 4)
#   N                — matrix size                     (default: 190464)
#   NB               — block size                      (default: 1024)
#   P                — process-grid rows               (default: 2)
#   Q                — process-grid cols               (default: 2)
#   HPL_DAT          — path to a custom HPL.dat file   (default: auto-generate)
#
# ── Optional environment variables ───────────────────────────────────────────
#   BENCH_PARTITION, BENCH_ACCOUNT, BENCH_SIF_DIR, BENCH_SIF_FILE, ...
#   (see common/config.sh for the full list)
#
# ── Usage examples ───────────────────────────────────────────────────────────
#   # Quick 1-node 4-GPU run with defaults:
#   sbatch p1_nvidia_hpl/run_hpl.sh
#
#   # Custom 2-node 8-GPU run:
#   NODES=2 GPUS_PER_NODE=4 N=264192 P=4 Q=2 sbatch p1_nvidia_hpl/run_hpl.sh
#
#   # Use a hand-crafted HPL.dat:
#   HPL_DAT=/path/to/my/HPL.dat NODES=1 GPUS_PER_NODE=1 sbatch p1_nvidia_hpl/run_hpl.sh
#
#   # Sweep mode — run multiple (N, P, Q) combos sequentially:
#   HPL_SWEEP="92160:1:1 136192:2:1 190464:2:2" sbatch p1_nvidia_hpl/run_hpl.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name="HPL_run"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Load shared configuration ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/config.sh"

# ── Apply SBATCH overrides (these take effect only if not already on CLI) ────
# sbatch CLI flags override these; these are fallback defaults.
: "${SLURM_JOB_PARTITION:=}" # only set SBATCH if not already in a job
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Re-submit ourselves with proper SBATCH flags via environment
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

# ── HPL-specific defaults ────────────────────────────────────────────────────
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
N="${N:-190464}"
NB="${NB:-1024}"
P="${P:-2}"
Q="${Q:-2}"
HPL_DAT="${HPL_DAT:-}"
HPL_SWEEP="${HPL_SWEEP:-}"

TOTAL_GPUS=$(( NODES * GPUS_PER_NODE ))
NTASKS="${TOTAL_GPUS}"

# ── Generate HPL.dat inline ──────────────────────────────────────────────────
generate_hpl_dat() {
    local n="$1" nb="$2" p="$3" q="$4" outfile="$5"
    cat > "${outfile}" <<EOF
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
${n}         Ns
1            # of NBs
${nb}        NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
${p}         Ps
${q}         Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criteria
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
EOF
}

# ── Run a single HPL instance ────────────────────────────────────────────────
run_hpl_single() {
    local n="$1" nb="$2" p="$3" q="$4"

    local dat_arg
    if [[ -n "${HPL_DAT}" ]]; then
        dat_arg="--dat ${HPL_DAT}"
        echo "Using user-supplied HPL.dat: ${HPL_DAT}"
    else
        local tmpdat
        tmpdat="$(mktemp /tmp/HPL-XXXXXX.dat)"
        generate_hpl_dat "${n}" "${nb}" "${p}" "${q}" "${tmpdat}"
        dat_arg="--dat ${tmpdat}"
        echo "Generated HPL.dat: N=${n}, NB=${nb}, P=${p}, Q=${q}"
    fi

    local cont
    cont="$(bench_container_path)"

    echo ""
    echo "── HPL Run: N=${n} NB=${nb} P=${p} Q=${q} GPUs=${TOTAL_GPUS} ──"

    srun --nodes="${NODES}" \
         --ntasks="${NTASKS}" \
         --ntasks-per-node="${GPUS_PER_NODE}" \
         --gpus-per-node="${GPUS_PER_NODE}" \
         --mpi="${BENCH_MPI}" \
         singularity exec --nv "${cont}" \
         bash /workspace/hpl.sh \
         ${dat_arg}

    # Clean up tmpdat if we generated one
    if [[ -z "${HPL_DAT}" ]] && [[ -f "${tmpdat:-}" ]]; then
        rm -f "${tmpdat}"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
bench_load_modules
bench_job_header

echo ""
echo "HPL (FP64 Linpack) Benchmark"
echo "Nodes=${NODES}  GPUs/node=${GPUS_PER_NODE}  Total GPUs=${TOTAL_GPUS}"
echo ""

if [[ -n "${HPL_SWEEP}" ]]; then
    # Sweep mode: HPL_SWEEP="N1:P1:Q1 N2:P2:Q2 ..."
    echo "Sweep mode: ${HPL_SWEEP}"
    for combo in ${HPL_SWEEP}; do
        IFS=':' read -r sw_n sw_p sw_q <<< "${combo}"
        run_hpl_single "${sw_n}" "${NB}" "${sw_p}" "${sw_q}"
    done
else
    run_hpl_single "${N}" "${NB}" "${P}" "${Q}"
fi

echo ""
echo "Job finished at: $(bench_timestamp)"
