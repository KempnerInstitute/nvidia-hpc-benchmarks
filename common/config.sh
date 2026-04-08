#!/usr/bin/env bash
# common/config.sh — Shared configuration for all benchmark launchers.
# Source this file from any launcher script to pick up defaults.
# All values are env-var-overridable: set them before sourcing this file
# or export them in your shell / SLURM preamble.
#
# Usage:
#   source "$(dirname "$0")/../common/config.sh"

set -euo pipefail

# ── Cluster / SLURM defaults ────────────────────────────────────────────────
BENCH_PARTITION="${BENCH_PARTITION:-kempner_h100}"
BENCH_ACCOUNT="${BENCH_ACCOUNT:-kempner_dev}"
BENCH_CONSTRAINT="${BENCH_CONSTRAINT:-h100}"
BENCH_CPUS_PER_TASK="${BENCH_CPUS_PER_TASK:-24}"
BENCH_MEM="${BENCH_MEM:-375G}"
BENCH_WALLTIME="${BENCH_WALLTIME:-2:00:00}"
BENCH_MPI="${BENCH_MPI:-pmix}"

# ── Container ────────────────────────────────────────────────────────────────
BENCH_SIF_DIR="${BENCH_SIF_DIR:-/n/holylfs06/LABS/kempner_dev/Everyone/nvidia-hpl}"
BENCH_SIF_FILE="${BENCH_SIF_FILE:-nvidia-hpc-benchmarks-25-04.sif}"
BENCH_CONTAINER_TAG="${BENCH_CONTAINER_TAG:-25.04}"

# ── Modules (space-separated) ───────────────────────────────────────────────
BENCH_MODULES="${BENCH_MODULES:-intel/24.0.1-fasrc01 intelmpi/2021.11-fasrc01}"

# ── Derived ──────────────────────────────────────────────────────────────────
BENCH_CONT="${BENCH_SIF_DIR}/${BENCH_SIF_FILE}"

# ── Helper functions ─────────────────────────────────────────────────────────

bench_load_modules() {
    for mod in ${BENCH_MODULES}; do
        module load "$mod"
    done
}

bench_container_path() {
    if [[ ! -f "${BENCH_CONT}" ]]; then
        echo "ERROR: Container not found at ${BENCH_CONT}" >&2
        echo "Set BENCH_SIF_DIR and BENCH_SIF_FILE to point to a valid .sif image." >&2
        return 1
    fi
    echo "${BENCH_CONT}"
}

bench_print_config() {
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Benchmark Configuration"
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Partition        : ${BENCH_PARTITION}"
    echo "  Account          : ${BENCH_ACCOUNT}"
    echo "  Constraint       : ${BENCH_CONSTRAINT}"
    echo "  CPUs/task        : ${BENCH_CPUS_PER_TASK}"
    echo "  Memory           : ${BENCH_MEM}"
    echo "  Wall time        : ${BENCH_WALLTIME}"
    echo "  Container        : ${BENCH_CONT}"
    echo "  Container tag    : ${BENCH_CONTAINER_TAG}"
    echo "  Modules          : ${BENCH_MODULES}"
    echo "  MPI launcher     : ${BENCH_MPI}"
    echo "════════════════════════════════════════════════════════════════════"
}

bench_timestamp() {
    date "+%Y-%m-%dT%H:%M:%S"
}

bench_job_header() {
    echo "Job started at: $(bench_timestamp)"
    echo "Running on hosts: $(scontrol show hostname 2>/dev/null || hostname)"
    bench_print_config
}

# Compare semver strings: returns 0 if $1 >= $2
bench_version_ge() {
    local v1="$1" v2="$2"
    # Replace dots with spaces, compare major.minor
    local v1_major v1_minor v2_major v2_minor
    v1_major="${v1%%.*}"; v1_minor="${v1#*.}"; v1_minor="${v1_minor%%.*}"
    v2_major="${v2%%.*}"; v2_minor="${v2#*.}"; v2_minor="${v2_minor%%.*}"
    if (( v1_major > v2_major )); then return 0; fi
    if (( v1_major == v2_major && v1_minor >= v2_minor )); then return 0; fi
    return 1
}
