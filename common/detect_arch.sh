#!/usr/bin/env bash
# common/detect_arch.sh — GPU architecture detection and precision gating.
# Source this file from any launcher that needs arch-aware behaviour.
#
# Exports:
#   GPU_SM        — numeric SM version (e.g. 90, 100)
#   GPU_ARCH      — human-readable arch name (hopper, blackwell, rtx_blackwell, unknown)
#   GPU_NAME      — GPU product name from nvidia-smi
#
# Functions:
#   detect_gpu_arch           — populate the variables above
#   arch_supports_precision   — check if a precision is supported on the detected arch
#
# Usage:
#   source "$(dirname "$0")/../common/detect_arch.sh"
#   detect_gpu_arch
#   arch_supports_precision fp8  || { echo "FP8 not supported"; exit 1; }

# ── Detection ────────────────────────────────────────────────────────────────

detect_gpu_arch() {
    # Try nvidia-smi first (works on compute nodes)
    if command -v nvidia-smi &>/dev/null; then
        # Get compute capability (e.g. "9.0" for H100)
        local cc
        cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')
        # Convert "9.0" → 90, "10.0" → 100
        GPU_SM=$(echo "$cc" | awk -F. '{printf "%d", $1*10 + $2}')
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    else
        echo "WARNING: nvidia-smi not found. Falling back to manual GPU_SM=${GPU_SM:-0}." >&2
        GPU_SM="${GPU_SM:-0}"
        GPU_NAME="${GPU_NAME:-unknown}"
    fi

    # Map SM version to architecture name
    case "${GPU_SM}" in
        90)  GPU_ARCH="hopper"        ;;   # H100, H200
        100) GPU_ARCH="blackwell"     ;;   # B100, B200, GB200
        120) GPU_ARCH="rtx_blackwell" ;;   # RTX 5090, RTX 5080, etc.
        *)   GPU_ARCH="unknown"       ;;
    esac

    export GPU_SM GPU_ARCH GPU_NAME
}

# ── Precision gating ─────────────────────────────────────────────────────────
#
# Precision support matrix:
#   Arch            FP16    FP8     FP4
#   hopper (SM90)   ✓       ✓       ✗
#   blackwell       ✓       ✓       ✓
#   rtx_blackwell   ✓       ✓       ✓
#
# Returns 0 (true) if the precision is supported, 1 otherwise.

arch_supports_precision() {
    local precision="${1,,}"  # lowercase

    case "${GPU_ARCH}" in
        hopper)
            case "${precision}" in
                fp16|fp8) return 0 ;;
                fp4|nvfp4) return 1 ;;
                *) return 1 ;;
            esac
            ;;
        blackwell|rtx_blackwell)
            case "${precision}" in
                fp16|fp8|fp4|nvfp4) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *)
            echo "WARNING: Unknown GPU architecture '${GPU_ARCH}' (SM ${GPU_SM}). Cannot verify precision support." >&2
            return 1
            ;;
    esac
}

# ── Pretty-print ─────────────────────────────────────────────────────────────

print_gpu_info() {
    echo "────────────────────────────────────────────────────────"
    echo "  GPU Architecture Detection"
    echo "────────────────────────────────────────────────────────"
    echo "  GPU Name   : ${GPU_NAME}"
    echo "  SM Version : ${GPU_SM}"
    echo "  Arch Family: ${GPU_ARCH}"
    echo "  FP16 TC    : $(arch_supports_precision fp16 && echo '✓' || echo '✗')"
    echo "  FP8 TC     : $(arch_supports_precision fp8  && echo '✓' || echo '✗')"
    echo "  FP4 TC     : $(arch_supports_precision fp4  && echo '✓' || echo '✗')"
    echo "────────────────────────────────────────────────────────"
}
