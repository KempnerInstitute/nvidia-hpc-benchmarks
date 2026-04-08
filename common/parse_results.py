#!/usr/bin/env python3
"""
common/parse_results.py — Parse benchmark output logs and produce summary tables.

Scans a directory tree for .out files, auto-detects the benchmark type from
content, extracts key metrics, and prints Markdown summary tables.

Usage:
    python3 common/parse_results.py [--dir <path>] [--arch h100]

Dependencies: Python 3.6+ standard library only (no pip packages).
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# ── Theoretical peaks (TFLOPS, dense) ────────────────────────────────────────
PEAKS = {
    "h100": {
        "fp64": 33.5,
        "fp16": 989.4,
        "fp8": 1978.9,
        "hpl_fp64": 33.5,       # HPL theoretical (FP64 only)
        "stream_bw_tb": 3.35,   # TB/s peak HBM bandwidth
    },
    "h200": {
        "fp64": 33.5,
        "fp16": 989.4,
        "fp8": 1978.9,
        "hpl_fp64": 33.5,
        "stream_bw_tb": 4.8,    # H200 has more HBM bandwidth
    },
}


# ── HPL parser ───────────────────────────────────────────────────────────────
def parse_hpl(content, filename):
    """Parse HPL (FP64 Linpack) output."""
    results = []
    # Look for the summary line: WC0  N  NB  P  Q  Time  Gflops (per GPU)
    pattern = re.compile(
        r"WC\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.e+]+)\s+\(\s*([\d.e+]+)\)"
    )
    for m in pattern.finditer(content):
        n, nb, p, q = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        time_s = float(m.group(5))
        gflops = float(m.group(6))
        gflops_per_gpu = float(m.group(7))
        total_gpus = p * q

        # Check residual
        passed = "PASSED" in content

        results.append({
            "benchmark": "HPL",
            "precision": "FP64",
            "N": n, "NB": nb, "P": p, "Q": q,
            "total_gpus": total_gpus,
            "time_s": time_s,
            "gflops": gflops,
            "gflops_per_gpu": gflops_per_gpu,
            "passed": passed,
            "source": filename,
        })
    return results


# ── HPL-MxP parser ───────────────────────────────────────────────────────────
def parse_hpl_mxp(content, filename):
    """Parse HPL-MxP (mixed-precision Linpack) output."""
    results = []

    # Extract sloppy-type
    st_match = re.search(r"--sloppy-type\s*=\s*(\d+)", content)
    sloppy_type = int(st_match.group(1)) if st_match else -1
    prec_map = {0: "FP64emu", 1: "FP8", 2: "FP16", 3: "FP4"}
    precision = prec_map.get(sloppy_type, f"unknown({sloppy_type})")

    # Extract N, NB, NPROW, NPCOL from the result block
    result_pattern = re.compile(
        r"N\s*=\s*(\d+),\s*NB\s*=\s*(\d+),\s*NPROW\s*=\s*(\d+),\s*NPCOL\s*=\s*(\d+)"
    )
    gflops_pattern = re.compile(
        r"GFLOPS\s*=\s*([\d.e+]+),\s*per GPU\s*=\s*([\d.]+)"
    )
    lu_gflops_pattern = re.compile(
        r"LU GFLOPS\s*=\s*([\d.e+]+),\s*per GPU\s*=\s*([\d.]+)"
    )
    passed_pattern = re.compile(r"PASSED")

    rm = result_pattern.search(content)
    if not rm:
        return results

    n, nb = int(rm.group(1)), int(rm.group(2))
    nprow, npcol = int(rm.group(3)), int(rm.group(4))
    total_gpus = nprow * npcol

    gm = gflops_pattern.search(content)
    lm = lu_gflops_pattern.search(content)
    passed = bool(passed_pattern.search(content))

    gflops = float(gm.group(1)) if gm else 0.0
    gflops_per_gpu = float(gm.group(2)) if gm else 0.0
    lu_gflops = float(lm.group(1)) if lm else 0.0
    lu_gflops_per_gpu = float(lm.group(2)) if lm else 0.0

    results.append({
        "benchmark": "HPL-MxP",
        "precision": precision,
        "N": n, "NB": nb,
        "nprow": nprow, "npcol": npcol,
        "total_gpus": total_gpus,
        "gflops": gflops,
        "gflops_per_gpu": gflops_per_gpu,
        "lu_gflops": lu_gflops,
        "lu_gflops_per_gpu": lu_gflops_per_gpu,
        "passed": passed,
        "source": filename,
    })
    return results


# ── HPCG parser ──────────────────────────────────────────────────────────────
def parse_hpcg(content, filename):
    """Parse HPCG output."""
    results = []
    # HPCG final result line
    gflops_match = re.search(r"Final Summary::HPCG result is VALID with a GFLOP/s rating of=\s*([\d.]+)", content)
    if not gflops_match:
        # Alternative pattern
        gflops_match = re.search(r"HPCG result is VALID.*?GFLOP/s.*?=\s*([\d.]+)", content)
    if not gflops_match:
        # Try the rating line
        gflops_match = re.search(r"rating of=\s*([\d.]+)", content)

    if gflops_match:
        gflops = float(gflops_match.group(1))
        # Get GPU count from filename or content
        # Filenames may have the bug where 2N_4GPUs means 2×4=8, not 4 total
        gpu_match = re.search(r"(\d+)GPU", filename)
        node_match = re.search(r"(\d+)N_", filename)
        gpus_per_node = int(gpu_match.group(1)) if gpu_match else 1
        nodes = int(node_match.group(1)) if node_match else 1
        total_gpus = nodes * gpus_per_node

        results.append({
            "benchmark": "HPCG",
            "precision": "Mixed (FP64+FP32)",
            "gflops": gflops,
            "total_gpus": total_gpus,
            "source": filename,
        })
    return results


# ── STREAM parser ────────────────────────────────────────────────────────────
def parse_stream(content, filename):
    """Parse STREAM GPU output."""
    results = []
    # STREAM output lines like: Copy:      3065453.4495   0.0026   0.0026   0.0026
    dtype = "FP32" if "fp32" in filename.lower() or "--dt fp32" in content else "FP64"
    pattern = re.compile(r"(Copy|Scale|Add|Triad|COPY|SCALE|ADD|TRIAD):?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    for m in pattern.finditer(content):
        func = m.group(1).upper()
        bw = float(m.group(2))
        results.append({
            "benchmark": "STREAM",
            "function": func,
            "precision": dtype,
            "bandwidth_mbs": bw,
            "source": filename,
        })
    return results


# ── Tensor GEMM parser ──────────────────────────────────────────────────────
def parse_tensor_gemm(content, filename):
    """Parse Tensor GEMM benchmark output."""
    results = []
    # Lines like: fp16     8192     8192     8192      1.128      974.3
    pattern = re.compile(
        r"(fp16|fp8_e4m3|fp8|fp4)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)"
    )
    for m in pattern.finditer(content):
        results.append({
            "benchmark": "TensorGEMM",
            "precision": m.group(1).upper(),
            "M": int(m.group(2)),
            "N": int(m.group(3)),
            "K": int(m.group(4)),
            "time_ms": float(m.group(5)),
            "tflops": float(m.group(6)),
            "source": filename,
        })

    # Best achieved line
    best_match = re.search(r"Best achieved:\s+([\d.]+)\s+TFLOPS\s+\((\w+)\)", content)
    if best_match and results:
        results[-1]["best_tflops"] = float(best_match.group(1))

    return results


# ── Auto-detect and dispatch ─────────────────────────────────────────────────
def detect_and_parse(content, filename):
    """Auto-detect benchmark type and parse."""
    if "HPL-MxP" in content or "hpl-mxp" in content or "HPL MxP" in content:
        return parse_hpl_mxp(content, filename)
    elif "HPLinpack" in content or "HPL-NVIDIA" in content:
        return parse_hpl(content, filename)
    elif "HPCG" in content and ("Conjugate" in content or "VALID" in content):
        return parse_hpcg(content, filename)
    elif "STREAM" in content and ("COPY" in content or "TRIAD" in content):
        return parse_stream(content, filename)
    elif "tensor_gemm" in filename.lower() or "TensorGEMM" in content or "TFLOPS" in content:
        return parse_tensor_gemm(content, filename)
    return []


# ── Table formatters ─────────────────────────────────────────────────────────
def print_hpl_table(results, arch):
    if not results:
        return
    print("\n## HPL (FP64 Linpack)\n")
    print("| GPUs | N | NB | P×Q | Time (s) | GFLOPS | Per-GPU GFLOPS | Passed |")
    print("|------|---|----|-----|----------|--------|----------------|--------|")
    for r in sorted(results, key=lambda x: x["total_gpus"]):
        print(f"| {r['total_gpus']} | {r['N']} | {r['NB']} | {r['P']}×{r['Q']} | "
              f"{r['time_s']:.2f} | {r['gflops']:.0f} | {r['gflops_per_gpu']:.0f} | "
              f"{'✓' if r['passed'] else '✗'} |")


def print_hpl_mxp_table(results, arch):
    if not results:
        return
    print("\n## HPL-MxP (Mixed-Precision Linpack)\n")
    print("| Precision | GPUs | N | NB | NPROW×NPCOL | GFLOPS | Per-GPU | LU GFLOPS | LU Per-GPU | Passed |")
    print("|-----------|------|---|----|-------------|--------|---------|-----------|------------|--------|")
    for r in sorted(results, key=lambda x: (x["precision"], x["total_gpus"])):
        print(f"| {r['precision']} | {r['total_gpus']} | {r['N']} | {r['NB']} | "
              f"{r['nprow']}×{r['npcol']} | {r['gflops']:.0f} | {r['gflops_per_gpu']:.0f} | "
              f"{r['lu_gflops']:.0f} | {r['lu_gflops_per_gpu']:.0f} | "
              f"{'✓' if r['passed'] else '✗'} |")


def print_hpcg_table(results, arch):
    if not results:
        return
    print("\n## HPCG (Sparse / System)\n")
    print("| GPUs | HPCG GFLOP/s |")
    print("|------|-------------|")
    for r in sorted(results, key=lambda x: x["total_gpus"]):
        print(f"| {r['total_gpus']} | {r['gflops']:.3f} |")


def print_stream_table(results, arch):
    if not results:
        return
    print("\n## STREAM (Memory Bandwidth)\n")
    peak_bw = PEAKS.get(arch, {}).get("stream_bw_tb", 0)
    peak_mbs = peak_bw * 1e6 if peak_bw else 0

    print("| Function | Precision | Bandwidth (MB/s) | % Peak |")
    print("|----------|-----------|------------------|--------|")
    for r in sorted(results, key=lambda x: (x["precision"], x["function"])):
        pct = (r["bandwidth_mbs"] / peak_mbs * 100) if peak_mbs else 0
        pct_str = f"{pct:.1f}%" if peak_mbs else "N/A"
        print(f"| {r['function']} | {r['precision']} | {r['bandwidth_mbs']:.1f} | {pct_str} |")


def print_tensor_gemm_table(results, arch):
    if not results:
        return
    print("\n## Tensor GEMM (Tensor-Core Saturation)\n")
    peaks = PEAKS.get(arch, {})

    print("| Precision | M | N | K | Time (ms) | TFLOPS | % Peak |")
    print("|-----------|---|---|---|-----------|--------|--------|")
    for r in sorted(results, key=lambda x: (x["precision"], -x["tflops"])):
        prec_key = r["precision"].lower().replace("_e4m3", "")
        peak = peaks.get(prec_key, 0)
        pct = (r["tflops"] / peak * 100) if peak else 0
        pct_str = f"{pct:.1f}%" if peak else "N/A"
        print(f"| {r['precision']} | {r['M']} | {r['N']} | {r['K']} | "
              f"{r['time_ms']:.3f} | {r['tflops']:.1f} | {pct_str} |")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Parse benchmark logs and produce summary tables.")
    parser.add_argument("--dir", default=".", help="Root directory to scan for .out files")
    parser.add_argument("--arch", default="h100", choices=list(PEAKS.keys()) + ["none"],
                        help="GPU architecture for peak comparisons (default: h100)")
    args = parser.parse_args()

    all_results = defaultdict(list)
    file_count = 0

    for root, dirs, files in os.walk(args.dir):
        for fname in sorted(files):
            if not fname.endswith(".out"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                print(f"WARNING: Could not read {fpath}: {e}", file=sys.stderr)
                continue

            results = detect_and_parse(content, fname)
            for r in results:
                all_results[r["benchmark"]].append(r)
            if results:
                file_count += 1

    if not all_results:
        print("No benchmark results found in .out files under:", args.dir)
        return

    arch = args.arch if args.arch != "none" else ""
    print(f"# Benchmark Results Summary")
    print(f"\nScanned {file_count} output files under `{args.dir}`")
    if arch:
        print(f"Architecture for peak comparisons: **{arch.upper()}**")

    print_hpl_table(all_results.get("HPL", []), arch)
    print_hpl_mxp_table(all_results.get("HPL-MxP", []), arch)
    print_hpcg_table(all_results.get("HPCG", []), arch)
    print_stream_table(all_results.get("STREAM", []), arch)
    print_tensor_gemm_table(all_results.get("TensorGEMM", []), arch)


if __name__ == "__main__":
    main()
