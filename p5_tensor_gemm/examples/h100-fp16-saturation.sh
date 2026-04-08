#!/usr/bin/env bash
# Example: FP16 tensor-core saturation sweep on H100 (1 GPU)
#SBATCH -p kempner_h100
#SBATCH --account=kempner_dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=01:00:00
#SBATCH --job-name="TensorGEMM_FP16_saturation"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export DTYPE=fp16
export MODE=saturation
export WARMUP=10
export ITERS=100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_tensor_gemm.sh"
