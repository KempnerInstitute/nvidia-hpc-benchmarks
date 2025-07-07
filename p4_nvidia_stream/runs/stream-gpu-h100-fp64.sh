#!/bin/bash
#SBATCH -p kempner_eng
#SBATCH --account=kempner_dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=00:30:00
#SBATCH --job-name="STREAM_GPU_H100_FP64"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Timestamp for logs
echo "Job started at: $(date "+%Y-%m-%dT%H:%M:%S")"
echo "Running on hosts: $(scontrol show hostname)"

# Load modules if needed
module load intel/24.0.1-fasrc01
module load intelmpi/2021.11-fasrc01

# Singularity image
SIF_DIR="/n/holylfs06/LABS/kempner_dev/Everyone/nvidia-hpl"
SIF_FILE="nvidia-hpc-benchmarks-25-04.sif"
CONT="${SIF_DIR}/${SIF_FILE}"

# STREAM benchmark parameters
DEVICE=0
N=1000000000  # Number of elements per array
DTYPE=""       # Leave empty for double precision; use "--dt fp32" for FP32
TESTS="CSAT"   # COPY, SCALE, ADD, TRIAD

# Run STREAM benchmark
srun --gres=gpu:1 \
     singularity exec --nv "${CONT}" \
     bash /workspace/stream-gpu-test.sh \
     --d ${DEVICE} \
     --n ${N} \
     ${DTYPE} \
     --t ${TESTS}

echo "Job finished at: $(date "+%Y-%m-%dT%H:%M:%S")"
