#!/bin/bash
#SBATCH -p kempner_h100                 
#SBATCH --account=kempner_dev           
#SBATCH -N 1                            
#SBATCH --ntasks=2                      
#SBATCH --cpus-per-task=24              
#SBATCH --mem=375G                      
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100                    
#SBATCH --time=2:00:00                 
#SBATCH --job-name="HPL_H100_1N_2GPUs"           
#SBATCH --output=%x_%j.out 
#SBATCH --error=%x_%j.err

# Timestamp for logs
DATESTRING=$(date "+%Y-%m-%dT%H:%M:%S")
echo "Job started at: $DATESTRING"
echo "Running on hosts: $(scontrol show hostname)"

# Load required modules
module load intel/24.0.1-fasrc01
module load intelmpi/2021.11-fasrc01

# Singularity container image
SIF_DIR="/n/holylfs06/LABS/kempner_dev/Everyone/nvidia-hpl"
SIF_FILE="nvidia-hpc-benchmarks-25-04.sif"
CONT="${SIF_DIR}/${SIF_FILE}"

# Run the HPL benchmark using the sample .dat for 2 GPUs
srun --gres=gpu:2 \
     --ntasks=2 \
     --mpi=pmix \
     singularity exec --nv "${CONT}" \
     bash /workspace/hpl.sh \
     --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat

echo "Job finished at: $(date "+%Y-%m-%dT%H:%M:%S")"