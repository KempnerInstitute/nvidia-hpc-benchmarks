#!/bin/bash
#SBATCH -p kempner_h100                 
#SBATCH --account=kempner_dev           
#SBATCH -N 1                            
#SBATCH --ntasks=4                      
#SBATCH --cpus-per-task=24              
#SBATCH --mem=375G                      
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100                    
#SBATCH --time=2:00:00                 
#SBATCH --job-name="HPL_MxP_H100_1N_4GPUs"           
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

# Input parameters for HPL-MxP
NGPUS=4
N=190464
NB=1024
NPROW=2
NPCOL=2
NPORDER="row"
GPU_AFFINITY=0:1:2:3
NTASKS=4


# Run the HPL benchmark using the sample .dat for 1 GPU
srun --gres=gpu:${NGPUS} \
     --ntasks=${NTASKS} \
     --mpi=pmix \
     singularity exec --nv "${CONT}" \
     bash /workspace/hpl-mxp.sh \
     --n ${N} \
     --nb ${NB} \
     --nprow ${NPROW} \
     --npcol ${NPCOL} \
     --nporder ${NPORDER} \
     --gpu-affinity ${GPU_AFFINITY} 

echo "Job finished at: $(date "+%Y-%m-%dT%H:%M:%S")"