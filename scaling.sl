#!/bin/bash
#SBATCH -A LRN069
#SBATCH -J scaling
#SBATCH -o logs/scaling-%j.o
#SBATCH -e logs/scaling-%j.e
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 1

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.1.3
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

# Activate your environment
source activate ~/.conda/envs/set_qfit

# Get address of head node
export MASTER_ADDR=$(hostname -i)

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Run script
N=(1 2)
B=(16 16)
Q=(3 4)
srun -N1 -n8 -c1 --gpus-per-task=1 --gpu-bind=closest python3 -W ignore -u ./scaling_frontier.py 64 12 8 --master_addr=$MASTER_ADDR --master_port=3442
#srun -N1 -n2 -c1 --gpus-per-task=1 --gpu-bind=closest python3 -W ignore -u ./scaling_frontier.py 16 4 2 --master_addr=$MASTER_ADDR --master_port=3442
