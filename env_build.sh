#!/bin/bash

set -e

#### THIS ASSUMES YOU DON'T ALREADY HAVE MINIFORGE/CONDA LOADED -- DO NOT LOAD OUTSIDE OF THIS BEFOREHAND ####
#### OR, IF ALREADY LOADED, MODIFY BEFOREHAND (comment out loading the miniforge module) ####

# optional first step:
# pip cache purge

# Load modules
module reset
export ROCM_VER=6.4.0
module load PrgEnv-gnu/8.6.0
module load gcc-native/14.2
module load rocm/${ROCM_VER}
module load craype-accel-amd-gfx90a
module load cmake
module load cray-hdf5-parallel/1.12.2.11
module load miniforge3
module unload darshan-runtime

# Setup ROCm paths (sometimes needed for building)
export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-${ROCM_VER}

# Create initial conda env with common dependencies
conda create -p /lustre/orion/proj-shared/lrn069/tqd_py310 python=3.10 numpy=1.26.3 'scipy<1.13.0' matplotlib seaborn -c conda-forge
source activate /lustre/orion/proj-shared/lrn069/tqd_py310

# More initial dependencies via pip
pip install pyyaml typing_extensions ninja packaging pytest einops

# Install mpi4py
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

# Install parallel h5py
HDF5_MPI="ON" CC=cc HDF5_DIR=${HDF5_ROOT} pip install --no-cache-dir --no-binary=h5py h5py

# Install torch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4

#### Installing torchquantum_dist Process ####

pip install .

########
echo 'build script complete, you should run: source activate_tqd_env.sh'
