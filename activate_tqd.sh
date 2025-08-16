#!/bin/bash

# Load modules
module reset
export ROCM_VER=6.4.0
module load PrgEnv-gnu/8.6.0
module load gcc-native/14.2
module load rocm/${ROCM_VER}
module load craype-accel-amd-gfx90a
module load cray-hdf5-parallel/1.12.2.11
module unload darshan-runtime
module load miniforge3

source activate /lustre/orion/proj-shared/lrn069/tqd_py310
