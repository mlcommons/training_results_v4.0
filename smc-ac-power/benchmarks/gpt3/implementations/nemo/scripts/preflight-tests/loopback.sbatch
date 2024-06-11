#!/bin/bash

: "${CONTAINER:=/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/containers/dl+dgx+pytorch+24.03-py3-devel.sqsh}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CUMEM_ENABLE=1
export NCCL_DEBUG=warn
export UCX_LOG_LEVEL=warn
export UCX_TLS=rc
export NCCL_IB_SL=1
export NCCL_ALGO=ring
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

srun --mpi=pmix --ntasks-per-node 8 \
     --container-image="${CONTAINER}" \
     all_reduce_perf_mpi -b16G -e16G -g 1
