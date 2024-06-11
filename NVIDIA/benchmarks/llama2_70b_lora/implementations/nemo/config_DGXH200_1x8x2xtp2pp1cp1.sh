#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=1024
export LR=0.0005
export MINIBS=2

export TP=2
export PP=1
export SP=1
export CP=1

export SKIP_EVALS=3
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export VBOOST_VALUE=1

export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=True 

export TP_COMM_OVERLAP=True
export NCCL_MIN_CTAS=32

export UCX_TLS=self,tcp

# system parameters
export DGXNNODES=1
export WALLTIME_MINUTES=50
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
