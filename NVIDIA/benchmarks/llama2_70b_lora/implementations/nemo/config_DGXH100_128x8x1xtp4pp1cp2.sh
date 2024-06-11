#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export LR=0.0006
export MAX_STEPS=1024
export MINIBS=1 

export TP=4
export PP=1
export CP=2
export SP=1
export TP_COMM_OVERLAP=True

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=True
export FP8_AMAX_HISTORY=32

export SKIP_EVALS=18
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=128
export WALLTIME_MINUTES=15
export SBATCH_NETWORK=sharp
export SHARP=True
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
