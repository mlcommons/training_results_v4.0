#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=896
export LR=0.0005
export MINIBS=4

export TP=4
export PP=1
export CP=1
export SP=1
export TP_COMM_OVERLAP=True
export VBOOST_VALUE=1

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=32

export SKIP_EVALS=3
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=1
export WALLTIME_MINUTES=45
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
