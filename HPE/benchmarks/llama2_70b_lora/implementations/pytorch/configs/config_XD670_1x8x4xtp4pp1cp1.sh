#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_XD670_common.sh
export SUBMISSION_ORG=HPE
export MLPERF_SUBMITTER=HPE
export MLPERF_SYSTEM_NAME="HPE Cray XD670"
export MLPERF_STATUS=onprem
export MLPERF_DIVISION=closed
export MLPERF_NUM_NODES=1

# hyperparameters
export MAX_STEPS=1024
export MINIBS=4 #

export TP=4
export PP=1
export CP=1 #
export SP=1
export TP_COMM_OVERLAP=True

export FP8=True
export FP8_AMAX_ALGO=max #most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4

export SKIP_EVALS=3
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=1
export WALLTIME_MINUTES=45
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
