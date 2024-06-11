#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=1024
export MINIBS=1
export LR=0.0005

export TP=4
export PP=1
export SP=1
export CP=2
export TP_COMM_OVERLAP=True

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=True
export FP8_AMAX_HISTORY=32

export SKIP_EVALS=10

# system parameters
export DGXNNODES=64
export WALLTIME_MINUTES=15
export SBATCH_NETWORK=sharp
export SHARP=True
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
