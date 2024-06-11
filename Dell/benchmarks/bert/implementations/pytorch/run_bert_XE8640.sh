#!/bin/bash

set -x 


source config_XE8640x4H100_SXM_80GB.sh

export NEXP=10
#export NEXP=12
export CUDA_VISIBLE_DEVICES=0,1,2,3

export LOGDIR=/mnt/training_v4.0_workingdir/20240419/scripts/bert/results_XE8640x4H100
DGXSYSTEM=XE8640x4H100_SXM_80GB CONT=025851af44e9   ./run_with_docker.sh


