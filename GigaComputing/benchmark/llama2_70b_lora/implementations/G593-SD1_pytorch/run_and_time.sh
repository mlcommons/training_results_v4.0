#!/bin/bash

cd ../pytorch
source configs/config_DGXH100_1x8x4xtp4pp1cp1.sh
export CONT=./nvdlfwea+mlperfv40+lora.pytorch.sqsh
export DATADIR=/path/to/datasets
export MODEL=/path/to/model
export LOGDIR=/path/to/folder
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G593-SD1

sbatch -N $DGXNNODES -t $WALLTIME run.sub
