#!/bin/bash

cd ../pytorch
source config_G593-SD1_01x08x64.sh
export CONT=./nvdlfwea+mlperfv40+sd.pytorch.sqsh
export DATADIR=/path/to/datasets
export CHECKPOINTS=/path/to/checkpoints
export LOGDIR=/path/to/folder
export NEMOLOGS=/path/to/folder
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G593-SD1

sbatch -N $DGXNNODES -t $WALLTIME run.sub
