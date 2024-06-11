#!/bin/bash

set -x 
export NEXP=40
source config_XE8640x4H100SXM80GB.sh
export DGXSYSTEM=XE8640x4H100SXM80GB 
export CONT=6adf093327d0
#export DATADIR=/mnt/data/unet3d/ 
export DATADIR=/mnt/training_ds/unet/
export LOGDIR=results_XE8640x4H100SXM80GB 
./run_with_docker.sh

