#!/bin/bash
set -x 
source config_XE8640x4H100.sh

export DATADIR=/mnt/training_ds/openimages_ds/open-images-v6-mlperf
export LOGDIR=`pwd`/results_XE8640x4xH100
CONT=6ea5a8c3bd6f ./run_with_docker.sh
