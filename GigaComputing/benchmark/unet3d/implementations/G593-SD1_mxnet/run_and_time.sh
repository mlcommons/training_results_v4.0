#!/bin/bash

cd ../mxnet
source config_G593-SD1_1x8x7.sh
export CONT=mlperf_trainingv4.0-gigacomputing:unet3d
export DATADIR=/path/to/dataset
export LOGDIR=/path/to/results
./run_with_docker.sh
