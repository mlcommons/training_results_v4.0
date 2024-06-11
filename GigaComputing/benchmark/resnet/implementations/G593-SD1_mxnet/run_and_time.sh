#!/bin/bash

cd ../mxnet
source config_G593-SD1.sh
export CONT=mlperf_trainingv4.0-gigacomputing:resnet
export DATADIR=/path/to/preprocess/data
export LOGDIR=/path/to/logfile
./run_with_docker.sh
