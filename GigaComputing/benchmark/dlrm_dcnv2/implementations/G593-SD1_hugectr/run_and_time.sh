#!/bin/bash

cd ../hugectr
source config_G593-SD1_1x8x6912.sh
export CONT=mlperf_trainingv4.0-gigacomputing:dlrmv2
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
./run_with_docker.sh
