#!/bin/bash

num_of_run=10

source config_L40Sx8_v1b-fl.sh

logdir=$(realpath -m ../logs/logs-resnet)

# start benchmark process
for idx in $(seq 1 $num_of_run); do
    NEXP=1 CONT=nvcr.io/nvdlfwea/mlperfv40/resnet:20240419.mxnet DATADIR=${DATADIR}  \
        LOGDIR=$logdir PULL=0 ./run_with_dockerx8.sh
done

