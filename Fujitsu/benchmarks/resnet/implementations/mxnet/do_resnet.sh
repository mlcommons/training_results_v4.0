#!/bin/bash

num_of_run=5

source config_GX2560M7x4H100_SXM_80GB.sh

logdir=$(realpath -m ../logs/logs-resnet)

# start benchmark process
for idx in $(seq 1 $num_of_run); do
    NEXP=1 CONT=nvcr.io/nvdlfwea/mlperfv40/resnet:20240419.mxnet DATADIR=${DATADIR}  \
        LOGDIR=$logdir PULL=0 ./run_with_docker.sh
done

