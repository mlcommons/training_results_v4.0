#!/bin/bash

cd ../dgl
source config_G593-SD1_1x8x2048.sh
export CONT=mlperf_trainingv4.0-gigacomputing:gnn
export LOGDIR=/path/to/logfile
export DATA_DIR="/path/to/data/full"
export GRAPH_DIR="/path/to/graph/full"
./run_with_docker.sh
