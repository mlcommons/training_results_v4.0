#!/bin/bash
source config_D74H_7U.sh
DATE=$(date +"%Y%m%d_%H%M%S")
MLPERF_RULESET='4.0.0' NEXP=10 CONT=nvcr.io/nvdlfwea/mlperfv40/gnn:20240429.dgl DATA_DIR=/data/data/gnn/data/full GRAPH_DIR=/data/data/gnn/graph/full LOGDIR=/data/training_v4.0/gnn/logs/results ./run_with_docker.sh 2>&1 | tee ../logs/run_train-${DATE}.log
