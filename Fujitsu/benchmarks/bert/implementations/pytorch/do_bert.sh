#! /bin/bash

num_of_run=10

source config_L40Sx16.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# The bellow three variables are set by the above config
#export DATADIR_PHASE2=/mnt/data4/work/bert_data/packed_data
#export EVALDIR=/mnt/data4/work/bert_data/hdf5/eval_varlength
#export CHECKPOINTDIR_PHASE1=/mnt/data4/work/bert_data/phase1

container=nvcr.io/nvdlfwea/mlperfv40/bert:20230926.pytorch
logdir=$(realpath ../logs/logs-bert)

for idx in $(seq 1 $num_of_run); do
    CONT=$container LOGDIR=$logdir ./run_with_docker.sh
done
