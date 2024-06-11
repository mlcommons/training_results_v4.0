#!/bin/bash
set -x 

DGXNGPU=4 DGXSYSTEM=XE8640x4H100 CONT=2f718ee52465 DATADIR=/mnt/dlrmv2_ds/dlrmv2/criteo_1tb_multihot_raw LOGDIR=/mnt/training_v4.0_workingdir/20240419/scripts/dlrmv2/scripts/results_XE8640x4H100 ./run_with_docker.sh

