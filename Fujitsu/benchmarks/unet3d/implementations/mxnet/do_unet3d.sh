#! /bin/bash

num_of_run=40

container_name=nvcr.io/nvdlfwea/mlperfv40/unet3d:20240416.mxnet
data_dir=/mnt/data4/work/3d-unet/data-dir
result_dir=$(realpath ../logs/logs-unet3d)

#source config_L40Sx8_1x8x5.sh
#source config_L40Sx16_1x16x4.sh
source config_L40Sx14_1x14x4.sh
#CONT=$container_name DATADIR=$data_dir LOGDIR=$result_dir ./run_with_dockerx8.sh

for idx in $(seq 1 $num_of_run); do
    CONT=$container_name DATADIR=$data_dir LOGDIR=$result_dir ./run_with_docker.sh
done
