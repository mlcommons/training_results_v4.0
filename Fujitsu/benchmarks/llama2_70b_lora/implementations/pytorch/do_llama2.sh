#! /bin/bash

container=nvcr.io/nvdlfwea/mlperfv40/lora:20240427.pytorch
datadir=/mnt/data4/work/llama2/dataset/scrolls_gov_report_8k
model=/mnt/data4/work/llama2/models/Llama2-70b-fused-qkv-mlperf
logdir=$(realpath ../logs/logs-llama2)

source configs/config_L40S_1x16x1xtp4pp1cp2.sh
#source configs/config_DGXH100_1x8x8x4x2_fp8.sh  # use appropriate config
CONT=$container DATADIR=$datadir MODEL=$model LOGDIR=$logdir ./run_with_docker.sh
