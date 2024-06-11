#! /bin/bash
container=nvcr.io/nvdlfwea/mlperfv40/lora:20240427.pytorch
dataset=/mnt/data4/work/llama2/dataset/scrolls_gov_report_8k
model=/mnt/data4/work/llama2/models/Llama2-70b-fused-qkv-mlperf
docker run -it --gpus all -v $dataset:/data -v $model:/model $container bash
