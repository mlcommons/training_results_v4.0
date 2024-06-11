#docker run --cap-add=sys_nice --gpus=all --ipc=host -it --rm -v /raid/mlperf/lora/gov_report:/data -v /raid/mlperf/lora/model/Llama-2-70b-bf16-mcore:/ckpt -v ${PWD}/results:/results gitlab-master.nvidia.com/dl/mlperf/optimized:lora_tot1
docker run --cap-add=sys_nice --gpus=all --ipc=host -it --rm -v /raid/mlperf/lora/gov_report:/data -v /raid/mlperf/lora/model/Llama-2-70b-bf16-mcore:/ckpt -v ${PWD}/results:/results gitlab-master.nvidia.com/dl/mlperf/optimized:llama2_70b_lora.pytorch.14547118
#/workspace/ft-llm/NeMo/nemo/collections/nlp/modules/common/megatron/utils.py
