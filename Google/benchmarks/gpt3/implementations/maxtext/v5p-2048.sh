#!/bin/bash
set -euox pipefail

BRANCH=${BRANCH:-f586e43f7ee92c701515fe0a2db17dc50f18dc81}

if [[ ! -d "maxtext" ]]; then
  git clone https://github.com/google/maxtext.git
fi

# switch branch
cd maxtext
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"

bash preflight.sh PLATFORM=gke
sleep 60

# flags set as default

# hlo dump
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file"

# debug
export TPU_STDERR_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0

# tunable parameter
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_enable_experimental_fusion_cost_model=false --xla_tpu_dot_dot_fusion_duplicated=false --xla_tpu_dot_dot_fusion=false --xla_jf_conv_input_fusion=true --xla_jf_conv_output_fusion=false --xla_tpu_rwb_fusion=false  --xla_tpu_copy_fusion_pad_unpad_ratio=300 --xla_tpu_enable_aggressive_loop_fusion_layout_opt=false --xla_tpu_enable_copy_fusion=false --xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=false --xla_tpu_scavenge_vmem_for_fusions=false --xla_tpu_vector_load_fusion_window=256 --xla_tpu_vector_store_fusion_window=64 --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_megacore_fusion=true --xla_enable_async_all_gather=true --xla_enable_async_collective_permute=true --xla_always_enable_all_gather_2d_asymmetric=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_dcn_max_overlap_estimation=32"}

# checkpoint loading from a fixed folder
RUNNAME=convergence_test_0
BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY:-gs://some-bucket}
DATASET_PATH=${DATASET_PATH:-gs://some-bucket/some-dataset-path}
DATASET_NAME=c4/en:3.0.7
SEED=8745

# set enable_checkpointing as true to load a checkpoint
# tunable parameters: ici_tensor_parallelism, per_device_batch_size, remat_policy, attention, int8_training
#  ici_tensor_parallelism is tunable and should be compatibility to topology
python3 MaxText/train.py MaxText/configs/base.yml run_name="${RUNNAME}" model_name=gpt3-175b\
  base_output_directory="${BASE_OUTPUT_DIRECTORY}"\
  enable_checkpointing=true async_checkpointing=false\
  steps=4000\
  per_device_batch_size=2\
  eval_per_device_batch_size=1\
  ici_data_parallelism=8 ici_fsdp_parallelism=16 ici_tensor_parallelism=8\
  remat_policy=save_dot_except_logits_ffn1\
  attention=flash\
  quantization=int8\
  dataset_type=c4_mlperf\
  dataset_path="${DATASET_PATH}" dataset_name="${DATASET_NAME}"\
  tokenizer_path=gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model\
  data_shuffle_seed="${SEED}"\
  2>&1 | tee /tmp/large_scale_multislice_test_log

EXP_FOLDER="${BASE_OUTPUT_DIRECTORY}/${RUNNAME}/${WORKLOAD_NAME}/${TIMESTAMP}"

if [[ ${MEGASCALE_SLICE_ID} == "0" ]]; then
  if [[ ${TPU_WORKER_ID} == "0" ]]; then
    gsutil -m cp -r /tmp/xla_dump_file "${EXP_FOLDER}/xla/"
  fi
fi

if [[ $(grep "MLLOG" /tmp/large_scale_multislice_test_log | wc -l) -gt 0 ]];then
  gsutil -m cp /tmp/large_scale_multislice_test_log "${EXP_FOLDER}/large_scale_multislice_test_log"
  bash ../parser_metrics.sh 2>&1 | tee /tmp/parser_metrics_log
  gsutil -m cp /tmp/parser_metrics_log "${EXP_FOLDER}/parser_metrics_log"
fi
