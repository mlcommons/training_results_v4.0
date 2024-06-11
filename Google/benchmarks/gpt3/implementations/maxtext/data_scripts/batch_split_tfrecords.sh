#!/bin/bash

script_dir=$(pwd)
input_dir=gs://mlperf-llm-public2/c4/en/3.0.4
input_filename=c4-train2.tfrecord
output_dir=gs://some-bucket/some-dataset-path/c4/en/3.0.7
num_inputs=1024
num_splits=6
batch_size=16

set -x

num_inputs_str=$(printf "%05d" $num_inputs)

for i in $(seq 0 $((num_inputs -1))); do
  i_str=$(printf "%05d" "$i")
  input_tfrecord="${input_dir}/${input_filename}-${i_str}-of-${num_inputs_str}"
  echo "splitting $input_tfrecord"
  output_prefix="${output_dir}/c4-train2"
  log="/tmp/c4-train2-${i_str}.txt"

  python3 "${script_dir}"/split_tfrecord.py \
    --input_tfrecord="${input_tfrecord}" \
    --num_splits="${num_splits}" \
    --output_prefix="${output_prefix}" \
    > "${log}" 2>&1 &

  if [[ $(((i + 1) % batch_size)) -eq 0 ]]; then
    wait
  fi
done

wait
gsutil cp "${input_dir}/data_governance.textproto" "${output_dir}"
gsutil cp "${input_dir}/data_info.json" "${output_dir}"
gsutil cp "${input_dir}/features.json" "${output_dir}"
set +x
