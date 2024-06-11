# Instruction for GPT3 MLPerf workload

## 1. Problem

Large Language Model - GPT3 175B

### Requirements

*   [Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
*   [GKE (Google Kubernetes Engine) verson: 1.27.4-gke.900](https://cloud.google.com/kubernetes-engine)


## 2. Directions

### Environment Setup
```
bash setup.sh
```

#### Network Setup

```bash
ZONE=us-east5-c
TPU_TYPE=v5p-1024  # can be any one of v5p-1024, v5p-2048, v5p-3072, v5p-12288
CLUSTER_NAME="mlperf-${TPU_TYPE}-${ZONE}"
NETWORK_NAME="${CLUSTER_NAME}-mtu9k"
NETWORK_FW_NAME="${NETWORK_NAME}-fw"
PROJECT=some-cloud-tpu-project-id
NUM_SLICES=1

# network setup
gcloud compute networks create "${NETWORK_NAME}" --mtu=8896 --project="${PROJECT}" --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create "${NETWORK_FW_NAME}" --network "${NETWORK_NAME}" --allow tcp,icmp,udp --project="${PROJECT}"
```

#### Cluster Setup
We use [xpk](https://github.com/google/xpk) to create cluster.

```bash
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
python3 xpk.py cluster create --cluster "${CLUSTER_NAME}" \
  --num-slices="${NUM_SLICES}" --tpu-type="${TPU_TYPE}" --zone="${ZONE}" \
  --project="${PROJECT}" --on-demand \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"
```

### Steps to launch training
We use [xpk](https://github.com/google/xpk) to deploy jobs as well.
We can utilize commands similar to those in `run_and_time.sh`, such as
```
WORKLOAD_NAME=${USER}-v5p-1024 SCRIPT=v5p-1024.sh bash xpk_run.sh
WORKLOAD_NAME=${USER}-v5p-2048 SCRIPT=v5p-2048.sh bash xpk_run.sh
WORKLOAD_NAME=${USER}-v5p-3072 SCRIPT=v5p-3072.sh bash xpk_run.sh
WORKLOAD_NAME=${USER}-v5p-12288 SCRIPT=v5p-12288.sh bash xpk_run.sh
```

The `SCRIPT` will be attached as the workload inside each pod.
And we use `xpk_run.sh` to trigger the work deployment on a cluster.

Each `SCRIPT` covers both gpt3 task running and timing at the end.

## 3. Dataset

Please refer to the
[instructions](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md)
from the reference to download the dataset.

The C4 dataset location: `gs://mlperf-llm-public2/c4`

The tokenizer location as the SPM variable are: `gs://mlperf-llm-public2/vocab`.

### Dataset Split

There are 1024 tfrecords in the original train split `gs://mlperf-llm-public2/c4/en/3.0.4/` which didn't match 1536 hosts in large scale v5p-12288 run.

We simply split each tfrecord file into 6 and create a new split containing 6144 tfrecords to address the above issue.
Additionally, `dataset_info.json` should be updated accordingly.

```
cd data_scripts

# change to your gcs folder in batch_split_tfrecords.sh
bash batch_split_tfrecords.sh

# a gcs folder gs://some-bucket/some-dataset-path/c4/en/3.0.7/ as an example
python create_new_shard_info.py --gcs-prefix=gs://some-bucket/some-dataset-path/c4/en/3.0.7/
```

## 4. Model

The model largely follows the GPT-3 paper, with key model architecture configs
refer
[here](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md#3-model)

### List of Layers

The model largely follows the GPT3 [paper](https://arxiv.org/abs/2005.14165),
refer
[here](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md#3-model)
for model details.

### Model checkpoint

In the benchmarking region, we convert from a reference pax checkpoint which
is trained with Global Batch Size of 1536 for 4000 iterations.

To resume training, firstly the checkpoint needs to be converted from the [Paxml](https://github.com/google/paxml)
reference checkpoint to [maxtext](https://github.com/google/maxtext/tree/main) one by running

```
WORKLOAD_NAME=${USER}-ckpt-convert SCRIPT=ckpt-convert.sh bash xpk_run.sh
```

See [`convert_gpt3_ckpt_from_paxml.py`](https://github.com/google/maxtext/blob/f586e43f7ee92c701515fe0a2db17dc50f18dc81/MaxText/convert_gpt3_ckpt_from_paxml.py) for detailed conversion.

## 5. Quality

### Quality metric

Log Perplexity

### Quality target

2.69

### Evaluation frequency

Evaluate after every 24576 sequences with a length of 2048 each (=50.33B tokens)

### Evaluation thoroughness

Evaluation on the validation subset that consists of 24567 examples.

## 6. Additional notes

postprocess for MLLOG from raw run

```
cat ${job_dir}/large_scale_multislice_test_log | grep MLLOG  > ${job_dir}/result_0.txt
```