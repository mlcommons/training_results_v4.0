#!/bin/bash
ZONE=us-east5-c
TPU_TYPE=v5p-1024  # can be any one of v5p-1024, v5p-2048, v5p-3072, v5p-12288
CLUSTER_NAME="mlperf-${TPU_TYPE}-${ZONE}"
NETWORK_NAME="${CLUSTER_NAME}-mtu9k"
NETWORK_FW_NAME="${NETWORK_NAME}-fw"
PROJECT=some-cloud-tpu-project-id
NUM_SLICES=1

# network setup
gcloud compute networks create "${network_name}" --mtu=8896 --project="${project}" --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create "${network_fw_name}" --network "${network_name}" --allow tcp,icmp,udp --project="${project}"

# cluster setup
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
python3 xpk.py cluster create --cluster "${CLUSTER_NAME}" \
  --num-slices="${NUM_SLICES}" --tpu-type="${TPU_TYPE}" --zone="${ZONE}" \
  --project="${PROJECT}" --on-demand \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"