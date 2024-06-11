## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
## network flags
export NCCL_DEBUG=INFO
# extra added by ocloud
export NCCL_IB_GID_INDEX=3
export NCCL_TEST=0
export NCCL_DEBUG_SUBSYS=env,net
