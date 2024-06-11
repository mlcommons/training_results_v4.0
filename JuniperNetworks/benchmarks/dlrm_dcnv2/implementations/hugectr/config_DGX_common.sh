#Performance paramenter
export DLRM_BIND="numactl --membind=0,1"
#Mandatory param NCCL_IB_GID_INDEX=3
export NCCL_IB_GID_INDEX=3
export NCCL_TEST=0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,net,sys
