#!/bin/bash

#source $(dirname ${BASH_SOURCE[0]})/config_common.sh
export WARMUP=True
export DGXNGPU=8
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# TP comm overlap settings
#Large value if overlap is disabled
export NCCL_MIN_CTAS=8
export TP_COMM_OVERLAP=True
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True
export MC_TP_OVERLAP_RS_DGRAD=False
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export NVTE_RS_STRIDED_ATOMIC=2
export LORA_A2A=1

export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1

export POSSIBLE_USER_WARNINGS=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # Disable caching NCCL communication buffer
export NCCL_NVLS_ENABLE=0 # Disable NVL SHARP, which don't use
# other
export MBS=1
export SKIP_EVALS=4
export VAL_CHECK_INTERVAL=384








# hyperparameters
export MAX_STEPS=1024
export MINIBS=4

export TP=4
export PP=1
export CP=1
export SP=1
export TP_COMM_OVERLAP=True

export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4

export SKIP_EVALS=3
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=1
export WALLTIME_MINUTES=45
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))


#for troubleshooting
export TP_COMM_OVERLAP=False
