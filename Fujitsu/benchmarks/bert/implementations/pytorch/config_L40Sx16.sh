## DL params
export BATCHSIZE=24
export PACKING_FACTOR=2
export GRADIENT_STEPS=1
export LR=0.00096
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=6700
export OPT_LAMB_BETA_1=0.60466
export OPT_LAMB_BETA_2=0.85437
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
export WEIGHT_DECAY_RATE=0.1
export INIT_LOSS_SCALE=1024.0

export EXTRA_PARAMS="--dense_seq_output --unpad --fused_mha --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=8

if [[ "${MLPERF_POWER_TRAIN_AFTER_RUN_STOP:-0}" == "1" ]]; then
  export WALLTIME_MINUTES=$((${WALLTIME_MINUTES} + 15))  
  export SUSTAINED_TRAINING_TIME=11
fi
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]] || [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export WALLTIME_MINUTES=$((${WALLTIME_MINUTES} + 10))
  ## gpc frequency at maxQ and minEDP point
  export MAXQ_CLK=1305
  export MINEDP_CLK=1650
fi
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_CDI_common.sh
export DGXNGPU=16

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1


