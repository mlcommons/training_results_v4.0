## DL params

# tunable HPs
# 10 experiments show that 
# this config converges at around 2.50 epochs
export EPOCHS="3" 
export BATCH_SIZE="512" # local batch size
export LEARNING_RATE="0.005" # optimal LR at this config

# WG related
export WG_SHARDING_LOCATION="cuda"
export WG_SHARDING_PARTITION="global"
export WG_SHARDING_TYPE="distributed"
export SAMPLING_DEVICE="cuda"
export GRAPH_DEVICE="cuda"
export NUM_SAMPLING_THREADS="1"
export NUM_WORKERS="0"

# Knobs
export TRAIN_OVERLAP="1"
export EVAL_OVERLAP="1"
export HIGH_PRIORITY_EMBED_STREAM="1"
export PAD_NODE_COUNT_TO="3072"

export FAN_OUT="5,10,15"
export HIDDEN_DIM="512"
export NUM_HEADS="4"
export AMP="1"

# debugging
export TIMETAG="1"
export DEBUG="1"

# training related
export EVAL_FREQUENCY="0.05"
export VALIDATION_BATCH_SIZE="1024"

## System run params
export DGXNNODES=64
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# 10 experiments show that the avg run time is 77.32s (<2min)
# adding some margins here
WALLTIME_MINUTES=5
export WALLTIME=$((10+(${NEXP:-1}*${WALLTIME_MINUTES})))

