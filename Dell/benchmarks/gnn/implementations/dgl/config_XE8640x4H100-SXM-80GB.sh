## DL params

# tunable HPs
export EPOCHS="1"
#export BATCH_SIZE="2048" # local batch size - kill 9
export BATCH_SIZE="2048" # local batch size
export LEARNING_RATE="0.002" 

# WG related
export WG_SHARDING_LOCATION="cpu"
export WG_SHARDING_PARTITION="global"
export WG_SHARDING_TYPE="continuous"
export WG_GATHER_SM="8"
export SAMPLING_DEVICE="cuda"
export GRAPH_DEVICE="cuda"
export NUM_SAMPLING_THREADS="1"
export NUM_WORKERS="0"

# Knobs
export TRAIN_OVERLAP="1"
export EVAL_OVERLAP="1"
export HIGH_PRIORITY_EMBED_STREAM="1"
export USE_CONCAT_EMBEDDING="0"
export PAD_NODE_COUNT_TO="3072"

# model configs not fixed on reference branch for now
# need to remove them after the reference branch is fixed. 
export FAN_OUT="5,10,15"
export HIDDEN_DIM="512"
export NUM_HEADS="4"
export AMP="1"

# debugging
export TIMETAG="1"
export DEBUG="1"

# training related
export EVAL_FREQUENCY="0.05"
export VALIDATION_BATCH_SIZE="2048"

## System run params
export DGXNNODES=1
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=30 # every experiment takes 35 minutes, give 5 minutes as margin
export WALLTIME=$((10+(${NEXP:-1}*${WALLTIME_MINUTES})))
