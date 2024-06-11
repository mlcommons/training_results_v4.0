: "${TIMEOUT:=02:00:00}"

# Usage:
# 1. Source this script
# 2. Run interactive
#     command - source messy_run_interactive.sh; interactive
# 3. Now you are on a node, source this script once again
# 4. Run run
#     command - = LOGDIR=/lustre/fsw/coreai_mlperf_training/llm/spalsamudram/v4.0/logs source messy_run_interactive.sh; CONT=gitlab-master.nvidia.com/dl/mlperf/optimized:large_language_model.pytorch.<pipeline> run
# 5. You are inside docker container. source desired config
# 6. source config_DGXH100_1x8x32x8x1.sh && bash run_and_time.sh 2>&1 | tee run.log


# Steps to install diff version of TE in the container. It should take ~10 mins:
# cd /opt
# git clone https://github.com/NVIDIA/TransformerEngine
# cd TransformerEngine
# git checkout main
# git pull
# git checkout da30634a6c9ccdbb6c587b6c93b1860e4b038204
# git submodule init
# git submodule update
# NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .


export SEED=1
export DGXNGPU=8
export PARTITION=batch
export SPM="/lustre/fsr/datasets/llm/c4/tokenizers/google_c4_spm/c4_en_301_5Mexp2_spm.model"
export LOAD_CHECKPOINTS_PATH="/lustre/fsr/datasets/llm/dist_ckpts/googleckpt-1101/nemo"
export CHECKPOINT_NAME=ckpt4000-consumed_samples=0
export PREPROC_DATA="/lustre/fsr/datasets/llm/c4/preprocessed_c4_googlespm"
export NPY_INDEX_DIR="/lustre/fsw/coreai_mlperf_training/llm/spalsamudram/v4.0/npy_index"

export GPU_ARCH=h

export _cont_mounts="${SPM}:/workspace/llm/tokenizer.model,${LOAD_CHECKPOINTS_PATH}:/load_checkpoints"
export _cont_mounts="${LOGDIR}:/results,${NPY_INDEX_DIR}:/npy_index,${_cont_mounts}"
export _cont_mounts="${_cont_mounts},$PREPROC_DATA:/preproc_data"


# Can uncomment these if you want to mount Nemo and optimized repo
# export OPTIMIZED_LLM="/lustre/fsw/coreai_mlperf_training/llm/spalsamudram/v4.0/optimized/large_language_model/pytorch"
# export _cont_mounts="${OPTIMIZED_LLM}:/workspace/llm,${_cont_mounts}"
# export NEMO="/lustre/fsw/coreai_mlperf_training/llm/spalsamudram/v4.0/shriya-NeMo"
# export _cont_mounts="${NEMO}:/workspace/NeMo,${_cont_mounts}"
# export EOS_LLM="/lustre/fsw/coreai_mlperf_training/llm/spalsamudram/v4.0"
# export _cont_mounts="${EOS_LLM}:${EOS_LLM},${_cont_mounts}"


function interactive(){
salloc -n 1 -N 1 -A coreai_mlperf_training -t ${TIMEOUT} -J interactive -p ${PARTITION}
}

function run(){
JOBID=$(squeue --me --name interactive -o "%i" -h)
# JOBID=300520
srun --jobid=${JOBID} --container-image ${CONT} --container-mounts ${_cont_mounts} --pty bash
}

# To run another session on the same node
#run --jobid=${JOBID} --container-image ${CONT} --container-mounts /lustre/fsw:/lustre/fsw --pty --overlap bash
