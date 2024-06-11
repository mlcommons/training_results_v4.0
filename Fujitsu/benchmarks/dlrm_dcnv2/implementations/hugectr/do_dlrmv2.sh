source config_GX2560M7_1x4x6912.sh

CONT=nvcr.io/nvdlfwea/mlperfv40/dlrm:20240416.hugectr
DATADIR=/mnt/data4/work/dlrm_final
LOGDIR=$(realpath ../logs/logs-dlrm)

NEXP=1
num_of_run=1

export MLPERF_SUBMISSION_ORG=Fujitsu
export MLPERF_SUBMISSION_PLATFORM=GX2560M7

for idx in $(seq 1 $num_of_run); do
    CONT=$CONT DATADIR=$DATADIR LOGDIR=$LOGDIR BACKBONE_DIR=$TORCH_HOME NEXP=$NEXP bash run_with_docker.sh
done
