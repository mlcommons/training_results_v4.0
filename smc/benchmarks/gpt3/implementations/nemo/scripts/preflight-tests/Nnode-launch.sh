#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'First arg: type of test is missing, choose between (1node/Nnode/e2e)'
    exit 1
fi

PARTITION=batch
RESERVATION=mlperf_training

DATE=$(date '+%m_%d')

#vars with defaults:
: "${CONTAINER:=/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/containers/dl+mlperf+optimized+large_language_model.pytorch.14295872.sqsh}"

# these were using the wrong syntax (:- instead of :=) so weren't working to define defaults.
# But they can't be defined simultaneously anyway, so don't default them.
#: "${EXCLUDELIST:=/lustre/fsw/portfolios/coreai/users/${USER}/logs/${DATE}/exclude.list}"
#: "${NODELIST:=/lustre/fsw/portfolios/coreai/users/${USER}/logs/${DATE}/all.list}"

echo "EXCLUDELIST is ${EXCLUDELIST:-}"
echo "NODELIST is ${NODELIST:-}"

OPTIMIZED="/lustre/fsw/portfolios/coreai/users/${USER}/optimized"

LOGDIRBASE="/lustre/fsw/portfolios/coreai/users/${USER}/logs/${DATE}"
mkdir -p "${LOGDIRBASE}"

# MLPERF_VERSION ENV is used to v3.1 version of config_common.sh if we are running v3.1 container
if [[ $CONTAINER =~ "large_language_model.pytorch.10082486" || $CONTAINER =~ "large_language_model.pytorch.14295872" ]]; then
    MLPERF_VERSION=v31
else
    MLPERF_VERSION=v40
fi

EXCLUDEARG=
NODEARG=

if [[ "${EXCLUDELIST:-}" ]]; then
    echo "setting excludearg"
    EXCLUDEARG="--slurm-extra=--exclude=${EXCLUDELIST}"
fi

if [[ "${NODELIST:-}" ]]; then
    echo "setting nodearg"
    NODEARG="--slurm-extra=--nodelist=${NODELIST}"
fi

if [[ "${NODEARG:-}" && "${EXCLUDEARG:-}" ]]; then
    echo "Set either EXCLUDELIST or NODELIST. Both together do not work"
    exit 1
fi

case $1 in
    1node)
        DIR_NAME_SUFFIX=${2:-}
        CONFIG=config_DGXH100_1x8x8x8x1_mbs1.sh
        LOGDIR=${LOGDIRBASE}/1n${DIR_NAME_SUFFIX}
        mkdir -p ${LOGDIR}
        
        # launch one run per node
        cat ${NODELIST} | while read LINE; do
            DGXNNODES=1 USE_SYNTHETIC_DATASET=0 NO_CKPT=1 MLPERF_VERSION=${MLPERF_VERSION} LOGDIR=${LOGDIR} bash ${OPTIMIZED}/mlperf_utils/launch-mlperf-benchmark.sh --config ${OPTIMIZED}/large_language_model/pytorch/${CONFIG} --runsub ${OPTIMIZED}/large_language_model/pytorch/run.sub --container ${CONTAINER} --partition $PARTITION --npar 1 --slurm-extra=--nodelist=$LINE --slurm-extra=--output="${LOGDIR}/slurm-%j.out" --slurm-extra=--reservation=$RESERVATION
        done
        ;;
    loopback)
        DIR_NAME_SUFFIX=${2:-}
        LOGDIR=${LOGDIRBASE}/1n-loopback${DIR_NAME_SUFFIX}
        mkdir -p ${LOGDIR}
        
        # launch one run per node
        cat ${NODELIST} | while read LINE; do
            sbatch --exclusive --gres=gpu:8 -t 10 -N 1 \
		   --nodelist="${LINE}" --output="${LOGDIR}/slurm-loopback-%N-%j.out" --reservation="${RESERVATION}" --partition ${PARTITION} --account=coreai_mlperf_training --job-name=coreai_mlperf_training:.loopback-test "${OPTIMIZED}"/large_language_model/pytorch/scripts/preflight-tests/loopback.sbatch
        done
        ;;
	
    Nnode)
        if [ "$#" -lt 3 ]; then
            echo "Illegal number of parameters"
            echo 'Second arg: number of nodes per run'
            echo 'Third arg: number of runs'
            echo '(Optional) Fourth arg: directory name suffix'
            exit 1
        fi
        NNODES=${2}
        NRUNS=${3}
        DIR_NAME_SUFFIX=${4:-}
        CONFIG=config_DGXH100_Nx8xMx4x8_mbs1_nonbag.sh
        LOGDIR=${LOGDIRBASE}/${NNODES}n${DIR_NAME_SUFFIX}
        mkdir -p ${LOGDIR}
        
        # launch runs
        for i in $(seq 1 $NRUNS)
        do
            MLPERF_VERSION=${MLPERF_VERSION} DGXNNODES=$NNODES MINIBS=8 LOGDIR=${LOGDIR} bash ${OPTIMIZED}/mlperf_utils/launch-mlperf-benchmark.sh --config ${OPTIMIZED}/large_language_model/pytorch/${CONFIG} --runsub ${OPTIMIZED}/large_language_model/pytorch/run.sub --container ${CONTAINER} --partition ${PARTITION} --npar 1 --slurm-extra=--reservation=$RESERVATION --slurm-extra=--output="${LOGDIR}/slurm-%j.out" ${EXCLUDEARG} ${NODEARG}
        done
        ;;
    e2e)
        if [ "$#" -lt 3 ]; then
            echo "Illegal number of parameters"
            echo 'Second arg: number of nodes'
            echo 'Third arg: MINIBS'
            echo '(Optional) Fourth arg: config name override. Default is config_DGXH100_Nx8xMx4x8_mbs1.sh'
            exit 1
        fi
        NNODES=${2}
        GA=${3}
        TP=${4:-4}
        PP=${5:-8}
        CONFIG=${6:-config_DGXH100_Nx8xMxTPxPP_mbs1_cg.sh}
        LOGDIR=${LOGDIRBASE}/${NNODES}n
        mkdir -p ${LOGDIR}

        echo "GA: " $GA " TP: " $TP " PP: " $PP

        # launch one run
        TP=${TP} PP=${PP} MLPERF_VERSION=${MLPERF_VERSION} DGXNNODES=${NNODES} MINIBS=${GA} LOGDIR=${LOGDIR} bash ${OPTIMIZED}/mlperf_utils/launch-mlperf-benchmark.sh --config ${OPTIMIZED}/large_language_model/pytorch/${CONFIG} --runsub ${OPTIMIZED}/large_language_model/pytorch/run.sub --container ${CONTAINER} --partition ${PARTITION} --npar 1 --slurm-extra=--reservation=$RESERVATION --slurm-extra=--output="${LOGDIR}/slurm-%j.out" ${EXCLUDEARG} ${NODEARG} --no-requeue
        ;;
    *)
        echo "Error: unrecognized option $1, pass 1node/Nnode/e2e " >&2
        exit 3
        ;;
esac
