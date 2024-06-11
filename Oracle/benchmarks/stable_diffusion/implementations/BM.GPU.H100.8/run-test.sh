#! /usr/bin/bash
for i in {1..10}
do
    # 8 node run
    source /nfs/scratch/sd/stable_diffusions/scripts/config_DGXH100_08x08x16.sh
    # 16 node run
    #source /nfs/scratch/sd/stable_diffusions/scripts/config_DGXH100_16x08x08.sh
    export LOGDIR=/nfs/scratch/sd/stable_diffusions/logs
    export DATADIR=/mnt/localdisk/sd/stable_diffusions/stable_diffusions/datasets
    export CHECKPOINTS=/mnt/localdisk/sd/stable_diffusions/stable_diffusions/checkpoints
    export NEMOLOGS=/nfs/scratch/sd/stable_diffusions/nemologs
    export WALLTIME=1800
    CONT=/mnt/localdisk/sd/stable_diffusions/stable_diffusions/images/sd+mlperf-nvidia+sd.sqsh \
    NCCL_TEST=0 \
    MLPERF_SYSTEM_NAME="BM.GPU.H100.8" \
    MLPERF_SUBMITTER="Oracle" \
    MLPERF_STATUS="cloud" \
    MLPERF_DIVISION="closed" \
    NEXP=1 \
    MLPERF_CLUSTER_NAME="BM.GPU.H100.8 Cluster" \
    sbatch -p compute -N $DGXNNODES -t $WALLTIME run.sub
    while :
    do
      if [[ $(squeue |egrep -i sd-conta|awk '{print $1}'|wc -l) == "0" ]]; then
        break
      else
	sleep 1m
      fi
    done
done
