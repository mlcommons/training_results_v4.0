# 1 node run
#source /nfs/scratch/sd/ssd/scripts/config_DGXH100_001x08x032.sh
# 8 node run
#source /nfs/scratch/sd/ssd/scripts/config_DGXH100_008x08x004.sh
# 16 node run
source /nfs/scratch/sd/ssd/scripts/config_DGXH100_016x08x004.sh
export LOGDIR=/nfs/scratch/sd/ssd/logs
#export DATASET_DIR=/mnt/localdisk/sd/ssd/ssd/datasets
#export DATASET_DIR=/nfs/scratch/sd/ssd/datasets
export DATADIR=/mnt/localdisk/sd/ssd/ssd/datasets/datasets
#export DATADIR=/nfs/scratch/sd/ssd/newdatasets
export CHECKPOINTS=/mnt/localdisk/sd/ssd/ssd/checkpoint
export BACKBONE_DIR=/mnt/localdisk/sd/ssd/ssd/weights
export NEMOLOGS=/nfs/scratch/sd/ssd/nemologs 
export WALLTIME=300
#CONT=/mnt/localdisk/sd/ssd/ssd/images/sd+mlperf-nvidia+ssd.sqsh \
CONT=/nfs/scratch/sd/ssd/images/sd+mlperf-nvidia+ssd.sqsh \
NCCL_TEST=0 \
MLPERF_SYSTEM_NAME="BM.GPU.H100.8" \
MLPERF_SUBMITTER="Oracle" \
MLPERF_STATUS="cloud" \
MLPERF_DIVISION="closed" \
NEXP=5 \
MLPERF_CLUSTER_NAME="BM.GPU.H100.8 Cluster" \
sbatch -p compute -N $DGXNNODES -t $WALLTIME run.sub
