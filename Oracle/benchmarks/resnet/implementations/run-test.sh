#8NODERUN#source /nfs/scratch/sd/ResNet50/scripts/config_DGXH100_multi_8x8x50.sh
source /nfs/scratch/sd/ResNet50/scripts/config_DGXH100.sh
export LOGDIR=/nfs/scratch/sd/ResNet50/logs 
export DATADIR=/mnt/localdisk/sd/ResNet50/ResNet50/dataset/prepared
export WALLTIME=120
CONT=/mnt/localdisk/sd/ResNet50/ResNet50/images/sd+mlperf-nvidia+resnet.sqsh \
NCCL_TEST=0 \
MLPERF_SYSTEM_NAME="BM.GPU.H100.8" \
MLPERF_SUBMITTER="Oracle" \
MLPERF_STATUS="cloud" \
MLPERF_DIVISION="closed" \
NEXP=5 \
NUMEPOCHS=38 \
MLPERF_CLUSTER_NAME="BM.GPU.H100.8 Cluster" \
sbatch -p compute -N $DGXNNODES -t $WALLTIME run.sub
