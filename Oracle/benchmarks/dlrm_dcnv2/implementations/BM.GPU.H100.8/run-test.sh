source /nfs/scratch/rchen/DLRMv2/dlrm/config_OCI-H100_1x8x6912.sh
#LOGDIR=/nfs/scratch/rchen/DLRMv2/logs 
#DATADIR=/nfs/scratch/rchen/DLRMv2/dataset/criteo/criteo_1tb_multihot_raw \
#source /nfs/scratch/rchen/DLRMv2/dlrm/config_OCI-H100_8x8x2112.sh
CONT=/nfs/scratch/rchen/DLRMv2/dlrm/ruzhuchen+mlperf-nvidia+recommendation_hugectr.sqsh \
LOGDIR=/nfs/scratch/rchen/DLRMv2/logs \
DATADIR=/mnt/localdisk/DLRM-Data \
NEXP=10 \
NCCL_TEST=0 \
MLPERF_CLUSTER_NAME="BM.GPU.H100.8 cluster" \
MLPERF_SYSTEM_NAME="BM.GPU.H100.8" \
MLPERF_SUBMITTER="Oracle" \
MLPERF_STATUS="cloud" \
MLPERF_DIVISION="closed" \
sbatch -w compute-permanent-node-435 -p arnaud -N $DGXNNODES -t $WALLTIME run-oci.sub
