#source /nfs/scratch/rchen/U-Net3D/scripts/config_OCI-H100_1x8x7.sh
source /nfs/scratch/rchen/U-Net3D/scripts/config_OCI-H100_9x8x1.sh
CONT=/nfs/scratch/rchen/U-Net3D/scripts/ruzhuchen+mlperf-nvidia+image_segmentation-mxnet.sqsh \
LOGDIR=/nfs/scratch/rchen/U-Net3D/logs \
DATADIR=/nfs/scratch/rchen/U-Net3D/dataset \
NEXP=40 \
NCCL_TEST=0 \
MLPERF_CLUSTER_NAME="BM.GPU.H100.8 cluster" \
MLPERF_SUBMITTER="Oracle" \
MLPERF_STATUS="cloud" \
MLPERF_DIVISION="closed" \
sbatch -w compute-permanent-node-[594,634,640,654,666,669,673,689]  -p arnaud -N $DGXNNODES -t $WALLTIME run-oci.sub
