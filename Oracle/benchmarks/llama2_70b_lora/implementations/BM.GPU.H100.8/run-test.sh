#source /nfs/scratch/rchen/LoRA/ft-llm/configs/config_OCI-H100_1x8x1x4x1.sh
source /nfs/scratch/rchen/LoRA/ft-llm/configs/config_OCI-H100_8x8x1x4x2.sh
#source /nfs/scratch/rchen/LoRA/ft-llm/configs/config_OCI-H100_64x8x1x4x2.sh
CONT=/nfs/scratch/rchen/LoRA/ft-llm/ruzhuchen+mlperf-nvidia+lora-pytorch.sqsh \
LOGDIR=/nfs/scratch/rchen/LoRA/logs \
DATADIR=/mnt/localdisk/LoRA-Data/dataset \
MODEL=/mnt/localdisk/LoRA-Data/model \
MLPERF_SYSTEM_NAME="BM.GPU.H100.8" \
MLPERF_SUBMITTER="Oracle" \
MLPERF_STATUS="cloud" \
MLPERF_DIVISION="closed" \
NEXP=10 \
sbatch -p arnaud -N $DGXNNODES -t $WALLTIME run-oci.sub
