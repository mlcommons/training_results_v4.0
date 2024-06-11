## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

export DATADIR_PHASE2_PACKED=/mnt/data4/work/bert_data/packed_data
export DATADIR_PHASE2=/mnt/data4/work/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength
export EVALDIR=/mnt/data4/work/bert_data/hdf5/eval_varlength
export CHECKPOINTDIR_PHASE1=/mnt/data4/work/bert_data/phase1
