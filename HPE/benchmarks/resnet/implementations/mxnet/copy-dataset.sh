#!/bin/bash

FS=lvol
case $FS in
        beeond | lvol | nfsond )
                mkdir -p /$FS/mlperf/
                pushd .
                cd /$FS/mlperf/
                tar xf /cstor/SHARED/datasets/MLPERF/training2.1/resnet.bz2
                popd
                export DATADIR="/${FS}/mlperf/resnet/preprocess"
                ;;
        daos )
                srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "export LD_PRELOAD=/usr/lib64/libioil.so"
                mkdir -p /$FS/mlperf/
                pushd .
                cd /$FS/mlperf/
                tar xf /pfss/hddfs1/MLCOMMONS/training2.1/resnet.bz2
                popd
                export DATADIR="/${FS}/mlperf/resnet/preprocess"
                ;;
        pfss)
                SBATCH_FS=''
                export DATADIR="${TGT_DIR}/resnet/preprocess"
                ;;
esac

