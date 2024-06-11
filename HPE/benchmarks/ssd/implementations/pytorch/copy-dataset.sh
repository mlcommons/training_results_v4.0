#!/bin/bash

FS=lvol
case $FS in
        beeond | lvol | nfsond )
                mkdir -p /$FS/mlperf/
                pushd .
                cd /$FS/mlperf/
		pigz -dc /cstor/SHARED/datasets/MLPERF/openimages-mlperf2.1.tar.gz | tar xf -
		popd
                export DATADIR="/$FS/mlperf/training2.1/openimages"
                ;;
        daos )
                srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "export LD_PRELOAD=/usr/lib64/libioil.so"
                mkdir -p /$FS/mlperf/
                pushd .
                cd /$FS/mlperf/
                tar xf /pfss/hddfs1/MLCOMMONS/training2.1/opeimages.tar
                popd
                export DATADIR="/$FS/mlperf/openimages"
                ;;
        pfss)
                SBATCH_FS=''
                export DATADIR="/pfss/nvmefs1/MLCOMMONS/training2.1/openimages"
               ;;
esac


