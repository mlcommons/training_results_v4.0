#!/bin/bash

smc_init() {
    storage_init
    echo "SMC start"
#    power_monitoring_init
}

smc_stop() {
    power_monitoring_stop
    echo "SMC stop"
}

storage_init() {
    if [[ -v LOCAL_NVME_CACHE && -v RSYNC_SOURCE_DIR && -v RSYNC_TARGET_DIR ]]; then
        srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 /bin/bash -c "/cm/shared/smc/sean/mlperf_raid/prepare_raid.sh $RSYNC_SOURCE_DIR $RSYNC_TARGET_DIR"
    fi
}

#power init is already in original sub file, so no change here.
power_monitoring_init() {
	echo test
}

power_monitoring_stop() {
    if [ -f "$POWERCMDDIR/power_monitor.sh"  ]; then
        for i in "${NODELIST[@]}"
        do
            ssh $i 'pkill -f power_monitor.sh'
            echo "ssh $i 'pkill -f power_monitor.sh'"
        done
    fi
}
