#!/bin/bash

# Check if sufficient arguments were provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 RSYNC_SOURCE_DIR RSYNC_TARGET_DIR"
    exit 1
fi

RSYNC_SOURCE_DIR=$1
RSYNC_TARGET_DIR=$2

# Perform rsync operation
mkdir -p $RSYNC_TARGET_DIR
echo "Start syncing data from $RSYNC_SOURCE_DIR $RSYNC_TARGET_DIR"
source /cm/shared/smc/sean/mlperf_raid/psync.sh
psync "$RSYNC_SOURCE_DIR" "$RSYNC_TARGET_DIR" 8 8
echo "Data synced"


