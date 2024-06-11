# Run preflight/e2e tests on a cluster

Edit `Nnode-launch.sh` manually to update values for `DATE`.  You might also need to manually update `PARTITION`, and `RESERVATION` if those have changed since last time.

The variables `OPTIMIZED`, `LOGDIRBASE`, will default to "optimized/" and "logs/${DATE}" under your /lustre user directory.  You'll need to edit those if you want those directories somewhere else.  Output `slurm-<jobid>.out` files are saved to the `LOGDIRBASE/DATE/<num nodes>n`.

`CONTAINER` defaults to the sqsh file for the regression testing container which is the 3.1 container, but is overridable by exporting it from your shell while running the script.

`NODELIST` defaults to `$LOGDIRBASE/$DATE/all.list`, but is overridable by exporting it from your shell while running the script. Same variable also controls the list of nodes used for the 1 node preflight tests. `EXCLUDELIST` defaults to `$LOGDIRBASE/$DATE/exclude.list` but is overridable by exporting it from your shell while running the script. It controls the list of nodes that will get excluded from the Nnode and e2e runs. Both `NODELIST` and `EXCLUDELIST` cannot be set together because slurm does not honor them both simultaneously, so if you only want to set `NODELIST`, pass `EXCLUDELIST=` from your shell while running the script.

## Launch 1 node loopback tests
1-node `loopback.sbatch` is used to run on every node provided in the `NODELIST`. Pass `EXCLUDELIST=` from your shell so that only `NODELIST` is used.

```
# launch command
# (Optional) second arg: directory name suffix
bash Nnode-launch.sh loopback 
```

This creates `slurm-loopback-<node name>-<job ID>.out` for each node. We expect all healthy and good nodes to have `Avg bus bandwidth >= 47.2`. So to find the list of nodes with slow nccl loopback test results, use the folling grep command.

```
grep "Avg bus bandwidth" 1n-loopback/slurm-loopback*.out | awk '$6<47.2 {print}'
```

## Launch 1 node preflight tests
1-node `DGXH100_1x8x8x8x1_mbs1` config is used to run on every node provided in the `NODELIST`. This test runs on real data and it does not load a checkpoint. Pass `EXCLUDELIST=` from your shell so that only `NODELIST` is used.

```
# launch command
bash Nnode-launch.sh 1node
```

## Launch N-node preflight tests
It runs `config_DGXH100_Nx8xMx4x8_mbs1_nonbag` config. Similar to 1node preflight tests but unlike the 1-node preflight test that runs a smaller 12-layer gpt3 model, this config runs the full 96-layer gpt3 model and you can run multiple N-node runs at once.

```
# Second arg: number of nodes per run
# Third arg: number of runs
# (Optional) Fourth arg: directory name suffix

# launch command to run 30 32-node jobs
bash Nnode-launch.sh Nnode 32 30
```

Logs are written to `/lustre/fsw/portfolios/coreai/users/<your user name>/logs/<date set in script>/<number of nodes per run>n<optional suffix>`

## parse_preflight.py
Process preflight result logs to catch nodes with launch issues or slow nodes/switches. The first arg is expected to be the nodelist but pass a random nodelist path if you did not use a nodelist for the preflight runs.

```
# command
python3 parse_preflight.py nodelist logdir
```

## Launch N-node e2e tests
After the preflight tests are done, we can run large scale e2e runs on the good set of nodes. It runs `config_DGXH100_Nx8xMx4x8_mbs1` config.

```
# Second arg: number of nodes
# Third arg: mini batch size (or GA)
# Fourth arg: TP
# Fifth arg: PP
# (Optional) sizth arg: config name override. Default is config_DGXH100_Nx8xMx4x8_mbs1.sh

# launch command to run 512-node run with MINIBS 16, TP4 PP8
bash Nnode-launch.sh e2e 512 16 4 8
```

Note that if you are passing `NODELIST`, the number of nodes in the `NODELIST` should be exactly the same as the number of nodes which is the second arg for this run.

## Run v3.1 version of the code
To run v3.1 containers from main, we source config_common_v31.sh instead of config_common.sh so that all settings match v3.1. To toggle between v3.1 vs now, the script sets `MLPERF_VERSION` based on the whether we use container `10082486` or not.

## Slurm commands to work with nodelists

### To get the list of nodes in a reservation

Use the script `getnodelist <reservation_name>`.  This prints all nodes in the reservation that are not currently down or drained.  It prints the nodelist one per line like this-

```
slurm-h100-027-01
slurm-h100-027-02
slurm-h100-027-03
slurm-h100-027-04
slurm-h100-028-01
slurm-h100-028-02
slurm-h100-028-03
slurm-h100-028-04
slurm-h100-029-01
...
```

Sometimes if the same node is assigned to multiple reservations, then the above command does not work properly. In that case use `getreservationlist <reservation_name>'

### To get the list of nodes from a run that got drained during/after the job

Use the script `getdrainednodeslist <JOBID>`