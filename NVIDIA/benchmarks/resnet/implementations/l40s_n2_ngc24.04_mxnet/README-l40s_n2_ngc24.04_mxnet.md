## Steps to launch training

### l40s_n2_ngc24.04_mxnet

Launch configuration and system-specific hyperparameters for the
l40s_n2_ngc24.04_mxnet submission are in the
`benchmarks/resnet/implementations/l40s_n2_ngc24.04_mxnet/config_L40S_2x4x204.sh` script.

Steps required to launch training for l40s_n2_ngc24.04_mxnet.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_L40S_2x4x204.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
