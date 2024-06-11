## Steps to launch training

### eos_n8_ngc24.04_mxnet

Launch configuration and system-specific hyperparameters for the
eos_n8_ngc24.04_mxnet submission are in the
`benchmarks/resnet/implementations/eos_n8_ngc24.04_mxnet/config_DGXH100_multi_8x8x50.sh` script.

Steps required to launch training for eos_n8_ngc24.04_mxnet.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXH100_multi_8x8x50.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
