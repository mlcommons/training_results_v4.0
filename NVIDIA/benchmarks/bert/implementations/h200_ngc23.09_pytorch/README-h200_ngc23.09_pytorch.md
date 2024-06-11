## Steps to launch training

### h200_ngc23.09_pytorch

Launch configuration and system-specific hyperparameters for the
h200_ngc23.09_pytorch submission are in the
`benchmarks/bert/implementations/h200_ngc23.09_pytorch/config_DGXH100_1x8x48x1_pack.sh` script.

Steps required to launch training for h200_ngc23.09_pytorch.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXH100_1x8x48x1_pack.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
