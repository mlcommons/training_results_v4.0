## Steps to launch training

### eos_n434_ngc23.09_pytorch

Launch configuration and system-specific hyperparameters for the
eos_n434_ngc23.09_pytorch submission are in the
`benchmarks/bert/implementations/eos_n434_ngc23.09_pytorch/config_DGXH100_434x8x2x1_pack_clipafter.sh` script.

Steps required to launch training for eos_n434_ngc23.09_pytorch.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXH100_434x8x2x1_pack_clipafter.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
