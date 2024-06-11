## Steps to launch training

### eos_n96_ngc23.09_mxnet

Launch configuration and system-specific hyperparameters for the
eos_n96_ngc23.09_mxnet submission are in the
`benchmarks/unet3d/implementations/eos_n96_ngc23.09_mxnet/config_DGXH100_96x8x1.sh` script.

Steps required to launch training for eos_n96_ngc23.09_mxnet.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXH100_96x8x1.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
