## Steps to launch training

### l40s_n16_ngc24.04_nemo

Launch configuration and system-specific hyperparameters for the
l40s_n16_ngc24.04_nemo submission are in the
`benchmarks/stable_diffusion/implementations/nemo/config_L40S_16x04x32.sh` script.

Steps required to launch training for l40s_n16_ngc24.04_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_L40S_16x04x32.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
