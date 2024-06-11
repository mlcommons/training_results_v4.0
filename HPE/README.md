# MLPerf v4.0 HPE Submission

This is a repository of HPE's submission to MLPerf v4.0 benchmarks, which is a variant from NVIDIA's original repository at ../NVIDIA.  It
includes implementations of the benchmark code optimized for running on HPE systems with NVIDIA
GPUs.  The reference implementations can be found elsewhere:
https://github.com/mlcommons/training.git

# Contents

Each implementation in the `benchmarks` subdirectory provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to build a container for the benchmark.
* Documentation on the dataset, model, and machine setup.

# Running Benchmarks

These benchmarks have been tested on the following machine configuration:

* HPE Cray XD670 servers with 8x80GB NVIDIA H100 SXM gpus.
* The required software stack includes Slurm, with Enroot for running
  containers and the Slurm Pyxis plugin

Generally, a benchmark can be run with the following steps:

1. Follow the instructions in the README to download and format the input data and any required checkpoints.
2. Build the Dockerfile and convert the docker container to enroot format
3. Source the appropriate `config_*.sh` file.
4. `sbatch -N $DGXNNODES -t $WALLTIME run.sub`

**Example:**
```
source ./benchmarks/bert/implementations/pytorch/config_XD670_1x8x48x1_pack.sh && \
NEXP=10 \
CONT=<path_to_bert_enroot_container.sqsh> \
DATADIR_PHASE2=<path_to_dataset/bert-packed/packed_data> \
CHECKPOINTDIR_PHASE1=<path_to_dataset/phase1> \
EVALDIR=<path_to_dataset/hdf5/eval_varlength> \
LOGDIR=<path_to_dataset/results/1-node-test> \
sbatch -N 1 -p <slurm_partition_name>
```