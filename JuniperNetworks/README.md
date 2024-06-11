# MLPerf v4.0 Juniper Networks Submission

This is a repository of Juniper Networks' submission to the MLPerf v4.0 benchmark. The reference implementations can be found 
[here](https://github.com/mlcommons/training_results_v3.1/tree/main/NVIDIA/) and [here](https://github.com/mlcommons/training/tree/master/llama2_70b_lora)

# v4.0 release
Submission based on below implementations:
- [Bert](https://github.com/mlcommons/training_results_v3.1/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch) 
- [DLRMv2](https://github.com/mlcommons/training_results_v3.1/tree/main/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr)
- [LLaMA2-70B](https://github.com/mlcommons/training/tree/master/llama2_70b_lora)

# Objective

This benchmark aims to provide a framework to run MLPerf training on a multi-node GPU infrastructure.
We utilize an Optimized Ethernet fabric, leveraging RDMA over Converged Ethernet (ROCE) version 2 as the transport protocol for inter GPU communication in the data-center.
With ROCE v2, we ensure high-speed, low-latency communication between nodes, optimizing the training workflow. Through rigorous testing and analysis, we aim to uncover insights into the system's operation and scalability potential in real-world applications.

# Environment

* NVIDIA DGX H100 servers with 8x80GB H100 SXM
  gpus.
* NVIDIA A100 H100 servers with 8x80GB A100 SXM
  gpus.  
* Ethernet based Network Fabric

# Pre-requistes
* Slurm
* Pyxis
* Enroot
* Docker
* docker-buildx
* Python 3.10.x
* External Storage cluster

# Running Benchmarks

Generally, a benchmark can be run with the following steps:

1. Follow the instructions in the README to download and format the input data and any required checkpoints.
2. Build the Dockerfile
3. Source the appropriate `config_*.sh` file.
4. `sbatch -N $DGXNNODES -t $WALLTIME run.sub`
5. `sbatch -J $JOBNAME -p $PARTITION -N $DGXNNODES --exclusive --gpus-per-node=${DGXNGPU} -t $WALLTIME -o "$LOGDIR/slurm-%j.out" run.sub`
