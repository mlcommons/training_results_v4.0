## MLPerf v4.0 Juniper Networks Submission 
This is a repository of Juniper Networks implementations for the [MLPerf](https://mlcommons.org/en/) training benchmark on [DLRM DCNv2](https://github.com/mlcommons/training_results_v3.1/tree/main/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr).

## Objective

This benchmark aims to provide a framework to run MLPerf training on a multi-node GPU infrastructure.

We utilize an Optimized Ethernet fabric, leveraging RDMA over Converged Ethernet (ROCE) version 2 as the transport protocol for inter GPU communication in the data-center.

With ROCE v2, we ensure high-speed, low-latency communication between nodes, optimizing the inference workflow. Through rigorous testing and analysis, we aim to uncover insights into the system's operation and scalability potential in real-world applications.

## Prerequisites

- Servers with A100 or H100 GPUs
- Optimized Ethernet Fabric for Accelerator Interconnect

## Software Dependencies

- Docker
- Slurm with Enroot + Pyxis
- Nvidia Container Toolkit
- Pytorch
- CUDA

# Running Nvidia HugeCTR DLRM DCNv2 MLPerf benchmark
We follow the steps as documented by Nvidia - https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr/README.md#running-nvidia-hugectr-dlrm-dcnv2-mlperf-benchmark


## Running training

Note that this benchmark needs a high bandwidth for input/output operations. For the best results, verify your storage I/O and ensure your network switches are rail optimized and provide adequate bandwidth.

### Steps to launch training on multiple nodes
https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr/README.md#steps-to-launch-training-on-multiple-nodes

