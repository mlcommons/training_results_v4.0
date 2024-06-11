## MLPerf v4.0 Juniper Networks Submission 
This is a repository of Juniper Networks implementations for the [MLPerf](https://mlcommons.org/en/) training benchmark on [Bert-large](https://github.com/mlcommons/training_results_v3.1/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch).

## Objective

This benchmark aims to provide a framework to run MLPerf training on a multi-node GPU infrastructure.

We utilize an Optimized Ethernet fabric, leveraging RDMA over Converged Ethernet (ROCE) version 2 as the transport protocol for inter GPU communication in the data-center.

With ROCE v2, we ensure high-speed, low-latency communication between nodes, optimizing the training workflow. Through rigorous testing and analysis, we aim to uncover insights into the system's operation and scalability potential in real-world applications.

## Prerequisites

- Servers with A100 or H100 GPUs
- Optimized Ethernet Fabric for Accelerator Interconnect
- Storage

## Software Dependencies
- Docker
- Slurm with Enroot + Pyxis
- NVIDIA Container Toolkit
- Pytorch
- CUDA

# Running Nvidia BERT MLPerf benchmark
Follow the steps as documented by NVIDIA - https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/README.md


### Steps to launch training on multiple nodes
https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/README.md
