# Llama2 70B LoRA benchmark

This submission aims to provide an implementation for Fine-Tuning a Llama2_70B model using Huggingface Trainer, PEFT and Dataloader

## Setup

This benchmark can be executed either on a single node or multiple-nodes. Find the setup instructions for these two approaches below.

### Single-node

Install python and pip in your environment. Run the following command:

```bash
pip install -r requirements/requirements.txt
```

### Multi-node

Slurm is used for running this benchmark on a multi-node environment.

- Create a Docker image using the Dockerfile under docker/.
- Import Docker image to enroot using the following command:

```bash
enroot import dockerd://{dockerImageName}
```

## Download Data and Model

MLCommons hosts the model for download exclusively by MLCommons Members.
You must first agree to the [confidentiality notice](https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform), then follow the provided link to a directory containing the file "CLI Download Instructions" with instructions to use Rclone to download the model and data. Follow steps 1-3 to install and activate Rclone. Finally, download the model to the desired download directory (default ./models):

```bash
mkdir models
cd models
rclone copy mlc-llama2:Llama2-70b-fused-qkv-mlperf ./Llama2-70b-fused-qkv-mlperf -P
```

Similarly download the data to the desired download directory (default ./dataset):

```bash
mkdir dataset
cd dataset
rclone copy mlc-llama2:training/scrolls_gov_report_8k ./scrolls_gov_report_8k -P
```

## Run on Single Node

```bash
run/run_single_node.sh
```

## Run Multi-Node (SLURM based)

```bash
sbatch run/run.sub
```
