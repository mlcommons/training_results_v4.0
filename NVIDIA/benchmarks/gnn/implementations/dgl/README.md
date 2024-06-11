## Running NVIDIA DGL GNN GATConv MLPerf Benchmark

This file contains the instructions for downloading and preprocessing the dataset, specifying the location of the input files, building the docker image, and running the model. Those steps are the same, irrespective of the hardware platform. However, the actual commands for running the benchmark are different on single node and multiple nodes. Please refer to the last section of this file for those instructions. 

### Dataset downloading and preprocessing

Steps required to launch GNN training on NVIDIA DGX H100s: 

#### Hardware Requirements

- **2TB CPU memory is required**. 
- At least 6TB disk space is required. 
- 80GB GPU memory is strongly recommended. 
- GPU is not needed for preprocessing scripts, but is needed for training. 

#### Software Requirements

- [DGL 24.04-py3 NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/dgl)
- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) (multi-node)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)

#### Container building and running

To build the container and push to a docker registry, we can directly utilize the `Dockerfile` under the current repository: 

```
docker build -f Dockerfile -t <docker/registry>/mlperf-nvidia:gnn_dgl .
docker push <docker/registry>/mlperf-nvidia:gnn_dgl
```

#### Dataset Preparation

To run the docker image during the dataset preparation step, we use the following command:

```
docker run -it --rm \
--network=host --ipc=host \
-v <igbh_dataset>:/data \
-v <data_path>:/converted \
-v <graph_path>:/graph \
<docker/registry>/mlperf-nvidia:gnn_dgl
```

After the dataset preparation step is done, we should have the following files outside the container: 
- `<igbh_dataset>`: this path holds all original IGBH-Full data. 
- `<data_path>`: this path holds all preprocessed IGBH-Full data, including features and graph structures. 
- `<graph_path>`: this path holds only small files, such as graph structures. 

##### Download and verify data

To make it more convenient, we have copied the reference's dataset downloading script as-is from the reference branch to our docker image. To download the dataset: 

```
bash utility/download_igbh_full.sh
```

Please notice that the downloading takes about 2-3 days to complete.

##### Training seed generation

To make it more convenient, we have copied GLT's seed generation script as-is from the reference branch to our docker image, since this script depends only on PyTorch, not GLT. Thus, we can generate the train and validation seeds inside *our* docker image. 

To generate the seeds for training and validation, we run the following code inside the container: 

```
python3 utility/split_seeds.py --dataset_size="full" --path /data/igbh
```

After we have downloaded the dataset and prepared seeds, the original FP32-precision IGBH-Full data should reside under path `/data/igbh`. The directory should look like this: 

```
/data
│
└───igbh
    │
    └───full
        │
        └───processed
            │
            └───train_idx.pt
            └───val_idx.pt
            └───paper
            |   │
            |   └───node_feat.npy
            |       node_label_2K.npy
            |       paper_id_index_mapping.npy
            |       ...
            │
            └───...
```

After training seed generation is complete, we can run the following command to verify the correctness of the generated training seeds: 

```bash
python3 utility/verify_dataset_correctness.py --data_dir /data/igbh --dataset_size full
```

If no errors are reported, then the seeds are generated correctly. 

##### Preprocessing

We have created a preprocessing script to perform FP16 conversion and graph format conversion. To perform such preprocessing, we need to use the `utility/preprocessing.py` file that comes together with *our* docker image. 

Our preprocessing script is set up so that we copy everything to a new directory and later we can directly train from that directory, not touching the original downloaded dataset at all. **This preprocessing step does NOT require GPU**. 

Please refer to the instruction above for how to build and run the container. Inside the container, we run: 

```
python3 utility/preprocessing.py 
    --data_dir /data/igbh
    --convert_dir /converted
    --precision float16 # we can specify float32 for FP32 features, but it is not recommended
    --size full
    --shuffle paper author conference journal fos institute
    --graph_storage_copy /graph
    --seed 0
    --concat_features
```

After this script is done, you should see a lot of files under `/converted/full` directory and `/graph/full` directory inside the container. Most importantly, a `config.yml` file should exist under `/converted/full`, which is crucial for subsequent data loading. 

The above script should take about 3hr 50min to finish, measured on a single DGX H100 node. 

#### Optional: FP8 conversion

We have created a separate script to convert the dataset further from FP16 to FP8, using NVIDIA's TransformerEngine. Please notice that **this conversion script requires 1 GPU with 80GB GPU Memory**. If GPU is present in the docker host system, we can launch a container with GPU by adding **`--gpus all`** argument to the container launch command: 

```
docker run -it --rm \
--network=host --ipc=host \
-v <igbh_dataset>:/data \
-v <data_path>:/converted \
-v <graph_path>:/graph \
--gpus all \
<docker/registry>/mlperf-nvidia:gnn_dgl
```

To further perform such conversion, we run the following command inside the container with GPU mounted: 

```
python3 utility/fp8_conversion.py 
    --data_dir /converted/full
    --fp8_format e4m3
    --scale 1.0
```

After this script is done, the FP16 features that are previously inside `/converted` are now replaced with FP8 features, and we can subsequently enable training with FP8 features using flag `FP8_EMBEDDING=1`. 

**Note**: Please notice that FP8 features are set as the default option. To use FP16 features, we need to explicitly additionally set `FP8_EMBEDDING=0` environment variable when launching runs. 


##### Optional: Patching training seeds

If the training seeds are incorrectly generated by any means (incorrect command line arguments, incorrect random seeds, etc), and needs to be patched, we do not need to run the full preprocessing workflow again to patch them, and can instead follow the following procedure. 

Assuming that we're still in the same container, with: 

- Original IGBH dataset path mounted on `/data`
- Preprocessed features mounted under `/converted`
- Preprocessed Graph copy mounted under `/graph`

Then, we should see `train_idx.pt` and `val_idx.pt` under the following paths: 

- `/data/igbh/full/processed`
- `/converted/full`
- `/graph/full`

Inside this container, we run again the `split_seeds.py`. Attaching an example command with **all correct arguments**: 

```bash
python3 utility/split_seeds.py \
--dataset_size full \
--path /data/igbh \
--validation_frac 0.005 \
--num_classes 2983 \
--random_seed 42
```

Once this is done, we can see that the `train_idx.pt` and `val_idx.pt` files are updated under container path `/data/igbh/full/processed`. Either run through the preprocessing step again to regenerate the final `train_idx.pt` and `val_idx.pt` or utilize `reshuffle_indices.py` to directly shuffle the indices based on the shuffle map generated previously in the preprocessing step.

```bash
python3 reshuffle_indices.py \
--path_to_indices /data/igbh/full/processed/val_idx.pt \
--path_to_shuffle_map /converted/full/paper_shuffle_map.pt \
--path_to_output_indices val_idx_shuffled.pt
```

Then copy `val_idx_shuffled.pt` to `/converted` and `/graph` and rename it to `val_idx.pt`.

##### Using the preprocessed data path in the training script

As is specified before: 

> After the dataset preparation step is done, we should have the following files outside the container: 
> - `<igbh_dataset>`: this path holds all original IGBH-Full data. 
> - `<data_path>`: this path holds all preprocessed IGBH-Full data, including features and graph structures. 
> - `<graph_path>`: this path holds only small files, such as graph structures. 

To use the preprocessed data path in `train.py`, we need to set the following environment variables:

```
DATA_DIR=<data_path>/full
GRAPH_DIR=<graph_path>/full
FP8_EMBEDDING=1 # depending on whether FP8 embeddings are created. 
```

### Training

#### Steps to launch training on a single node

##### NVIDIA DGX H100 (single-node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX H100
single-node submission are in the `config_DGXH100_1x8x2048.sh` script.

To launch training on a single node with a Slurm cluster, run: 

```
source config_DGXH100_1x8x2048.sh
CONT=<docker/registry>/mlperf-nvidia:gnn_dgl LOGDIR=<path/to/output/dir> DATA_DIR=<data_path>/full GRAPH_DIR=<graph_path>/full FP8_EMBEDDING=0 sbatch -N 1 run.sub
```

Note that this benchmark requires a lot of CPU memory usage on a single node. To achieve optimal performance, 2TB CPU memory is required. 

If we want to use FP8 features instead, then we can run the following command: 

```
source config_DGXH100_1x8x2048.sh
CONT=<docker/registry>/mlperf-nvidia:gnn_dgl LOGDIR=<path/to/output/dir> DATA_DIR=<data_path>/full GRAPH_DIR=<graph_path>/full FP8_EMBEDDING=1 sbatch -N 1 run.sub
```

#### Steps to launch training on multiple nodes

##### NVIDIA DGX H100 (multi-node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX H100 multi-node submission is in the `config_DGXH100_8x8x1024.sh` script. 

To launch training on multiple nodes with a Slurm cluster, run: 

```
source config_DGXH100_<config>.sh
CONT=<docker/registry>/mlperf-nvidia:gnn_dgl LOGDIR=<path/to/output/dir> DATA_DIR=<data_path>/full GRAPH_DIR=<graph_path>/full sbatch -N $DGXNNODES run.sub
```
