# 1. Problem 
This benchmark represents a LLama2-70B LoRA finetuning on the [GovReport](https://gov-report-data.github.io/) dataset.

# 2. Dataset/Environment

GovReport is a dataset for long document summarization that consists of reports written by government research agencies. Dataset hosted on the MLPerf drive is already tokenized and packed so that each sequence has length 8192.

Download dataset from the MLPerf drive:
- [Train dataset](https://drive.google.com/file/d/1-JgY1mEafcJ7qhggt6UR3OEKAciIPd5s/view?usp=sharing)
- [Validation dataset](https://drive.google.com/file/d/1jrm6Lacrq49AYv0uB_Qy22xRmfPixQvs/view?usp=sharing)
or use `scripts/download_dataset.py` script in the container with mounted local directory to store the dataset.

Launch container with mounted directory for dataset:
```
docker run -it --gpus all -v <path/to/dataset>:/data <docker/registry>/mlperf-nvidia:lora-pytorch --pty bash
```
to execute `scripts/download_dataset.py` run:
```
python scripts/download_dataset.py
```
Convert dataset to numpy format:
```
python scripts/convert_dataset.py
```
After conversion you should see the following files in the `/data` directory:
```
train.npy  validation.npy
```
# 3. Model

Model hosted on MLPerf drive is the LLama2-70B with fused QKV. You will need 270GB to download and convert the model.

Download model from the MLPerf drive:
- [Model](https://drive.google.com/drive/folders/1sTeuxkPhwkNPKIPFnOLIYCcK53oB3Ypc?usp=sharing)
or use `scripts/download_model.py` script in the container with mounted local directory to store the model.

Launch container with mounted directory for model:
```
docker run -it --gpus all -v <path/to/model>:/model <docker/registry>/mlperf-nvidia:lora-pytorch --pty bash
```
to execute `scripts/download_model.py` run:
```
python scripts/download_model.py
```
Convert model to NeMo format (it will take around 1hour):
```
python scripts/convert_model.py --input_name_or_path=/model --output_path=/model/llama2-70b.nemo --hparams_file scripts/megatron_llama_config.yaml
```
Untar:
```
cd /model && find . -type f ! -name 'llama2-70b.nemo' -exec rm -f {} + && tar -xvf llama2-70b.nemo
```
After conversion you should see the following files in the `/model` directory:
```
ce72295014db4feea6907156602f2f39_tokenizer.model  llama2-70b.nemo  model_config.yaml  model_weights
```

# 4. Directions
### Steps to configure machine
Launch configuration and system-specific hyperparameters for the appropriate
NVIDIA DGX submission are in the `configs/config_DGXH100_*.sh` scripts.
Data related variables (DATADIR, MODEL) are not covered in the config files and must be set separately.

### Steps to launch training
Build the container and push to a docker registry:
```
docker build --pull -t <docker/registry>/mlperf-nvidia:lora-pytorch .
docker push <docker/registry>/mlperf-nvidia:lora-pytorch
```
### Launch the training:
```
source configs/config_DGXH100_1x8x8x4x2_fp8.sh  # use appropriate config
CONT=<container/name> LOGDIR=<path/to/output/dir> DATADIR=<path/to/dataset> MODEL=<path/to/model> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

# 5. Quality
### Quality metric
Cross entropy loss
### Quality target
0.925
### Evaluation frequency
Every 384 samples
### Evaluation thoroughness
Evaluation on the validation subset that consists of 173 examples