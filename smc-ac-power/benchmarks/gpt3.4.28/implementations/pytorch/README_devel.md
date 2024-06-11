For general benchmark information please refer to the [official README](README.md).

### How to run the benchmark

Execution path overview:
1. `launch-mlperf-benchmark.sh` calls `sbatch ... run.sub`
2. `run.sub` calls `srun ... run_and_time.sh`
3. `run_and_time.sh` calls `python megatron_gpt_pretraining_custom.py ...`

Examples below use a "smoke test" single-node config 1x8x32x8x1.
Actual run must span at least 8n (see [official README](README.md) for configs naming).

#### Run with utils scripts (recommended)
1. Clone the [optimized repo](https://gitlab-master.nvidia.com/dl/mlperf/optimized) on the cluster to OPTIMIZED_ROOT.
2. Clone the [mlperf_utils repo](https://gitlab-master.nvidia.com/dl/mlperf/mlperf_utils) to UTILS_ROOT.
3. On the cluster login node run:
```bash
cd $OPTIMIZED_ROOT/large_language_model/pytorch
export CONT=...  # e.g. v3.0 submission container: gitlab-master.nvidia.com/dl/mlperf/optimized:large_language_model.pytorch.8293937
export LOGDIR=... # directory to store the results
export PARTITION=...  # `luna` for selene, `viking` for preos
$UTILS_ROOT/launch-mlperf-benchmark.sh --container=$CONT --config=1x8x32x8x1 --nexp=1 --npar=1 --runsub=run.sub --partition=$PARTITION
```

#### Run directly with sbatch
```bash
export CONT=...
export LOGDIR=...
export PARTITION=...

source config_data_selene.sh  # or config_data_preos.sh
source config_DGXA100_1x8x32x8x1.sh  # or config_DGXH100_1x8x32x8x1.sh for preos

sbatch \
    -J mlperf-training:llm_optim \
    -A mlperf \
    -N $DGXNNODES \
    -t $WALLTIME \
    -p $PARTITION \
    run.sub
```

#### Run interactively
1. Allocate nodes, e.g. `salloc -J mlperf-training:llm_optim_inter -A mlperf -N 1 -t 1:00:00 -p interactive bash`
2. Inside the slurm sessions setup env vars:
```bash
export CONT=...
export LOGDIR=...
export NEXP=1
export NPAR=1

source config_data_selene.sh  # or config_data_preos.sh
source config_DGXA100_1x8x32x8x1.sh  # or config_DGXH100_1x8x32x8x1.sh for preos
```
3. Start training
```bash
bash run.sub
```

### Entrypoints
The Optimized/large_langauge_model is meant to run NeMo-Megatron trainings for Mlperf. The main script for that is megatron_gpt_pretraining_mlperf.py that is a copy of NeMo/examples/nlp/langauge_model/megatron_gpt_pretraining.py but contains additional mlperf logging staff and takes appropriate config parameters.

The training is parametrized similarily as in other benchmarks with two bash scripts (one sourcing the other).
- config_DGXH100_xx.sh
   here you set up specific params for that particular config.
- config_common.sh
   here training params common to all configs are defined e.g. datapaths, data blends, spm models etc.

The training is run within a container that is built according to Dockerfile. The general idea is to use a production NeMo container as base, and add mlperf requirements upon it e.g. mlperf_logging library and dependencies. 
It might happen however that we need to test incomming changes that are not yet pushed to production NeMo container. In that case, we have to update the base container in the Dockerfile to point to specific NeMo container build manually.

### Configuration files

1) Configuration of jobs take place via environment variables
2) NeMo/model specific configuration makes use of hydra configuraion files at `conf/`. The main configuration file is `megatron_gpt_config_custom.yaml`. It defines the order of config assembly.
The fields precedence is similar to the regular dictionray - the last occurence wins.
3) We base the config on NeMo's `examples/nlp/language_modeling/conf/megatron_gpt_config.yaml` file.
4) All the mlperf specific arguments and overrides are kept in `conf/custom.yaml` file.
5) If one configuration field is solely dependent on others, then all the logic for derivation of this field is placed in unresolved hydra config.
6) Overriging precedence. The default overrides for each configuration are in `config_DGX*.sh` files. If you want to override options defined in these files, you can export corresponging env variable or modify hydra config. Hydra config takes precendence over environment variables. Lastly hydra CLI overrides are taken into consideration overriding options defined in hydra config files.
7) `config.yaml` is defined in hydra's syntax, allowing us to pull variables from the environment. For example: `${oc.env:VAR,DEFULT}` takes value of VAR if VAR is defined, otherwise it takes DEFAULT value. Hydra treats all environment variables as strings, thus we need proper casting. For this purpose `${oc.decode:STR}` resolver is used in config files.
8) Resolved hydra config is a source of truth about a configuration. Hydra provides debug utility to view unresolved config (where all the overrides are present but before field interpolations) with `-c job` option (append it as an argument to hydra's entrypoint script). This allows to trace which variables are dependent on others and which are pulled from the environment. In case you want to view resolved config use `-c job --resolve`.
9) Any CLI override can be achieved by appending it to the `EXTRA_ARGS` environment variable. For example to override the data prefix:
```bash
BLEND="[0.1,/preproc_data/c4_en_validation_c4_spm_text_document]"
VALID_C4="/preproc_data/c4_en_validation_subset_c4_spm_text_document"
DATA_PREFIX="'{train: $BLEND, validation:[$VALID_C4], test: [$VALID_C4]}'"
# clear previous field and override it with new. Regular adding will cause dictionary merging
export EXTRA_ARGS="${EXTRA_ARGS} ~model.data.data_prefix +model.data.data_prefix=${DATA_PREFIX}"
```
10) If you want to save your data prefix for later use, create a configuration file under `conf/data_prefix` and add override:
```bash
export EXTRA_ARGS="${EXTRA_ARGS} ~model.data.data_prefix +data_prefix@model.data.data_prefix=your_config"
```
11) The point 10 shown crude way to do a subtree substitution, when the default subconfig is hardcoded in the config. If you want to do a subree substitution with no default subconfig or you don't want to deal with clearing residuals after the default subconfig. For example you want to swich one TP config with another there are 3 steps to follow:

   - You have to enable this option by marking field substitutable. You can do this by adding an item to the config's defaults list. This will look for the config file in `conf/tp_overlap` directory and place it under `model.ub_tp_comm_overlap_cfg` path in the final config:
   ```yaml
   defaults:
      - optional tp_overlap@model.ub_tp_comm_overlap_cfg:
   ```
   - Place your config under `conf/tp_overlap` directory

   - Then you can substitute the whole subtree by overriding it via `EXTRA_ARGS` env variable:

   ```bash
   export EXTRA_ARGS="${EXTRA_ARGS} tp_overlap@model.ub_tp_comm_overlap_cfg=my_tp_config"
   ```

#### Top down execution path

### Training spanning multiple slurm jobs
In order to share checkpoints between different runs (e.g. of a dryrun):
1) set ENABLE_RERUNS=1 to enable checkpointing
2) set SHARE_RERUNS=1 so that the checkpoints subdirectory is the same for all runs
3) set the same LOGDIR in all runs
4) run training with `run.sub` or set NEMO_RESULTS_SUBDIR manually to a fixed value
5) run dependent slurm job
6) set the same SEED in all runs

The simplest way to achieve point 5) is setting the `--dependency=singleton` [flag](https://slurm.schedmd.com/sbatch.html#OPT_singleton) of the `sbatch` command and making sure the relevant slurm jobs have identical names.

Points 1), 2), 6) can be achieved by sourcing `config_shared_sryruns.sh` file.

## Other
### MLCommons artifacts
We provide publicly accessible LLM dataset and checkpoints on MLCommons hosted AWS bucket `s3://mlcommons-training-wg-s3/gpt3/megatron-lm/`.
Access and detailed instructions are provided by MLCommons sysadmins, here are some internal tips:
1. A convenient way of managing S3 artifacts is with AWS CLI. To avoid sudo permissions on Selene, leverage `--install-dir` and `--bin-dir` flags during `./aws/install` installation.
1. Avoid compressed archives (e.g. `-z` for `tar`) for large data, e.g. the checkpoints.
1. Uploading a checkpoint requires archiving checkpoint directory with multiple files, which doesn't play well with Selene filesystem.
   1. Use compute nodes for archiving if possible for non-limited CPU performance and parallelism.
   1. `tar` has higher read overhead than e.g. `cat` or simple `cp`. Consider pre-copying (or preloading to cache) checkpoint files from lustre to local disk prior to archiving.
      1. Local disk might have limited space (e.g. 1TB) so some buffering might be needed. An (untested!) idea for dealing with that:
         1. Create a [named pipe](https://en.wikipedia.org/wiki/Named_pipe), e.g. `tarfiles`
         1. Tar files from the pipe: `tar ... --from-file=tarfiles`
         1. For each checkpoint file: copy file to local disk and send its name to `tarfiles`
         1. Somehow deal with deleting local files that are already archived (non-trivial, no good idea for that yet)
