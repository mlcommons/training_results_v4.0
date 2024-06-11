# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ctypes
import random
from types import MethodType

import torch
import torch._dynamo
import torch.distributed
from checkpoint_tools import MultiprocessCheckpointIO
from megatron.core import parallel_state
from mlperf_logging.mllog import constants
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    MegatronLatentDiffusion,
)
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.core.optim.distributed_adam import create_distribute_within_nodes_pgs
from nemo.utils import logging
from callbacks import SDCallback, mllogger
from nemo.utils.exp_manager import exp_manager
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.plugins.io import TorchCheckpointIO
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)


def l2_promote():
    _libcudart = ctypes.CDLL("libcudart.so")

    # Check what's the device limit for current device, should be 64 by default
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    result = _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))

    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    result = _libcudart.cudaDeviceSetLimit(
        ctypes.c_int(0x05),
        ctypes.c_int(128),
    )

    # Get the device limit again, should be 128
    result = _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    logging.info("L2 promotion: %d B", pValue[0])
    return result


@hydra_runner()
def main(cfg) -> None:
    mllogger.start(key=constants.INIT_START)

    if cfg.model.get("inductor", False):
        # Disable dynamic shape
        torch._dynamo.config.dynamic_shapes = False
        torch._dynamo.config.automatic_dynamic_shapes = False

    # Promote L2 fetch to 128 bytes
    l2_promote()

    seed = random.SystemRandom().randint(0, 2**32 - 1)
    mllogger.event(key=constants.SEED, value=seed)
    seed_everything(seed)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
    with_distributed_adam = cfg.model.optim.get("name") == "distributed_fused_adam"

    torch.backends.cuda.matmul.allow_tf32 = True

    plugins = []
    strategy = NLPDDPStrategy(
        # we don't use DDP for async grad allreduce
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    if cfg.model.precision in [16, "bf16", "16-mixed", "bf16-mixed"]:
        scaler = None
        if cfg.model.precision in [16, "16-mixed"]:
            scaler = GradScaler(
                init_scale=cfg.model.get("native_amp_init_scale", 65536.0),
                growth_interval=cfg.model.get(
                    "native_amp_growth_interval",
                    1000,
                ),
                hysteresis=cfg.model.get("hysteresis", 2),
            )
        if megatron_amp_O2 and not with_distributed_adam:
            plugins.append(
                MegatronHalfPrecisionPlugin(
                    precision=cfg.model.precision, device="cuda", scaler=scaler
                )
            )
        else:
            plugins.append(
                PipelineMixedPrecisionPlugin(
                    precision=cfg.model.precision, device="cuda", scaler=scaler
                )
            )

    if cfg.get("cluster_type", None) == "BCP":
        plugins.append(TorchElasticEnvironment())


    # Sample inputs for warmup
    assert cfg.model.first_stage_key == "images_moments" and \
        cfg.model.cond_stage_key == "clip_encoded", \
        "Expect images_moments and clip_encoded to warmup."
    n = cfg.model.micro_batch_size
    c = cfg.model.channels
    h = cfg.model.image_size
    d = cfg.model.unet_config.context_dim
    x = torch.randn((n, 2*c, h, h), dtype=torch.float32, device="cpu")
    cc = torch.randn((n, 77, d), dtype=torch.float32, device="cpu")
    inputs = {"images_moments": x, "clip_encoded": cc}

    callbacks = []
    cb = SDCallback(
        capture_iteration=cfg.model.capture_cudagraph_iters,
        warmup_iterations=cfg.model.capture_cudagraph_iters+1,
        warmup_inputs=inputs,
        logger=mllogger,
        train_log_interval=100,
        global_batch_size=cfg.model.global_batch_size,
        log_tracked_stats=False,
    )
    cb.save_full_cfg(cfg)
    callbacks.append(cb)

    checkpoint_io = MultiprocessCheckpointIO(
        checkpoint_io=TorchCheckpointIO(),
    )
    plugins.append(checkpoint_io)

    trainer = Trainer(
        plugins=plugins,
        strategy=strategy,
        callbacks=callbacks,
        enable_progress_bar=False,
        **cfg.trainer,
    )

    exp_manager(trainer, cfg.exp_manager)

    """
    # Update resume from checkpoint found by exp_manager
    if cfg.model.get("resume_from_checkpoint") is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = \
            trainer._checkpoint_connector.resume_from_checkpoint_fit_path
    logging.info(
        "Resuming training from checkpoint: %s",
        resume_from_checkpoint,
    )
    """

    trainer._checkpoint_connector = _CheckpointConnector(trainer)

    # Re-order communicator
    def get_setup_distributed_func(cfg):
        def setup_distributed_func(
            self, global_rank: int = None, world_size: int = None
        ) -> None:
            self._orig_setup_distributed(global_rank, world_size)

            group = parallel_state.get_data_parallel_group()
            if cfg.model.optim.get("name") == "distributed_fused_adam":
                if cfg.model.optim.get("distribute_within_nodes", False):
                    dist_pg_infos = create_distribute_within_nodes_pgs()
                    if dist_pg_infos:
                        group = dist_pg_infos['redundant_process_group']

            dummy = torch.randn(64, device="cuda", dtype=torch.float16)
            logging.info(
                "Warmup allreduce with communicator at %x, size %d",
                id(group),
                group.size(),
            )
            for _ in range(20):
                torch.distributed.all_reduce(dummy, group=group)

            # Prevent following communicators to lock the tree
            os.environ["NCCL_SHARP_DISABLE"] = "1"
            os.environ["NCCL_COLLNET_ENABLE"] = "0"

        return setup_distributed_func

    # Re-order communicator
    setup_distributed = get_setup_distributed_func(cfg)
    trainer.strategy._orig_setup_distributed = trainer.strategy.setup_distributed
    trainer.strategy.setup_distributed = MethodType(setup_distributed, trainer.strategy)

    model = MegatronLatentDiffusion(cfg.model, trainer)
    checkpoint_io.setup(model.state_dict())

    # Put on a side stream to meet the CUDA graph requirements
    with torch.cuda.stream(torch.cuda.Stream()):
        trainer.fit(model)

    # Since we created checkpoint in a new process, we wait to make sure the
    # last checkpoint is saved
    checkpoint_io.teardown()

    trainer.strategy.barrier()


if __name__ == "__main__":
    main()
