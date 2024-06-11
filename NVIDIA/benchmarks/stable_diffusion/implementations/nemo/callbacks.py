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

# This callback is on top of NeMo CUDA graph callback, with additional
# MLPerf logging and training warmup support.
# Ref: https://github.com/NVIDIA/NeMo/blob/r2.0.0.rc0.beta/nemo/utils/callbacks/cuda_graph.py

import time
import copy
from dataclasses import dataclass
from types import MethodType
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loops.optimization.automatic import ClosureResult
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.utilities import CombinedLoader
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import STEP_OUTPUT
from nemo.utils import logging
from nemo.utils.callbacks import CUDAGraphCallback
import nemo.utils.callbacks.cuda_graph as cuda_graph

from mlperf_logging_utils import constants
from mlperf_logging_utils import mllogger

# Alias to the functions in CUDA graph callback
struct_copy_one = cuda_graph.struct_copy_one
struct_copy_two = cuda_graph.struct_copy_two
zero_grad = cuda_graph.zero_grad
get_training_step = cuda_graph.get_training_step
to_tensor = cuda_graph.to_tensor
register_key = cuda_graph.register_key
update_metrics = cuda_graph.update_metrics


__all__ = ["SDCallback"]


class StaticBufferLoader:
    """Load data to static buffers."""

    def __init__(self, loader, state):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.static = None
        self.state = state

    def __iter__(self):
        # Warmup with fake inputs
        while self.state.current_iteration < self.state.warmup_iterations:
            yield from self.copy_and_yield(self.state.warmup_inputs)
        for inputs in self.loader:
            yield from self.copy_and_yield(inputs)

    def copy_and_yield(self, inputs):
        if self.static is None:
            with torch.cuda.stream(self.stream):
                self.static = struct_copy_one(inputs)

        with torch.cuda.stream(self.stream):
            struct_copy_two(self.static, inputs)
        torch.cuda.current_stream().wait_stream(self.stream)
        yield self.static

    def __len__(self):
        return len(self.loader)


def get_lr_func(state):
    def get_lr(lr_scheduler):
        if not hasattr(lr_scheduler, "static_lrs"):
            lrs = lr_scheduler.__orig_get_lr__()
            lr_scheduler.static_lrs = lrs
            lr_scheduler.initial_lrs = copy.deepcopy(lrs)

        # Do not step the LR scheduler during warmup
        if state.current_iteration < state.warmup_iterations:
            lrs = 0.0
        elif state.current_iteration == state.warmup_iterations:
            lrs = lr_scheduler.initial_lrs
            # Reset LR scheduler to initial state after warmup
            lr_scheduler.optimizer._step_count = 1
            lr_scheduler._step_count = 1
            lr_scheduler.last_epoch = 1
        else:
            lrs = lr_scheduler.__orig_get_lr__()
        for i in range(len(lr_scheduler.static_lrs)):
            lr_scheduler.static_lrs[i].copy_(lrs if isinstance(lrs, float) else lrs[i])
        return lr_scheduler.static_lrs
    return get_lr


def get_optimizer_step(state):
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None,) -> None:
        # Do not step the actual optimizer during warmup
        # LR for warmup is handled by LR scheduler
        if state.warmup_iterations > 0 and state.current_iteration == 0:
            for group in optimizer.param_groups:
                group["betas_"] = group["betas"]
                group["bias_correction_"] = group["bias_correction"]
                group["betas"] = [1.0, 1.0]
                group["bias_correction"] = False

        # Not all optimizer supports set_to_none.
        if not hasattr(optimizer, "support_set_to_none"):
            optimizer.support_set_to_none = is_param_in_hook_signature(
                optimizer.zero_grad, "set_to_none", explicit=True
            )
        if optimizer.support_set_to_none:
            zero_grad_kwargs = {"set_to_none": True}
        else:
            zero_grad_kwargs = {}

        if 0 <= state.current_iteration < state.capture_iteration or state.capture_iteration < 0:
            state.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(state.stream):
                optimizer.zero_grad(**zero_grad_kwargs)
                self.__orig_optimizer_step__(
                    epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure,
                )
            torch.cuda.current_stream().wait_stream(state.stream)

        if state.current_iteration == state.capture_iteration:
            torch.cuda.synchronize()

            # Recover optimizer configs changed by warmup
            for group in optimizer.param_groups:
                group["betas"] = group["betas_"]
                group["bias_correction"] = group["bias_correction_"]
                del group["betas_"]
                del group["bias_correction_"]
                if "step" in group:
                    # MegatronFusedAdam
                    if isinstance(group["step"], torch.Tensor):
                        group["step"].fill_(1)
                    else:
                        group["step"] = 1
            if hasattr(optimizer, "state") and isinstance(optimizer.state["step"], torch.Tensor):
                # DistributedFusedAdam
                optimizer.state["step"].fill_(1)

            # Sleep for one second to let environment stable
            time.sleep(1)
            logging.info("CUDAGraphCallback: capturing CUDA graph for module %s.", self.__class__.__name__)
            with torch.cuda.graph(state.graph, stream=state.stream, capture_error_mode="global"):
                # PyTorch CUDA graph doc for whole-network capturing mentions:
                #
                #   Sets grads to None before capture, so backward() will create
                #   .grad attributes with allocations from the graph's private pool
                #
                # But it's not necessary, and it can lead to CUDA kernels inside
                # `zero_grad()` being not captured.
                optimizer.zero_grad(**zero_grad_kwargs)
                self.__orig_optimizer_step__(
                    epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure,
                )
            torch.cuda.synchronize()

        # Graph replay and reconstruct missing result
        if state.current_iteration >= state.capture_iteration >= 0:
            state.graph.replay()
            optimizer_closure._result = ClosureResult.from_training_step_output(state.output)

        # If something is not capturable, try to put it there, e.g. `self.log()`.
        if hasattr(self, "non_cuda_graph_capturable"):
            self.non_cuda_graph_capturable()

    return optimizer_step


@dataclass
class CUDAGraphState:
    current_iteration: int = 0
    warmup_iterations: int = 0
    capture_iteration: int = -1  # -1 to disable
    warmup_inputs: Any = None
    stream: torch.cuda.Stream = None
    graph: torch.cuda.CUDAGraph = None
    output: Any = None  # static forward output


class SDCallback(CUDAGraphCallback):
    """PyTorch Lightning callback based on nemo.utils.callbacks.CUDAGraphCallback.

    This callback has following functionalities:

    * Inherit full iteration CUDA graph from NeMo;
    * Warmup the training process with fake data;
    * Integrate MLPerf logging;
    """

    def __init__(
            self,
            logger,
            global_batch_size,
            capture_iteration=-1,
            warmup_iterations=0,
            warmup_inputs=None,
            train_log_interval=5,
            validation_log_interval=1,
            log_tracked_stats=False,
    ):
        super().__init__(capture_iteration=capture_iteration)

        # Change state with warmup support
        self.state = CUDAGraphState(
            warmup_iterations=warmup_iterations,
            capture_iteration=capture_iteration,
            warmup_inputs=warmup_inputs,
        )

        # On top of MLPerfLoggingCallback
        self.logger = mllogger
        self.train_log_interval = train_log_interval
        self.global_batch_size = global_batch_size
        self.validation_log_interval = validation_log_interval
        self.log_tracked_stats = log_tracked_stats

        self.train_batch_start_time = time.perf_counter()
        self.train_batch_start_step = 0

        self.cfg = None

    def save_full_cfg(self, cfg):
        self.cfg = cfg

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit begins."""
        # MLPerfLoggingCallback
        # Callback init is before DDP init, put them here to avoid wrong
        # device placement
        self.summed_loss = torch.zeros(1, device="cuda")
        self.summed_loss_n = 0

        mllogger.event(
            key=constants.GRADIENT_ACCUMULATION_STEPS,
            value=trainer.accumulate_grad_batches,
        )
        mllogger.event(
            key=constants.GLOBAL_BATCH_SIZE, value=self.cfg.model.global_batch_size
        )

        mllogger.event(key=constants.OPT_NAME, value=constants.ADAMW)
        mllogger.event(key=constants.OPT_ADAMW_BETA_1, value=0.9)
        mllogger.event(key=constants.OPT_ADAMW_BETA_2, value=0.999)
        mllogger.event(key=constants.OPT_ADAMW_EPSILON, value=1e-08)
        mllogger.event(key=constants.OPT_ADAMW_WEIGHT_DECAY, value=0.01)

        mllogger.event(key=constants.OPT_BASE_LR, value=self.cfg.model.optim.lr)
        mllogger.event(
            key=constants.OPT_LR_WARMUP_STEPS,
            value=self.cfg.model.optim.sched.warmup_steps,
        )

        mllogger.event(key=constants.TRAIN_SAMPLES, value=6513144)
        mllogger.event(key=constants.EVAL_SAMPLES, value=30000)

        mllogger.mlperf_submission_log(
            benchmark=constants.STABLE_DIFFUSION,
            num_nodes=self.cfg.trainer.num_nodes,
        )

        super().on_fit_start(trainer, pl_module)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends."""
        # MLPerfLoggingCallback
        # No RUN_STOP here because it is after CLIP metric calculation
        # self.logger.end(constants.RUN_STOP, metadata=dict(status=constants.SUCCESS))

        super().on_fit_end(trainer, pl_module)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins.

        Override the StaticBufferLoader and LR scheduler part to support warmup.
        """
        if self.state.capture_iteration < 0:
            return

        # Ensure training dataloader loads data to static buffer
        dataloader = trainer.fit_loop._combined_loader._iterables
        assert isinstance(
            dataloader, torch.utils.data.dataloader.DataLoader
        ), f"Expect Dataloader type but got {type(dataloader)}"
        static_loader = StaticBufferLoader(dataloader, self.state)
        _mode = trainer.fit_loop._combined_loader._mode
        combined_loader = CombinedLoader(static_loader, mode=_mode)
        trainer.fit_loop.__orig_combined_loader__ = trainer.fit_loop._combined_loader
        trainer.fit_loop._combined_loader = combined_loader
        trainer.fit_loop._data_fetcher.setup(trainer.fit_loop._combined_loader)
        iter(trainer.fit_loop._data_fetcher)

        # Warn if `optimizer.zero_grad()` invoked during graph capturing
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, torch.optim.Optimizer), f"Expect Optimizer type but got {type(optimizer)}"
            optimizer.__orig_zero_grad__ = optimizer.zero_grad
            optimizer.zero_grad = MethodType(zero_grad, optimizer)

        # Ensure LR scheduler writes to static buffer
        # We don't include LR scheduler in the full CUDA graph for now since
        # its overhead is very small.
        for config in trainer.lr_scheduler_configs:
            assert isinstance(
                config.scheduler, torch.optim.lr_scheduler._LRScheduler
            ), f"Expect _LRScheduler type but got {type(config.scheduler)}"
            config.scheduler.__orig_get_lr__ = config.scheduler.get_lr
            get_lr = get_lr_func(self.state)
            config.scheduler.get_lr = MethodType(get_lr, config.scheduler)

        # Use smart metrics to avoid syncs
        LightningModule.__orig_to_tensor__ = LightningModule._LightningModule__to_tensor
        LightningModule._LightningModule__to_tensor = to_tensor
        _ResultCollection.__orig_register_key__ = _ResultCollection.register_key
        _ResultCollection.register_key = register_key
        _ResultCollection.__orig_update_metrics__ = _ResultCollection.update_metrics
        _ResultCollection.update_metrics = update_metrics

        # Save model outputs to static buffer for PL states reconstruct
        pl_module.__orig_training_step__ = pl_module.training_step
        training_step = get_training_step(self.state)
        pl_module.training_step = MethodType(training_step, pl_module)

        # Capture CUDA graph from model forward propagation to optimizer step
        pl_module.__orig_optimizer_step__ = pl_module.optimizer_step
        optimizer_step = get_optimizer_step(self.state)
        pl_module.optimizer_step = MethodType(optimizer_step, pl_module)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
        if self.state.current_iteration < self.state.warmup_iterations:
            return
        elif self.state.current_iteration == self.state.warmup_iterations:
            # Refer: pytorch_lightning/loops/training_epoch_loop.py
            if pl_module is None or pl_module.automatic_optimization:
                optim = trainer.fit_loop.epoch_loop.automatic_optimization
                optim.optim_progress.optimizer.step.total.completed = 0
            else:
                optim = trainer.fit_loop.epoch_loop.manual_optimization
                optim.optim_step_progress.total.completed = 0
            assert trainer.global_step == 0, "`trainer.global_step` failed to reset after warmup."

            mllogger.log_init_stop_run_start()

        # MLPerfLoggingCallback
        if trainer.global_step % self.train_log_interval == 0:
            samples_count = pl_module.compute_consumed_samples(trainer.global_step)
            data = {
                "key": constants.BLOCK_START,
                "value": "training_step",
                "metadata": {constants.SAMPLES_COUNT: samples_count},
            }
            self.logger.start(**data)
            self.train_batch_start_time = time.perf_counter()
            self.train_batch_start_step = trainer.global_step

        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        self.state.current_iteration += 1
        if self.state.current_iteration <= self.state.warmup_iterations:
            return

        # MLPerfLoggingCallback
        logs = trainer.callback_metrics
        self.summed_loss += logs["train/loss"]
        self.summed_loss_n += 1

        if (
            trainer.global_step - self.train_batch_start_step
        ) == self.train_log_interval:
            samples_count = pl_module.compute_consumed_samples(trainer.global_step)

            data = {
                "key": constants.BLOCK_STOP,
                "value": "training_step",
                "metadata": {constants.SAMPLES_COUNT: samples_count},
            }
            self.logger.end(**data)

        if (
            trainer.global_step - self.train_batch_start_step
        ) == self.train_log_interval and self.log_tracked_stats:
            throughput = (
                self.global_batch_size
                * self.train_log_interval
                / (time.perf_counter() - self.train_batch_start_time)
            )
            data = {
                "key": "tracked_stats",
                "metadata": {constants.STEP_NUM: trainer.global_step},
                "value": {
                    "throughput": throughput,
                    "loss": self.summed_loss.item() / (self.summed_loss_n + 1e-6),
                    "lr": logs["lr"].item(),
                },
            }
            self.logger.event(**data)

            self.summed_loss.fill_(0)
            self.summed_loss_n = 0

        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
