import os
import torch
import torch.distributed as dist
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

def get_submission_info(platform, train_dataset_length,eval_dataset_length,lora_alpha):
    return  {
            "submission_benchmark": "llama2_70b_lora",
            "submission_division": "closed",
            "submission_org": "Juniper Networks",
            "submission_platform": platform,
            "submission_status": "onprem",
            "train_dataset_length": train_dataset_length,
            "eval_dataset_length": eval_dataset_length,
            "lora_alpha": lora_alpha
        }
def log_mlperf_submission_info(seed, 
                               train_dataset_length,
                               eval_dataset_length,
                               mllogger, 
                               args):
    gbs=args.per_device_train_batch_size * args.gradient_accumulation_steps * int(os.getenv("WORLD_SIZE", 1))
    submission_info = get_submission_info(args.submission_platform,
                                          train_dataset_length,
                                          eval_dataset_length,
                                          lora_alpha=args.lora_alpha)
    mllogger.event(
        key=constants.CACHE_CLEAR, value="True",
    )
    mllogger.event(
        key=constants.SUBMISSION_BENCHMARK,
        value=submission_info["submission_benchmark"],
    )
    mllogger.event(
        key=constants.SUBMISSION_DIVISION,
        value=submission_info["submission_division"],
    )
    mllogger.event(
        key=constants.SUBMISSION_ORG, value=submission_info["submission_org"]
    )
    mllogger.event(
        key=constants.SUBMISSION_PLATFORM,
        value=submission_info["submission_platform"],
    )
    mllogger.event(
        key=constants.SUBMISSION_STATUS,
        value=submission_info["submission_status"],
    )
    mllogger.event(
        key=constants.GLOBAL_BATCH_SIZE,
        value=gbs,
    )
    mllogger.event(
        key=constants.TRAIN_SAMPLES,
        value=submission_info["train_dataset_length"],
    )
    mllogger.event(
        key=constants.EVAL_SAMPLES,
        value=submission_info["eval_dataset_length"],
    )
    mllogger.event(key=constants.SEED, value=seed)
    mllogger.event(key=constants.OPT_LR_WARMUP_FACTOR, value=args.warmup_ratio)
    mllogger.event(key=constants.OPT_LR_TRAINING_STEPS, value=args.max_steps)
    mllogger.event(key=constants.OPT_ADAMW_WEIGHT_DECAY, value=args.weight_decay)
    mllogger.event(key=constants.OPT_GRADIENT_CLIP_NORM, value=args.max_grad_norm)
    mllogger.event(key=constants.OPT_BASE_LR, value=args.learning_rate)
    mllogger.event(key=constants.LORA_ALPHA, value=submission_info["lora_alpha"])
    mllogger.event(key='lora_rank', value=16)
    mllogger.event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=args.gradient_accumulation_steps)
    mllogger.end(key=constants.INIT_STOP, value="")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def barrier():
    if not is_dist_avail_and_initialized():
        return
    torch.distributed.barrier()


class LoraLogger:
    def __init__(self, target_eval_loss=None, filename=None, default_stack_offset=2):
        self.mllogger = mllog.get_mllogger()
        mllog.config(
            default_stack_offset=default_stack_offset,
            filename=(
                filename or os.getenv("COMPLIANCE_FILE") or "mlperf_compliance.log"
            ),
            root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))),
        )
        self.target_eval_loss = target_eval_loss

    @property
    def rank(self):
        return get_rank()

    def event(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank == 0 if log_rank is None else self.rank == log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.event(key=key, value=value, metadata=metadata)

    def start(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank == 0 if log_rank is None else self.rank == log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.start(key=key, value=value, metadata=metadata)

    def end(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank == 0 if log_rank is None else self.rank == log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.end(key=key, value=value, metadata=metadata)


class MLPerfCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, logger):
        super().__init__()
        self.mllogger = logger
    
    def on_init_end(self, args, state, control, **kwargs):
        self.mllogger.start(key=constants.INIT_START, value="")
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.mllogger.start(key=constants.RUN_START, value="")
    
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.gbs=args.per_device_train_batch_size * args.gradient_accumulation_steps * int(os.getenv("WORLD_SIZE", 1))
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if (
            state.global_step % (state.logging_steps) == 0
            and state.global_step > 0
            and not state.global_step % (state.eval_steps) == 0
        ):
            self.mllogger.event(
                "train_loss",
                value=state.log_history[-1]["loss"],
                metadata={"samples_count": state.log_history[-1]["step"]*self.gbs},
            )
            control.should_log = True

        if state.global_step % (state.eval_steps) == 0 and state.global_step > args.eval_delay:
            self.mllogger.end(
                constants.BLOCK_STOP,
                value="",
                metadata={"samples_count": state.log_history[-1]["step"]*self.gbs},
            )
            self.mllogger.event(
                constants.EVAL_ACCURACY,
                value=state.log_history[-1]["eval_loss"],
                metadata={"samples_count": state.log_history[-1]["step"]*self.gbs},
            )
            self.mllogger.start(
                constants.BLOCK_START,
                value="",
                metadata={"samples_count": state.log_history[-1]["step"]},
            )            
            control.should_log = True
        eval_loss_list = [
            sl["eval_loss"] for sl in state.log_history if "eval_loss" in sl
        ]
        if eval_loss_list and eval_loss_list[-1] <= self.mllogger.target_eval_loss:
            control.should_training_stop = True
            self.mllogger.end(
                constants.RUN_STOP,
                value=eval_loss_list[-1],
                metadata={
                    "samples_count": state.log_history[-1]["step"]*self.gbs,
                    "status": "success",
                },
            )
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
            self.mllogger.end(
                constants.RUN_STOP,
                value=eval_loss_list[-1],
                metadata={"samples_count": state.log_history[-1]["step"]*self.gbs, "status": "fail"},
            )

        return control