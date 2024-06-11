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
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional

import torch
from collections import OrderedDict
from lightning_fabric.plugins import CheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO


def recursive_cast_cpu(struct):
    if isinstance(struct, torch.Tensor):
        return struct.to(device="cpu", non_blocking=True)
    if isinstance(struct, dict):
        return {k: recursive_cast_cpu(v) for k, v in struct.items()}
    if isinstance(struct, list):
        return [recursive_cast_cpu(v) for v in struct]
    if isinstance(struct, tuple):
        return tuple(recursive_cast_cpu(v) for v in struct)
    return struct


class MultiprocessCheckpointIO(_WrappingCheckpointIO):
    """``MultiprocessCheckpointIO`` enables saving the checkpoints asynchronously in a process.

    .. warning::

        This is currently an experimental plugin/feature and API changes are to be expected.

    Args:
        checkpoint_io: A checkpoint IO plugin that is used as the basis for async checkpointing.
    """

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None) -> None:
        super().__init__(checkpoint_io)

        self._executor = ProcessPoolExecutor(max_workers=1)
        self._process = None

    def setup(self, state_dict: "OrderedDict") -> None:
        self._state_dict = recursive_cast_cpu(state_dict)
        for k, v in self._state_dict.items():
            self._state_dict[k] = v.pin_memory()

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path,
        storage_options: Optional[Any] = None,
    ) -> None:
        """Uses the ``ThreadPoolExecutor`` to save the checkpoints using the
        base ``checkpoint_io``.
        """

        # We don't need optimizer_states since only eval run in future
        if "optimizer_states" in checkpoint:
            del checkpoint["optimizer_states"]

        # Move checkpoint to CPU
        state_dict = checkpoint.pop("state_dict")
        with torch.no_grad():
            checkpoint_cpu = recursive_cast_cpu(checkpoint)
            for k, v in state_dict.items():
                self._state_dict[k].copy_(v, non_blocking=True)
        checkpoint_cpu["state_dict"] = self._state_dict
        torch.cuda.synchronize()

        self._process = self._executor.submit(
            self.checkpoint_io.save_checkpoint, checkpoint_cpu, path, storage_options
        )
        print(f"Saving {path} in the background")

    def teardown(self) -> None:
        """This method is called to close the threads."""
        self._executor.shutdown(wait=True)
