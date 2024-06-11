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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed
import torch.utils.data
from inception_models import TFInceptionV3
from infer_and_eval_tools import (
    CLIPEncoder,
    SimplePILDataset,
    calculate_frechet_distance,
    get_fid_activations,
)
from mlperf_logging_utils import (
    constants,
    extract_step_from_ckpt_name,
    extract_consumed_samples_from_ckpt_name,
    extract_timestamp_from_ckpt_name,
    mllogger,
)
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    MegatronLatentDiffusion,
)
from nemo.collections.multimodal.parts.stable_diffusion.pipeline import pipeline
from nemo.collections.multimodal.parts.utils import (
    setup_trainer_and_model_for_inference,
)
from nemo.core.config import hydra_runner
from PIL import Image

TARGET_FID = 90.0
TARGET_CLIP = 0.15


def test_reduce():
    # All reduce test
    world_size = torch.distributed.get_world_size()

    vector = torch.ones(2, dtype=torch.float32).cuda()
    target = torch.ones(2, dtype=torch.float32).cuda() * world_size
    torch.distributed.all_reduce(vector)
    assert torch.allclose(vector, target)


def test_barrier():
    # Barrier test
    global_id = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for rank in range(world_size):
        if rank == global_id:
            print(f"Rank {rank} before barrier", flush=True)
        torch.distributed.barrier()


def run_eval(cfg, model_ckpt, fid_model, clip_model, local_id, device):
    cfg.model.restore_from_path = model_ckpt
    model_cfg_modifier = (
        ...
    )  # it is only used with .nemo ckpts, so it is not necessary
    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronLatentDiffusion,
        cfg=cfg,
        model_cfg_modifier=model_cfg_modifier,
    )

    # setup_trainer_and_model_for_inference initializes torch.distributed
    assert torch.distributed.is_nccl_available()

    global_id = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"Global ID: {global_id}, local ID: {local_id}, world size: {world_size}")

    test_barrier()
    test_reduce()

    model_ckpt_name = Path(model_ckpt).name
    step_num = extract_step_from_ckpt_name(model_ckpt_name)
    samples_count = extract_consumed_samples_from_ckpt_name(model_ckpt_name)

    mllogger.start(
        key=constants.EVAL_START,
        metadata={constants.SAMPLES_COUNT: samples_count},
        unique=True,
    )
    torch.distributed.barrier()

    model = megatron_diffusion_model.model
    model.to(device).eval()

    # Disable CUDA graph
    model.model.capture_cudagraph_iters = -1
    model.first_stage_model.capture_cudagraph_iters = -1
    model.cond_stage_model.capture_cudagraph_iters = -1

    rng = torch.Generator().manual_seed(cfg.infer.seed)

    # Read prompts from disk from .tsv file
    csv_path = cfg.custom.prompts_csv
    prompts_df = pd.read_csv(csv_path, sep="\t")

    # Sort by "id" column
    prompts_df = prompts_df.sort_values(by=["id"])
    if cfg.custom.num_prompts is not None:
        # Cut to the predefined number of prompts
        prompts_df = prompts_df[: cfg.custom.num_prompts]

    num_prompts_all = len(prompts_df)
    sharded_prompts_df = prompts_df[global_id::world_size]
    num_prompts_sharded = len(sharded_prompts_df)

    print(f"Assigned {num_prompts_sharded} prompts for this worker.")
    sharded_captions = sharded_prompts_df["caption"].tolist()
    cfg.infer.prompts = sharded_captions
    torch.distributed.barrier()

    generated_images = pipeline(model, cfg, rng=rng)
    generated_images = sum(generated_images, [])  # Flatten output
    assert len(generated_images) == num_prompts_sharded

    output_images_path = cfg.infer.save_images_to
    if output_images_path is not None:
        os.makedirs(output_images_path, exist_ok=True)

        output_images_path = os.path.join(output_images_path, model_ckpt_name)
        os.makedirs(output_images_path, exist_ok=True)
        for image, image_id in zip(generated_images, sharded_prompts_df.id):
            if isinstance(image, list):
                assert len(image) == 1
                image = image[0]

            assert isinstance(image, Image.Image)
            output_image_name = os.path.join(
                output_images_path, f"COCO_val2014_{image_id:0>12}.png"
            )
            image.save(os.path.join(output_images_path, output_image_name))

    mllogger.end(
        key=constants.EVAL_STOP,
        metadata={constants.SAMPLES_COUNT: samples_count},
        unique=True,
    )
    torch.distributed.barrier()

    # Inference FINISHED at this point
    # Computing FID activations
    synthetic_dataset = SimplePILDataset(
        pil_images=generated_images, pil_resize=256
    )
    loader_synthetic = torch.utils.data.DataLoader(
        synthetic_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    # Activations have shape [Nk, 2048]
    activations = get_fid_activations(
        model=fid_model,
        data_loader=loader_synthetic,
        progress_bar=global_id == 0,
        device=device,
    )

    activations_dim = activations.shape[1]
    all_activations = torch.zeros(
        size=[num_prompts_all, activations_dim], dtype=torch.float32
    ).to(device)
    all_activations[global_id::world_size] = activations
    torch.distributed.all_reduce(all_activations)
    torch.distributed.barrier()

    coco_stats = cfg.custom.precomputed_coco_activations_path
    with np.load(coco_stats) as f:
        target_m, target_s = f["mu"][:], f["sigma"][:]

    # Computing stats on GPU
    computed_m = torch.mean(all_activations, dim=0).detach().cpu().numpy()
    computed_s = torch.cov(all_activations.T).detach().cpu().numpy()

    FID = calculate_frechet_distance(target_m, target_s, computed_m, computed_s)
    mllogger.event(
        key=constants.EVAL_ACCURACY,
        value=FID,
        metadata={
            constants.SAMPLES_COUNT: samples_count,
            "metric": "FID",
        },
        unique=True,
    )
    torch.distributed.barrier()

    # Computing CLIP activations
    similarities = torch.zeros(size=[num_prompts_all], dtype=torch.float32).to(
        device
    )
    global_ids = range(global_id, num_prompts_all, world_size)

    calculated = 0
    for local_idx, global_idx in enumerate(global_ids):
        caption = sharded_captions[local_idx]
        image = generated_images[local_idx]
        sim = clip_model.get_clip_score(caption, image)
        similarities[global_idx] = sim

        # assert sim > 0.0
        calculated += 1

    torch.distributed.all_reduce(similarities)

    CLIP = torch.mean(similarities).item()
    mllogger.event(
        key=constants.EVAL_ACCURACY,
        value=CLIP,
        metadata={
            constants.SAMPLES_COUNT: samples_count,
            "metric": "CLIP",
        },
        unique=True,
    )
    torch.distributed.barrier()

    return FID, CLIP, model_ckpt_name, step_num


@hydra_runner(config_name="sd_mlperf_infer", config_path="conf")
def main(cfg):
    local_id = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    force_success = bool(os.environ.get("FORCE_SUCCESS_STATUS", False))
    torch.cuda.set_device(local_id)
    device = torch.device("cuda")

    fid_model = (
        TFInceptionV3(weights_path=cfg.custom.fid_weights_path).to(device).eval()
    )
    clip_model = CLIPEncoder(
        clip_version="ViT-H-14", cache_dir=cfg.custom.clip_cache_dir, device=device
    )

    checkpoints = [str(path) for path in Path(cfg.custom.sd_checkpoint_dir).iterdir()]
    checkpoints = sorted(checkpoints, key=extract_step_from_ckpt_name)

    infer_start_step = int(cfg.custom.infer_start_step)
    if infer_start_step:
        for i in range(len(checkpoints)):
            ckpt = checkpoints[0]
            step = extract_step_from_ckpt_name(ckpt)
            if step >= infer_start_step:
                break
            checkpoints.append(checkpoints.pop(0))

    print("Found checkpoints:")
    for ckpt in checkpoints:
        print(ckpt)

    for ckpt_idx, model_ckpt in enumerate(checkpoints):
        FID, CLIP, model_ckpt_name, step_num = run_eval(cfg, model_ckpt, fid_model, clip_model, local_id, device)

        if force_success or FID <= TARGET_FID and CLIP >= TARGET_CLIP:
            success_timestamp = extract_timestamp_from_ckpt_name(model_ckpt_name)

            if ckpt_idx == 0 and not force_success:
                print("Hit target accuracy on the first try. Evaluating a previous checkpoint")
                _ = run_eval(cfg, checkpoints[-1], fid_model, clip_model, local_id, device)

            mllogger.end(
                key=constants.RUN_STOP,
                sync=True,
                metadata={
                    "status": constants.SUCCESS,
                    constants.STEP_NUM: step_num,
                },
                internal_call=True,
                time_ms=success_timestamp,
                unique=True,
            )
            return
        torch.distributed.barrier()

    if len(checkpoints) > 0:
        aborted_timestamp = extract_timestamp_from_ckpt_name(model_ckpt_name)
    else:
        aborted_timestamp = -1
        step_num = -1

    mllogger.end(
        key=constants.RUN_STOP,
        sync=True,
        metadata={
            "status": constants.ABORTED,
            constants.STEP_NUM: step_num,
        },
        internal_call=True,
        time_ms=aborted_timestamp,
        unique=True,
    )


if __name__ == "__main__":
    main()
