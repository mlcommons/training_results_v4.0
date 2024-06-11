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
from __future__ import annotations

import functools
import glob
import pickle
import shutil
import tarfile
from pathlib import Path

import torch
from nemo.collections.multimodal.data.common.webdataset import pil_loader
from nemo.collections.multimodal.data.stable_diffusion.augmentation.augmentations import (
    identical_transform,
)
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    MegatronLatentDiffusion,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from omegaconf.omegaconf import open_dict
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from webdataset import split_by_node
from webdataset import warn_and_continue
from webdataset import WebDataset


class URLSDataset(Dataset):

    def __init__(self, urls):
        self.urls = urls

    def __getitem__(self, index):
        return self.urls[index]

    def __len__(self):
        return len(self.urls)


@hydra_runner(config_path="conf", config_name="sd2_mlperf_train_moments")
def main(cfg) -> None:
    assert cfg.get("out_dir"), "Please specify +out_dir=path in command line"
    out_dir = Path(cfg.out_dir)
    assert not out_dir.exists(), f"Path {cfg.out_dir} already exists"

    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = True

    # We don't need to run `trainer.fit()`, we only requires init.
    with open_dict(cfg):
        cfg.trainer.max_epochs = -1
        cfg.trainer.max_steps = 0

    plugins = []
    if cfg.get("cluster_type") == "BCP":
        plugins.append(TorchElasticEnvironment())

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    trainer = Trainer(
        plugins=plugins,
        strategy=strategy,
        enable_progress_bar=True,
        **cfg.trainer,
    )

    model = MegatronLatentDiffusion(cfg.model, trainer)
    trainer.fit(model)

    model.model.cond_stage_model.model.cuda()
    model.model.cond_stage_model.model.eval()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_pg = torch.distributed.new_group(ranks=[rank])

    def tuple_to_dict(inp):
        for input in inp:
            out_dict = {
                "key": input[0],
                "url": input[1],
                cfg.model.first_stage_key: input[2],
                cfg.model.cond_stage_key: input[3],
            }
            yield out_dict

    def transform_fn(sample):
        latents, text, key, url = (
            sample["npy"],
            sample["txt"],
            sample["__key__"],
            sample["__url__"],
        )
        latents = torch.from_numpy(latents)

        # latents are of shape ([4, 64, 64])
        text_transform = identical_transform
        return key, url, latents, text_transform(text)

    urls_dataset = URLSDataset(glob.glob(cfg.model.data.train.dataset_path))
    urls_sampler = SequentialSampler(urls_dataset)
    urls_dataloader = DataLoader(
        urls_dataset,
        batch_size=world_size,
        drop_last=False,
        shuffle=False,
        sampler=urls_sampler,
    )

    if rank == 0:
        print(f"{len(urls_dataset)} tars, {len(urls_dataloader)} mini-batches")
        print("Encoding captions with CLIP text encoder ...")
    assert len(urls_dataset) % world_size == 0, "There will be repeat processing."
    torch.distributed.barrier()
    out_dir.mkdir(parents=True, exist_ok=True)

    nodesplitter = functools.partial(split_by_node, group=local_pg)
    for urls in tqdm(urls_dataloader, disable=(rank != 0), leave=True):
        webdata = WebDataset(urls, nodesplitter=nodesplitter)
        webdata = webdata.decode(pil_loader, handler=warn_and_continue)
        webdata = webdata.map(transform_fn)
        webdata = webdata.compose(tuple_to_dict)
        dataset = list(iter(webdata))

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.model.micro_batch_size,
            pin_memory=True,
            num_workers=16,
            drop_last=False,
            shuffle=False,
            sampler=sampler,
        )

        """
        msg = "{}: processing {}, {} samples, {} mini-batches".format(
            rank,
            urls[rank],
            len(dataset),
            len(dataloader),
        )
        print(msg)
        """

        # Gemerate intermediate files under /tmp
        tmp_dir = Path("/tmp")
        tar_root = tmp_dir / Path(urls[rank]).stem
        tar_root.mkdir(parents=True, exist_ok=False)

        # Encode captions with CLIP text encoder
        for batch in dataloader:
            with torch.cuda.amp.autocast(
                model.autocast_dtype in (torch.half, torch.bfloat16),
                dtype=model.autocast_dtype,
            ):
                c = model.model.get_learned_conditioning(batch["captions"])

            c = c.detach().cpu().numpy()
            images = batch["images_moments"].cpu().numpy()

            for i in range(len(batch["captions"])):
                assert (
                    urls[rank] == batch["url"][i]
                ), "tarball processed on unexpected rank"

                data = {"image_embed": images[i], "captions_embed": c[i]}
                with open(str(tar_root / f"{batch['key'][i]}.pyd"), "wb") as fd:
                    pickle.dump(data, fd)

                with open(str(tar_root / f"{batch['key'][i]}.txt"), "w") as fd:
                    fd.write(batch["captions"][i])

        # Create tarball and move to destination
        tarball = tmp_dir / Path(urls[rank]).name
        with tarfile.open(tarball, "w") as tar:
            for afile in sorted(glob.glob(f"{tar_root}/*")):
                tar.add(afile, arcname=Path(afile).name)
        shutil.rmtree(tar_root)
        shutil.move(tarball, out_dir / Path(urls[rank]).name)

        torch.distributed.barrier()


if __name__ == "__main__":
    main()
