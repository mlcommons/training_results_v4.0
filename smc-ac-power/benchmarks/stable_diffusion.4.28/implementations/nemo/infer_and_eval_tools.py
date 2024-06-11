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
from typing import Union

import numpy as np
import open_clip
import torch
import torch.distributed
import torch.utils.data
from PIL import Image
from scipy import linalg
from torch import nn
from torchvision import transforms
from tqdm import tqdm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        clip_version="ViT-B/32",
        pretrained="",
        cache_dir=None,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        self.clip_version = clip_version
        if not pretrained:
            if self.clip_version == "ViT-H-14":
                self.pretrained = "laion2b_s32b_b79k"
            elif self.clip_version == "ViT-g-14":
                self.pretrained = "laion2b_s12b_b42k"
            else:
                self.pretrained = "openai"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_version, pretrained=self.pretrained, cache_dir=cache_dir
        )

        self.model.eval()
        self.model.to(device)

        self.device = device

    @torch.no_grad()
    def get_clip_score(self, text, image):
        if isinstance(image, str):  # filenmae
            image = Image.open(image)
        if isinstance(image, Image.Image):  # PIL Image
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if not isinstance(text, (list, tuple)):
            text = [text]
        text = open_clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        return similarity


class SimplePILDataset(torch.utils.data.Dataset):
    def __init__(self, pil_images, pil_resize=None):
        """
        Args:
            pil_images (list): List of PIL Image objects.
        """
        self.pil_images = pil_images
        self.pil_resize = pil_resize
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.pil_images)

    def __getitem__(self, idx):
        image = self.pil_images[idx].convert("RGB")
        if self.pil_resize is not None:
            image = image.resize(
                (self.pil_resize, self.pil_resize), resample=Image.BICUBIC
            )
        image = self.transform(image)
        image = 2 * image - 1
        return image


@torch.no_grad()
def get_fid_activations(model, data_loader, device, progress_bar=True):
    r"""Compute activation values and pack them in a list.

    Args:
        model (obj): Inception model.
        data_loader (obj): PyTorch dataloader object.
        device (str | device): Device to use.
        progress_bar (bool): Whether to show a progress bar.
    Returns:
        batch_y (tensor): Inception features of the current batch.
    """

    batch_y = []

    # Iterate through the dataset to compute the activation.
    pbar = tqdm(
        data_loader, desc="Calculating FID activations", disable=not progress_bar
    )
    for data in pbar:
        data = data.to(device)
        data.clamp_(-1, 1)
        y = model(data, align_corners=True)
        batch_y.append(y)

    batch_y = torch.cat(batch_y)
    print(f"Computed feature activations of size {batch_y.shape}")
    return batch_y
