# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
import torch.nn as nn
import torch.nn.functional as F

model_params = {
    'SigLIP': {'emb_idx': 1, 'padding': 'max_length'},
    'CLIP': {'emb_idx': 0, 'padding': True}
}


class GaussianSmoothingOverTime(nn.Module):

    def __init__(self, size, sigma):
        super().__init__()
        self.size = size
        self.sigma = sigma
        x = torch.arange(size) - size // 2
        g = torch.exp(-x**2 / 2 / sigma**2)
        self.register_buffer('kernel', torch.Tensor(g / g.sum())[None, None])

    @torch.no_grad()
    def forward(self, x):
        x = x.t()[:, None]
        x = F.conv1d(x, self.kernel)
        x = x[:, 0].t()
        x = _normalize(x)
        return x


@torch.no_grad()
def get_image_embeddings(image_input, image_processor, image_model, device, return_tensors=False):
    '''calculate normalized image embeddings
    Args:

        image_input: torch.Tensor[B, 3, H, W], input image/images
        image_processor: callable, a SigLIP or CLIP image preprocessor from HF
        image_model: callable, a SigLIP or CLIP model from HF
        device: torch.device

    output:
        output: np.ndarray[B, D], embeddigs
    '''

    model_family = 'SigLIP' if image_processor._processor_class == 'SiglipProcessor' else 'CLIP'

    x = image_processor(image_input, return_tensors='pt')
    x = x.pixel_values
    x = x.to(device)
    x = image_model(x)
    x = x[model_params[model_family]['emb_idx']]
    x = _normalize(x)
    if return_tensors:
        return x.detach()
    return x.detach().cpu().numpy()


@torch.no_grad()
def get_text_embeddings(text_input, text_processor, text_model, device, return_tensors=False):
    '''calculate normalized text mebeddings
    Args:

        text_input: str or List[str], a prompt or a set of prompts
        text_processor: callable, a SigLIP or CLIP text tokenizer from HF
        text_model: callable, a SigLIP or CLIP model from HF
        device: torch.device

    output:
        output: np.ndarray[B, D], embeddigs
    '''

    model_family = 'SigLIP' if text_processor._processor_class == 'SiglipProcessor' else 'CLIP'

    x = text_processor(
        text_input, padding=model_params[model_family]['padding'], return_tensors='pt')
    x = x.to(device)
    x = text_model(**x)
    x = x[model_params[model_family]['emb_idx']]
    x = _normalize(x)
    if return_tensors:
        return x.detach()
    return x.detach().cpu().numpy()


def _normalize(x):
    return x / x.norm(p=2, dim=-1, keepdim=True)
