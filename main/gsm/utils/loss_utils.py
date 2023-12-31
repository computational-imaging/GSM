#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision.ops import roi_align
from typing import Tuple, Dict, Callable, Any, List
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def extract_patches(x: torch.Tensor, patch_params: Dict, resolution: int) -> torch.Tensor:
    """
    Extracts patches from images and interpolates them to a desired resolution
    Assumes, that scales/offests in patch_params are given for the [0, 1] image range (i.e. not [-1, 1])
    """
    batch_size, _, h, w = x.shape
    assert h == w, "Can only work on square images (for now)"
    patch_offsets = patch_params['offsets'] # [batch_size, 1], [batch_size, 2]
    patch_offsets = patch_offsets * h
    patch_offsets_2 = patch_offsets + resolution
    patch_offsets_all = torch.stack((patch_offsets, patch_offsets_2), dim=1) # [batch_size, 2, 2]
    indices = torch.arange(0, batch_size).view(-1, 1).float().to(patch_offsets_all.device)
    rois = torch.cat((indices, patch_offsets_all.reshape(batch_size, 4)), dim=1)
    out = roi_align(x, rois, (resolution, resolution))

    # coords = compute_patch_coords(patch_params, resolution) # [batch_size, resolution, resolution, 2]
    # out = F.grid_sample(x, coords, mode='bilinear', align_corners=True) # [batch_size, c, resolution, resolution]
    return out

#----------------------------------------------------------------------------

# def compute_patch_coords(patch_params: Dict, resolution: int, align_corners: bool=True, for_grid_sample: bool=True) -> torch.Tensor:
#     """
#     Given patch parameters and the target resolution, it extracts
#     """
#     patch_scales, patch_offsets = patch_params['scales'], patch_params['offsets'] # [batch_size, 2], [batch_size, 2]
#     batch_size, _ = patch_scales.shape
#     coords = generate_coords(batch_size=batch_size, img_size=resolution, device=patch_scales.device, align_corners=align_corners) # [batch_size, out_h, out_w, 2]

#     # First, shift the coordinates from the [-1, 1] range into [0, 2]
#     # Then, multiply by the patch scales
#     # After that, shift back to [-1, 1]
#     # Finally, apply the offset converted from [0, 1] to [0, 2]
#     coords = (coords + 1.0) * patch_scales.view(batch_size, 1, 1, 2) - 1.0 + patch_offsets.view(batch_size, 1, 1, 2) * 2.0 # [batch_size, out_h, out_w, 2]

#     if for_grid_sample:
#         # Transforming the coords to the layout of `F.grid_sample`
#         coords[:, :, :, 1] = -coords[:, :, :, 1] # [batch_size, out_h, out_w]

#     return coords

#----------------------------------------------------------------------------

def sample_patch_params(batch_size: int, patch_cfg: Dict, device: str='cpu') -> Dict:
    """
    Samples patch parameters: {scales: [x, y], offsets: [x, y]}
    It assumes to follow image memory layout
    """
    if patch_cfg['distribution'] == 'uniform':
        return sample_patch_params_uniform(
            batch_size=batch_size,
            min_scale=patch_cfg['min_scale'],
            max_scale=patch_cfg['max_scale'],
            group_size=patch_cfg['mbstd_group_size'],
            device=device,
            half_width=patch_cfg['half_width']
        )
    elif patch_cfg['distribution'] == 'discrete_uniform':
        return sample_patch_params_uniform(
            batch_size=batch_size,
            min_scale=patch_cfg['min_scale'],
            max_scale=patch_cfg['max_scale'],
            discrete_support=patch_cfg['discrete_support'],
            group_size=patch_cfg['mbstd_group_size'],
            device=device,
            half_width=patch_cfg['half_width']
        )
    elif patch_cfg['distribution'] == 'beta':
        return sample_patch_params_beta(
            batch_size=batch_size,
            min_scale=patch_cfg['min_scale'],
            max_scale=patch_cfg['max_scale'],
            alpha=patch_cfg['alpha'],
            beta=patch_cfg['beta'],
            group_size=patch_cfg['mbstd_group_size'],
            device=device,
            half_width=patch_cfg['half_width']
        )
    else:
        raise NotImplementedError(f'Unkown patch sampling distrubtion: {patch_cfg["distribution"]}')

#----------------------------------------------------------------------------

def sample_patch_params_uniform(batch_size: int, min_scale: float, max_scale: float, discrete_support: List[float]=None, group_size: int=1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales and patch offsets
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
    """
    assert max_scale <= 1.0, f"Too large max_scale: {max_scale}"
    assert min_scale <= max_scale, f"Incorrect params: min_scale = {min_scale}, max_scale = {max_scale}"

    num_groups = batch_size // group_size

    if discrete_support is None:
        patch_scales_x = np.random.rand(num_groups) * (max_scale - min_scale) + min_scale # [num_groups]
    else:
        # Sampling from the discrete distribution
        curr_support = [s for s in discrete_support if min_scale <= s <= max_scale]
        patch_scales_x = np.random.choice(curr_support, size=num_groups, replace=True).astype(np.float32) # [num_groups]

    return create_patch_params_from_x_scales(patch_scales_x, group_size, **kwargs)

#----------------------------------------------------------------------------

def sample_patch_params_beta(batch_size: int, min_scale: float, max_scale: float, alpha: float, beta: float, group_size: int=1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales and patch offsets
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
    """
    assert max_scale <= 1.0, f"Too large max_scale: {max_scale}"
    assert min_scale <= max_scale, f"Incorrect params: min_scale = {min_scale}, max_scale = {max_scale}"
    num_groups = batch_size // group_size
    patch_scales_x = np.random.beta(a=alpha, b=beta, size=num_groups) * (max_scale - min_scale) + min_scale
    return create_patch_params_from_x_scales(patch_scales_x, group_size, **kwargs)

#----------------------------------------------------------------------------

def create_patch_params_from_x_scales(patch_scales_x: np.ndarray, group_size: int=1, device: str='cpu', half_width=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Since we assume that patches are square and we sample assets uniformly,
    we can share a lot of code parts.
    """
    patch_scales_x = torch.from_numpy(patch_scales_x).float().to(device)
    patch_scales = torch.stack([patch_scales_x, patch_scales_x], dim=1) # [num_groups, 2]

    if not half_width:
        # Sample an offset from [0, 1 - patch_size]
        patch_offsets = torch.rand(patch_scales.shape, device=device) * (1.0 - patch_scales) # [num_groups, 2]
    else:
        # Sample an offset from [0, 1 - patch_size] for height and an offset from [0.25, 0.75-patch_size]
        patch_offsets = torch.rand(patch_scales.shape, device=device) # [num_groups, 2]
        patch_offsets[:, 1] = patch_offsets[:, 1] * (1.0 - patch_scales[:,1])
        patch_offsets[:, 0] = 0.25 + patch_offsets[:, 0] * (0.5 - patch_scales[:,0])

    # Replicate the groups (needed for the MiniBatchStdLayer)
    patch_scales = patch_scales.repeat_interleave(group_size, dim=0) # [batch_size, 2]
    patch_offsets = patch_offsets.repeat_interleave(group_size, dim=0) # [batch_size, 2]

    return {'scales': patch_scales, 'offsets': patch_offsets}

#----------------------------------------------------------------------------

def generate_coords(batch_size: int, img_size: int, device='cpu', align_corners: bool=False) -> torch.Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[idx, 0, 0] = (-1, 1)
    - lower right corner: coords[idx, -1, -1] = (1, -1)
    In this way, the `y` axis is flipped to follow image memory layout
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float() # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1 # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = -x_coords.t() # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1) # [batch_size, 2, img_size, img_size]
    coords = coords.permute(0, 2, 3, 1) # [batch_size, 2, img_size, img_size]

    return coords