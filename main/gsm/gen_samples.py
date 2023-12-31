# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import PIL.ImageDraw as ImageDraw
from PIL import Image
import dnnlib
import legacy
import numpy as np
import torch
import matplotlib
from scene.gaussian_model_basic import (GaussianModelMini, render_shells_with_thresholding,
                                        render_thresholding_scaling)
from utils.dataset_utils import create_new_camera, parse_raw_labels
from torch_utils import misc
from pytorch3d.io import save_obj
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm import tqdm
from training.triplane import TriPlaneGenerator, render
from utils.dataset_utils import parse_raw_labels, create_new_camera
from utils.graphics_utils import save_texures_to_images
from utils.image_utils import get_bg_color

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--z_seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=False, metavar='DIR')
@click.option('--bg_color', help='the type of scale activation', metavar='STR',  type=click.Choice(['white', 'gray', 'random']), required=False)
@click.option('--old_code', help='Old dataset code (do not assume divide 2 for x focal)', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--dataset_path', help='Dataset path', type=str, required=False, metavar='PATH', show_default=True)
@click.option('--resolution', help='Resolution', type=int, required=False, metavar='INT', show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--recompute_scaling', help='Determine scaling analytically from NN distance', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--mul_num', help='xN number of Gaussians at test time', type=click.FloatRange(min=0), metavar='FLOAT', show_default=True)
def generate_images(
    network_pkl: str,
    z_seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    bg_color: str,
    dataset_path: str,
    resolution: Optional[int],
    reload_modules: bool,
    mul_num: int,
    recompute_scaling: bool,
    old_code: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 \\
        --network=ffhq-rebalanced-128.pkl
    """
    reload_modules = reload_modules or recompute_scaling or (mul_num is not None)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_init_kwargs = G.init_kwargs.copy()
        if mul_num is not None:
            G_init_kwargs["total_num_gaussians"] = int(G.init_kwargs["total_num_gaussians"] * mul_num)
        G_new = TriPlaneGenerator(*G.init_args, **G_init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    if resolution is not None:
        G.img_resolution = resolution
    if bg_color is not None:
        G.background = get_bg_color(bg_color)

    # opacity_mask = read_image("../../assets/smpl/mask_uv_edited.png")
    # opacity_mask = torchvision.transforms.functional.resize(opacity_mask, (G.img_resolution, G.img_resolution))

    if outdir is None:
        outdir = os.path.join(os.path.dirname(network_pkl), 'samples')
    os.makedirs(outdir, exist_ok=True)

    # Use the same camera but 4 different global orientation: forward, side and back
    # Use different body poses
    dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    dataset_kwargs.path = dataset_path
    dataset_kwargs.old_code = old_code
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    # Camera parameters for SHHQ dataset
    R = torch.eye(3, device=device)
    t = torch.tensor([0, -0.1, -10], device=device)
    cam2world = torch.eye(4, device=device)
    cam2world[:3, :3] = R
    cam2world[:3, 3] = t
    c_cam = cam2world.flatten()
    c_poses = torch.stack([torch.from_numpy(dataset.get_label(idx)) for idx in [2, 4, 8]]).to(device)  # preselected pose 3x107
    # use the same camera for all poses
    c_poses[:, :16] = c_cam

    render_func = lambda pc, cam, pipeline, background: render(cam, pc, pipeline, background, recompute_scaling=recompute_scaling)

    hi, lo = (1, -1)
    # Generate n_pose*n_z_rotation images per seed.
    for seed_idx, seed in enumerate(z_seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(z_seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        imgs = []
        masks = []
        gimgs = []
        ws = None
        for c_idx in range(c_poses.shape[0]):
            # for y_global_orientation in (0, np.pi/2, np.pi, -np.pi/2):
            for y_global_orientation in (0, ):
                if ws is None:
                    ws = G.mapping(z, c_poses[:1], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                c = c_poses[c_idx:(c_idx+1)].clone()
                c[:, 26] += y_global_orientation
                output = G.synthesis(ws=ws, c=c, noise_mode='const',
                                     cache_backbone=True, use_cached_backbone=(c_idx > 0))
                # tmp: multiply outermost shell's opacity with mask
                # G.gaussians._opacity[0] = G.gaussians._opacity[0]
                rendering = output["image"][0]

                labels = parse_raw_labels(c[0])
                camera = create_new_camera(labels, G.img_resolution, G.img_resolution, device)
                smpl_params = {}
                smpl_params["body_pose"] = labels["body_pose"].reshape(1, -1).to(device)
                smpl_params["betas"] = labels["betas"].reshape(1, -1).to(device)
                smpl_params["global_orient"] = labels["global_orient"].reshape(1, -1).to(device)
                smpl_params["transl"] = torch.zeros_like(smpl_params["body_pose"][:, :3]).to(device)

                mask = render(camera, G.gaussians, G.pipeline, torch.zeros_like(G.background).to(device),
                              override_color=torch.ones_like(G.gaussians._xyz).to(device),
                              recompute_scaling=recompute_scaling,
                              deformation_kwargs=smpl_params, device_=device)["render"]

                img = rendering.detach().cpu()
                img = (img - lo) / (hi - lo)
                imgs.append(img)
                masks.append(mask)

        save_image(imgs, os.path.join(outdir, f'seed{seed:04d}.png'), nrow=4, padding=0, normalize=False, range=(0, 1))
        save_image(masks, os.path.join(outdir, f'seed{seed:04d}_mask.png'), nrow=4, padding=0, normalize=False, range=(0, 1))

        # # Render opacity texture accumulated across all shells
        # opacity = G.synthesis(ws=ws, c=c_poses[:1], recompute_scaling=recompute_scaling,
        #                       noise_mode='const', cache_backbone=True, use_cached_backbone=(c_idx > 0))["opacity"][0, ..., 0]  # num_shells, h, w
        # save_image(opacity.sum(dim=0), os.path.join(outdir, f'seed{seed:04d}_opacity_acc.png'), padding=0, normalize=False, range=(0, 1))

        # Render texture planes
        save_texures_to_images(G.gaussians, outdir, prefix=f'seed{seed:04d}_')

        # Generate per shell image for 1 pose and one z_global_orientation
        if "tet" in G.gaussians._orig_class_name.lower():
            num_shells = G.num_shells - 1
        else:
            num_shells = G.num_shells
        imgs = []
        for shell_idx in range(num_shells):
            labels = parse_raw_labels(c_poses[:1])
            cameras = create_new_camera(labels, G.img_resolution, G.img_resolution, device)
            smpl_params = {}
            smpl_params["body_pose"] = labels["body_pose"].reshape(1, -1).to(device)
            smpl_params["betas"] = labels["betas"].reshape(1, -1).to(device)
            smpl_params["global_orient"] = labels["global_orient"].reshape(1, -1).to(device)
            smpl_params["transl"] = torch.zeros_like(smpl_params["body_pose"][:, :3]).to(device)
            img = render_shells_with_thresholding(G.gaussians, num_shells, tuple(range(shell_idx, shell_idx+1)),  # tuple(range(shell_idx, num_shells)
                                                  render_func,
                                                  deformation_kwargs=smpl_params,
                                                  cam=cameras,
                                                  pipeline=G.pipeline, background=G.background.to(device),
                                                  )["render"]
            # NOTE: direct render output is alreay [0, 1]
            # img = (img - lo) / (hi - lo)
            imgs.append(img)
        save_image(imgs, os.path.join(outdir, f'seed{seed:04d}_shell.png'), nrow=num_shells, padding=0, normalize=False, value_range=(0, 1))

        imgs = []
        threshold_levels = torch.linspace(G.gaussians.get_scaling.min().log().item(), G.gaussians.get_scaling.max().log().item(), 6)[:-1]
        for idx, thres in enumerate(threshold_levels):
            thres = thres.exp()
            labels = parse_raw_labels(c_poses[:1])
            cameras = create_new_camera(labels, G.img_resolution, G.img_resolution, device)
            smpl_params = {}
            smpl_params["body_pose"] = labels["body_pose"].reshape(1, -1).to(device)
            smpl_params["betas"] = labels["betas"].reshape(1, -1).to(device)
            smpl_params["global_orient"] = labels["global_orient"].reshape(1, -1).to(device)
            smpl_params["transl"] = torch.zeros_like(smpl_params["body_pose"][:, :3]).to(device)
            rendering = render_thresholding_scaling(G.gaussians, thres, smpl_params,
                                                    render_func, cameras,
                                                    G.pipeline, G.background.to(device)
                                                    )
            if rendering is not None:
                rendering = rendering["render"]
            else:
                rendering = G.background.reshape(3, 1, 1).expand(3, cameras.image_height, cameras.image_width).clone()

            rendering.clamp_(0, 1)
            img = (np.uint8(rendering.clone().detach().permute(1,2,0).cpu().numpy()*255))
            img = Image.fromarray(img)
            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(img)
            # Add Text to an image
            I1.text((28, 36), f"Scaling {thres:.2g}", fill=(255, 0, 0))
            imgs.append(pil_to_tensor(img).to(dtype=torch.float32)/255.0)
        save_image(imgs, os.path.join(outdir, f'seed{seed:04d}_scaling_thres.png'), nrow=len(threshold_levels), padding=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
