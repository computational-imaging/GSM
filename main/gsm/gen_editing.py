
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
from scene.gaussian_model_basic import (render_shells_with_thresholding,
                                        render_thresholding_opacity,
                                        render_thresholding_scaling)
from utils.dataset_utils import create_new_camera, parse_raw_labels
from torch_utils import misc
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from training.triplane import TriPlaneGenerator, render
from utils.dataset_utils import parse_raw_labels, create_new_camera
from utils.graphics_utils import save_texures_to_images

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
@click.option('--pts_path', 'pts_path', help='Indices of points to be replaced', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=False, default='0,1,2')
@click.option('--seeds_tobereplaced', type=int, help='the random seed to be replaced', required=False, default=6162)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=False, metavar='DIR')
@click.option('--old_code', help='Old dataset code (do not assume divide 2 for x focal)', type=bool, default=False, metavar='BOOL', show_default=True)
# @click.option('--dataset_path', help='Dataset path', type=str, required=False, metavar='PATH', show_default=True)
@click.option('--resolution', help='Resolution', type=int, required=False, metavar='INT', show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    seeds_tobereplaced: int,
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    # dataset_path: str,
    resolution: Optional[int],
    reload_modules: bool,
    old_code: bool,
    pts_path: str
):
 
    prefix = pts_path.split('/')[-1][:-4]
    replace_faces = np.loadtxt(pts_path)
    replace_faces = [int(replace_faces[i]) for i in range(len(replace_faces))]

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs, replace_faces=replace_faces).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
        G.gaussians.get_face_index(replace_faces)
    

    if resolution is not None:
        G.img_resolution = resolution

    if outdir is None:
        outdir = os.path.join(os.path.dirname(network_pkl), 'samples')
    os.makedirs(outdir, exist_ok=True)

    # Use the same camera but 4 different global orientation: forward, side and back
    # Use different body poses
    # dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    # dataset_kwargs.path = dataset_path
    # dataset_kwargs.old_code = old_code
    # dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    # Camera parameters for SHHQ dataset
    R = torch.eye(3, device=device)
    t = torch.tensor([0, -0.05, -11.5], device=device)
    cam2world = torch.eye(4, device=device)
    cam2world[:3, :3] = R
    cam2world[:3, 3] = t
    c_cam = cam2world.flatten()
    
    # c_poses = torch.stack([torch.from_numpy(dataset.get_label(idx)) for idx in [6667,0,1,2,3,4,5]]).to(device)
    c_poses = torch.from_numpy(np.load('../../assets/example_editing_poses.npy')).to(device)
    # use the same camera for all poses
    c_poses[:, :16] = c_cam
    c_poses[:,-10:] = c_poses[0,-10:]

    render_func = lambda pc, cam, pipeline, background: render(cam, pc, pipeline, background)

    

    hi, lo = (1, -1)

    for seed1 in tqdm(seeds):
        z1 = torch.from_numpy(np.random.RandomState(seed1).randn(1, G.z_dim)).to(device)
        z2 = torch.from_numpy(np.random.RandomState(seeds_tobereplaced).randn(1, G.z_dim)).to(device)
        if not os.path.isdir(os.path.join(outdir, f'{prefix}_seed{seed1:04d}_{seeds_tobereplaced:04d}')):
            os.makedirs(os.path.join(outdir, f'{prefix}_seed{seed1:04d}_{seeds_tobereplaced:04d}'), exist_ok=True)

        imgs = []
        masks = []
        ws1 = None
        ws2 = None
        
        for c_idx in range(c_poses.shape[0]):

            ws1 = G.mapping(z1, c_poses[:1], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            c = c_poses[c_idx:(c_idx+1)].clone()
            output = G.synthesis(ws=ws1, c=c, noise_mode='const')
            rendering1 = output["image"][0]
            replace_idx = output["replace_idx"]
            replace_color = output["replace_color"][0]
            replace_opacity = output["replace_opacity"][0]
            replace_scale = output["replace_scale"][0]
            replace_rotation = output["replace_rotation"][0]

            ws2 = G.mapping(z2, c_poses[:1], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            c = c_poses[c_idx:(c_idx+1)].clone()
            output2 = G.synthesis(ws=ws2, c=c, noise_mode='const')
            rendering2 = output2["image"][0]

            ws2 = G.mapping(z2, c_poses[:1], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            c = c_poses[c_idx:(c_idx+1)].clone()
            output3 = G.synthesis(ws=ws2, c=c, noise_mode='const',
                                replace_idx=replace_idx, replace_color=replace_color, replace_opacity=replace_opacity,
                                replace_scale=replace_scale, replace_rotation=replace_rotation)
            rendering3 = output3["image"][0]
            

            img1 = rendering1.detach().cpu()
            img1 = (img1 - lo) / (hi - lo)
            imgs.append(img1)
            save_image([img1], os.path.join(outdir, f'{prefix}_seed{seed1:04d}_{seeds_tobereplaced:04d}', f'seed{seed1:04d}_pose{c_idx:04d}.png'), padding=0, normalize=False, range=(0, 1))

            img2 = rendering2.detach().cpu()
            img2 = (img2 - lo) / (hi - lo)
            imgs.append(img2)
            save_image([img2], os.path.join(outdir, f'{prefix}_seed{seed1:04d}_{seeds_tobereplaced:04d}', f'seed{seeds_tobereplaced:04d}_pose{c_idx:04d}.png'), padding=0, normalize=False, range=(0, 1))

            img3 = rendering3.detach().cpu()
            img3 = (img3 - lo) / (hi - lo)
            imgs.append(img3)
            save_image([img3], os.path.join(outdir, f'{prefix}_seed{seed1:04d}_{seeds_tobereplaced:04d}', f'edited_pose{c_idx:04d}.png'), padding=0, normalize=False, range=(0, 1))


        save_image(imgs, os.path.join(outdir, f'{prefix}_seed{seed1:04d}_{seeds_tobereplaced:04d}.png'), nrow=3, padding=0, normalize=False, range=(0, 1))
        

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
