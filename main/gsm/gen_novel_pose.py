"""Generate animation videos for 3D human"""

import copy
import os
import zipfile

import click
import dnnlib
import imageio
import legacy
import numpy as np
import torch
import tqdm
from pytorch3d.structures import Meshes
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torchvision.utils import save_image
from training.triplane import TriPlaneGenerator
from utils.dataset_utils import (create_new_camera, parse_raw_labels,
                                 update_smpl_to_raw_labels)
from utils.system_utils import mkdir_p, parse_range
from utils.visualization_utils import render_meshes


def generate_one_video( z_seeds, G, c, seq,  output_dir, truncation_cutoff=14, truncation_psi=0.7):
    device = c.device

    seq = torch.from_numpy(np.load(seq[0]))
    lo, hi = [-1, 1]
    for seed1 in z_seeds:
        video_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(str(seed1) + '__novel'))[0])
        mkdir_p(video_output_dir)
        # z_seed  = 58
        mp4 = os.path.join(video_output_dir, "seed_{:04d}.mp4".format(seed1))
        vid = []
    
        for ii in range(len(seq)):

            z = torch.from_numpy(np.random.RandomState(seed1).randn(1, G.z_dim)).to(device)
            ws = G.mapping(z, c, truncation_cutoff=truncation_cutoff, truncation_psi=truncation_psi)
            image = G.synthesis(ws=ws, c=c, noise_mode='const', orient = seq[ii].reshape(1, -1))["image"].detach().cpu()[0]
            img = np.asarray(image.detach().permute(1, 2, 0).cpu().numpy(), dtype=np.float32)
            img = (img - lo) / (hi - lo) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)

            vid.append(img)
        imageio.v2.mimwrite(mp4, vid)
        print("Wrote video to {}".format(mp4))

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--dataset_path', help='The path of the dataset used for training', metavar='PATH', required=False)
@click.option('--seq', "seq",help='Pose data to animate generated result with', metavar='PATH', multiple=True, required=True)
@click.option('--outdir', help='Directory to output to', metavar='PATH')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--old_code', help='Old dataset code (do not assume divide 2 for x focal)', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--z_seeds', help='Seeds', type=parse_range)
@click.option('--mul_num', help='xN number of Gaussians at test time', type=int, default=1, metavar='INT', show_default=True)
@click.option('--max_scaling', help='Scaling activation clamping', type=float, metavar='FLOAT')
@click.option('--trunc_psi', "truncation_psi", help='Truncation parameters', type=float, default=0.7, metavar='FLOAT', show_default=True)
@click.option('--trunc_cutoff', "truncation_cutoff", help='Truncation parameters', type=int, default=14, metavar='FLOAT', show_default=True)
@torch.no_grad()
def generate_video(ctx, network_pkl, outdir, gpus,
                   reload_modules, old_code, z_seeds,
                   mul_num,
                   max_scaling,
                   truncation_cutoff, truncation_psi,seq, dataset_path = None):


    torch.manual_seed(0)

    device = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True


    # Load network and dataset
    with dnnlib.util.open_url(network_pkl, verbose=True) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'] # subclass of torch.nn.Module
        G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
        # dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
        # # Load dataset
        # if dataset_path is not None:
        #     dataset_kwargs.path = dataset_path
        # dataset_kwargs.old_code = old_code
        # dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    if reload_modules:
        print("Reloading Modules!")
        G.init_kwargs["total_num_gaussians"] *= mul_num
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    if max_scaling is not None:
        G.gaussians.max_scaling = max_scaling

    # Randomly picked this label, seems to be a good camera angle
    # c = torch.from_numpy(dataset[6667][-1][None]).to(device=device, dtype=torch.float32)
    c = torch.from_numpy(np.load("../../assets/pose_example.npy")).to(device=device, dtype=torch.float32)

    generate_one_video(z_seeds, G, c, seq, outdir, truncation_cutoff=truncation_cutoff, truncation_psi=truncation_psi)




if __name__ == "__main__":
    generate_video() # pylint: disable=no-value-for-parameter