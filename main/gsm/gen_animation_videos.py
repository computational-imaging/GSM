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

AMASS_SELECTED = ["../../data/Female1Walking_c3d/B13_-_walk_turn_right_(45)_stageii.npz",
                  "../../data/Female1General_c3d/A3_-_Swing_t2_stageii.npz",
                  "../../data/Female1Walking_c3d/B18_-_walk_to_leap_to_walk_stageii.npz",
                  ]

def get_video_files(root_dir):
    video_data = []

    # Scan the root directory and get all video subdirectories
    for video_name in os.listdir(root_dir):
        video_path = os.path.join(root_dir, video_name)
        if os.path.isdir(video_path):
            videos = os.listdir(video_path)
            if 'smpl.mp4' not in videos:
                continue
            video_files = [os.path.join(video_name, video) for video in videos if video.endswith('.mp4') and video[:4] == 'seed']
            video_files.insert(0, os.path.join(video_name, 'smpl.mp4'))
            video_data.append((video_name, video_files))

    return video_data


def generate_html(video_data, output_dir):
    # Start of the HTML document
    html_string = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Video Gallery</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
      </head>
      <body>
        <div class="container mt-5">
          <h1>Video Gallery</h1>
          <table class="table table-hover">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Video Name</th>
                <th scope="col">Driving Video</th>
                <th scope="col">Seed Videos</th>
              </tr>
            </thead>
            <tbody>
    '''

    # Generating table rows for each video entry
    for index, video in enumerate(video_data, start=1):
        video_name = video[0]
        video_files = video[1]

        # Adding a row for each video
        row_html = f'''
        <tr>
            <th scope="row">{index}</th>
            <td>{video_name}</td>
        '''

        # Embedding the videos within the table data
        for file in video_files:
            row_html += f'''
            <td>
                <video width="256" height="256" controls autoplay muted loop> <!-- 'autoplay' and 'muted' attributes are added here -->
                <source src="{file}" type="video/mp4">
                Your browser does not support the video tag.
              </video>
            </td>
            '''

        row_html += '</tr>\n'
        html_string += row_html

    # End of the HTML document
    html_string += '''
            </tbody>
          </table>
        </div>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.bundle.min.js"></script>
      </body>
    </html>
    '''
    # Write the HTML content to a file
    with open(os.path.join(output_dir,'index.html'), 'w') as html_file:
        html_file.write(html_string)
    return


def generate_one_video(amass_path, z_seeds, G, c, output_dir, truncation_cutoff=14, truncation_psi=0.7):
    device = c.device

    lo, hi = [-1, 1]

    # Target video and parse poses
    pose_target = np.load(amass_path)
    tgt_body_pose = torch.concat(
        [
            torch.from_numpy(pose_target["pose_body"].astype(np.float32)),
            torch.zeros((pose_target["pose_body"].shape[0], 6)),  # doesn't have hand joints
        ],
        dim=-1,
    )

    # First generate image matrix one pose per row
    mkdir_p(output_dir)
    for z_seed in z_seeds:
        z = torch.from_numpy(np.random.RandomState(z_seed).randn(1, G.z_dim)).to(device)
        image = G(z=z, c=c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)["image"].detach().cpu()[0]
        image = (image - lo) / (hi - lo)
        save_image(image, os.path.join(output_dir, "seed_{:04d}.png".format(z_seed)))


    # Generate deformation video for each seed
    video_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(amass_path))[0])
    mkdir_p(video_output_dir)
    for z_seed in z_seeds:
        z = torch.from_numpy(np.random.RandomState(z_seed).randn(1, G.z_dim)).to(device)

        mp4 = os.path.join(video_output_dir, "seed_{:04d}.mp4".format(z_seed))
        video_out = imageio.get_writer(mp4, mode='I', fps=100, codec='libx264', bitrate='10M')

        # map once and synthesis many times
        ws = None
        for frame_idx in tqdm.tqdm(range(tgt_body_pose.shape[0]), desc=f"Rendering {os.path.basename(amass_path)} for seed {z_seed}"):
            tgt_smpl_params = {
                "body_pose": tgt_body_pose[frame_idx].unsqueeze(0).to(device).float(),
            }
            c_anim = update_smpl_to_raw_labels(c, body_pose=tgt_smpl_params['body_pose'])

            # # Temporary fix for SHHQ
            # c_anim[:, 16] *= 2.0
            if frame_idx == 0:
                ws = G.mapping(z, c_anim, truncation_cutoff=truncation_cutoff, truncation_psi=truncation_psi)
            rendering = G.synthesis(ws=ws, c=c_anim, noise_mode='const', cache_backbone=True, use_cached_backbone=(frame_idx > 0))["image"][0]

            img = np.asarray(rendering.detach().permute(1, 2, 0).cpu().numpy(), dtype=np.float32)
            img = (img - lo) / (hi - lo) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            video_out.append_data(img)

        video_out.close()
        print("Wrote target video to {}".format(mp4))

        mp4 = os.path.join(output_dir, os.path.splitext(os.path.basename(amass_path))[0], "smpl.mp4".format(z_seed))
        video_out = imageio.get_writer(mp4, mode='I', fps=100, codec='libx264', bitrate='10M')

    # Render driving video
    for frame_idx in tqdm.tqdm(range(tgt_body_pose.shape[0]), desc=f"Rendering {os.path.basename(amass_path)} driving video"):
        tgt_smpl_params = {
            "body_pose": tgt_body_pose[frame_idx].unsqueeze(0).to(device).float(),
        }
        c_anim = update_smpl_to_raw_labels(c, body_pose=tgt_smpl_params['body_pose'])

        # render mesh
        labels = parse_raw_labels(c_anim)
        camera = create_new_camera(labels, image_width=img.shape[1], img_height=img.shape[0], device=device)
        smpl_params = {}
        smpl_params["body_pose"] = labels["body_pose"].reshape(1, -1).to(device)
        smpl_params["betas"] = labels["betas"].reshape(1, -1).to(device)
        smpl_params["global_orient"] = labels["global_orient"].reshape(1, -1).to(device)
        smpl_params["transl"] = torch.zeros_like(smpl_params["body_pose"][:, :3]).to(device)
        smpl_output = G.gaussians.deformer.smplx_model(**smpl_params)
        world2view = camera.world_view_transform.transpose(1, 2)
        intrinsics = camera.projection_matrix.transpose(1, 2)
        img = render_meshes(Meshes(smpl_output.vertices, G.gaussians.deformer.smplx_model.faces_tensor[None]),
                    world2view, intrinsics, img_size=(camera.image_height, camera.image_width), device=device)[0,...,:3]

        video_out.append_data(img)

    video_out.close()
    print("Wrote video to {}".format(mp4))

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--dataset_path', help='The path of the dataset used for training', metavar='PATH', required=False)
@click.option('--amass_path', help='Pose data to animate generated result with', metavar='PATH', multiple=True, required=False)
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
def generate_video(ctx, network_pkl, amass_path, outdir, gpus,
                   reload_modules, old_code, z_seeds,
                   mul_num,
                   max_scaling,
                   truncation_cutoff, truncation_psi, dataset_path = None):
    if not amass_path:
        amass_path = AMASS_SELECTED

    torch.manual_seed(0)

    device = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    if outdir is None:
        output_dir = os.path.join(os.path.dirname(network_pkl), 'videos')

    mkdir_p(outdir)

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
    # c = torch.from_numpy(dataset[3][-1][None]).to(device=device, dtype=torch.float32)
    c = torch.from_numpy(np.load("../../assets/pose_example.npy")).to(device=device, dtype=torch.float32)

    for video_path in amass_path:
        generate_one_video(video_path, z_seeds, G, c, outdir, truncation_cutoff=truncation_cutoff, truncation_psi=truncation_psi)

    videos = get_video_files(outdir)
    generate_html(videos, outdir)


if __name__ == "__main__":
    generate_video() # pylint: disable=no-value-for-parameter