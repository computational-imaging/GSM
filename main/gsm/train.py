# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."

Code adapted from
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
# os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir, cache_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    if cache_dir is not None:
        if rank == 0:
            print(f'Setting cache directory as `{cache_dir}`...')
            print()
        dnnlib.util.set_cache_dir(cache_dir)
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run, cache_dir):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print(f'Dataset old_code:     {c.training_set_kwargs.old_code}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir, cache_dir=cache_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir, cache_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, resolution, gaussian_weighted_sampler, sample_std, label_file="dataset.json"):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data,
            label_file=label_file, resolution=resolution, use_labels=True, max_size=None, xflip=False,
            gaussian_weighted_sampler=gaussian_weighted_sampler, sample_std=sample_std)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = resolution #dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['shhq', 'deepfashion', 'aist', 'ffhq']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--resume_opt',   help='Resume from given optimizer state', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--cache_dir',    help='Cache directory', metavar='DIR',                          type=str, default=None, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--resolution',   help='Resolution of the rendered image', metavar='INT',         type=click.IntRange(min=1), default=1024, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap_image',         help='How often to save snapshots of images', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap_network',         help='How often to save snapshots of networks', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
# @click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# @click.option('--sr_module',    help='Superresolution module', metavar='STR',  type=str, required=True)
@click.option('--neural_rendering_resolution_initial', help='Resolution to render at', metavar='INT',  type=click.IntRange(min=1), default=64, required=False)
@click.option('--neural_rendering_resolution_final', help='Final resolution to render at, if blending', metavar='INT',  type=click.IntRange(min=1), required=False, default=None)
@click.option('--neural_rendering_resolution_fade_kimg', help='Kimg to blend resolution over', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000, show_default=True)

@click.option('--blur_fade_kimg', help='Blur over how many', metavar='INT',  type=click.IntRange(min=0), required=False, default=200)
@click.option('--gen_pose_cond', help='If true, enable generator pose conditioning.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--c-scale', help='Scale factor for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=1)
@click.option('--c-noise', help='Add noise for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--gpc_reg_prob', help='Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0.5)
@click.option('--gpc_reg_fade_kimg', help='Length of swapping prob fade', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000)
@click.option('--disc_c_noise', help='Strength of discriminator pose conditioning regularization, in standard deviations.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--sr_noise_mode', help='Type of noise for superresolution', metavar='STR',  type=click.Choice(['random', 'none']), required=False, default='none')
@click.option('--resume_blur', help='Enable to blur even on resume', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--sr_num_fp16_res',    help='Number of fp16 layers in superresolution', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--g_num_fp16_res',    help='Number of fp16 layers in generator', metavar='INT', type=click.IntRange(min=0), default=0, required=False, show_default=True)
@click.option('--d_num_fp16_res',    help='Number of fp16 layers in discriminator', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--sr_first_cutoff',    help='First cutoff for AF superresolution', metavar='INT', type=click.IntRange(min=2), default=2, required=False, show_default=True)
@click.option('--sr_first_stopband',    help='First cutoff for AF superresolution', metavar='FLOAT', type=click.FloatRange(min=2), default=2**2.1, required=False, show_default=True)
@click.option('--style_mixing_prob',    help='Style-mixing regularization probability for training.', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0, required=False, show_default=True)
@click.option('--sr-module',    help='Superresolution module override', metavar='STR',  type=str, required=False, default=None)
@click.option('--density_reg',    help='Density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, required=False, show_default=True)
@click.option('--density_reg_every',    help='lazy density reg', metavar='int', type=click.FloatRange(min=1), default=4, required=False, show_default=True)
@click.option('--density_reg_p_dist',    help='density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.004, required=False, show_default=True)
@click.option('--reg_type', help='Type of regularization', metavar='STR',  type=click.Choice(['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation']), required=False, default='l1')
@click.option('--decoder_lr_mul',    help='decoder learning rate multiplier.', metavar='FLOAT', type=click.FloatRange(min=0), default=1, required=False, show_default=True)
@click.option('--decode_first', help='Whether decode the color attributes first and then do interpolation', metavar='STR',  type=click.Choice(['all', 'none', 'wo_color']), required=False, default='all')
@click.option('--reg_weight',    help='regularization weight.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.1, required=False, show_default=True)
@click.option('--opacity_reg',    help='regularization weight for opacity.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, required=False, show_default=True)
@click.option('--l1_loss_reg', help='Use l1 regularizer for the scaling ; if false, l2 will be used', metavar='BOOL', type=bool, required=False, default=True)
@click.option('--clamp_scale_loss', help='Whether use clamping in scaling regularization', metavar='BOOL', type=bool, required=False, default=True)

@click.option('--bg_color', help='the type of scale activation', metavar='STR',  type=click.Choice(['white', 'gray', 'random']), required=False, default='white')
@click.option('--scale_act', help='the type of scale activation', metavar='STR',  type=click.Choice(['exp', 'sigmoid', 'softplus']), required=False, default='exp')
@click.option('--num_shells',    help='The number of shells', metavar='INT', type=click.IntRange(min=1), default=5, required=False, show_default=True)
@click.option('--total_num_gaussians',    help='The number of total points across all shells', metavar='INT', type=int, default=6000, required=True, show_default=True)
@click.option('--offset_len',    help='The offset length of shells.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.05, required=False, show_default=True)
@click.option('--max_scaling',    help='Scaling threshold.', metavar='FLOAT', type=click.FloatRange(min=0, max=0.05), default=0.02, required=False, show_default=True)
@click.option('--perturb_pts', help='Whether to perturb points for each batch', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--gaussian_model', help='tets, shell or volume gaussian', metavar='STR',  type=click.Choice(['tets', 'shells', 'volume']), required=False, default="shells")
@click.option('--rotate_gaussians', help='Whether to rotate gaussians using skinning transformation', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--old_code', help='Old dataloading, no focal length x/2', metavar='BOOL',  type=bool, required=False, default=False)

@click.option('--lr_multiplier_color',    help='Learning rate of color head.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, required=False, show_default=True)
@click.option('--lr_multiplier_opacity',  help='Learning rate of opacity head.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, required=False, show_default=True)
@click.option('--lr_multiplier_scaling',  help='Learning rate of scaling head.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, required=False, show_default=True)
@click.option('--lr_multiplier_rotation', help='Learning rate of rotation head.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, required=False, show_default=True)

## progressive training
@click.option('--progressive_training', help='Whether to use progressive training', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--lod_initial_resolution', help='The initial resolution used in progressive training', metavar='KIMG', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--lod_training_kimg', help='Number of kimgs trained in each level', metavar='KIMG',  type=parse_comma_separated_list, default=None, show_default=True)
@click.option('--lod_transition_kimg', help='Number of kimgs used for transition from one level to the other', metavar='KIMG', type=parse_comma_separated_list, default=None, show_default=True)
@click.option('--ds_pg', help='Whether to use additional progressive training in discriminator', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--direct_max_res', help='Whether to directly train on large res', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--lr_mul_highres', help='whether decrease the lr in higher resolution. ', metavar='FLOAT', type=click.FloatRange(min=0), default=1, required=False, show_default=True)

## progressive training for textures
@click.option('--progressive_tex', help='Whether to use progressive training for textures', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--tex_init_res', help='The initial resolution used in progressive training for textures', metavar='KIMG', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--tex_final_res', help='The final resolution used in progressive training for textures', metavar='KIMG', type=click.IntRange(min=1),  default=512, show_default=True)
@click.option('--lod_transition_kimg_tex', help='Number of kimgs used for transition from one level to the other for textures', metavar='KIMG', type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('--warmup_modules', help='Modules whose related losses will be linearly adjusted for warmup_kimg', metavar='STR', type=parse_comma_separated_list, default=None, show_default=True)
@click.option('--warmup_kimg', help='Modules whose related losses will be linearly adjusted for warmup_kimg', metavar='KIMG', type=click.IntRange(min=0), default=0, show_default=True)

## gaussian sample dataset
@click.option('--gaussian_weighted_sampler', help='Whether to use gaussian_weighted_smapler', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--sample_std', help='The std for guassian sampling', metavar='FLOAT',                 type=click.FloatRange(min=1), default=15, show_default=True)

## Part discriminator
@click.option('--use_face_dist', help='Whether to use face discriminator', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_hand_dist', help='Whether to use hand discriminator', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_foot_dist', help='Whether to use fotot discriminator', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--face_dist_res', help='Resolution of the face discriminator', metavar='INT', type=click.IntRange(min=16), default=128, required=False, show_default=True)
@click.option('--hand_foot_dist_res', help='Resolution of the face discriminator', metavar='INT', type=click.IntRange(min=16), default=128, required=False, show_default=True)
@click.option('--face_weight',   help='weight for face-related loss.', metavar='FLOAT', type=click.FloatRange(min=0), default=1, required=False, show_default=True)
@click.option('--hand_foot_weight', help='Weight for hand/foot discriminator loss', metavar='FLOAT', type=click.FloatRange(min=0), default=1, required=False, show_default=True)
@click.option('--hand_from_start', help='Whether to train hand discriminator from start', metavar='BOOL',  type=bool, required=False, default=True)

## Patch discriminator
@click.option('--use_patch_dist', help='Whether to use patch discriminator', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--patch_res', help='Resolution of the patch discriminator', metavar='INT', type=click.IntRange(min=16), default=128, required=False, show_default=True)
@click.option('--hyper_mod', help='Whether to use modulated patch discriminator', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--patch_from_start', help='Whether to train patch discriminator from start', metavar='BOOL',  type=bool, required=False, default=True)

##progressive scale regularization
@click.option('--progressive_scale_reg_kimg', help='Whether use progressive training for scale regularization. 0 means this is not use.', metavar='FLOAT', type=click.FloatRange(min=0), default=0, required=False, show_default=True)
@click.option('--progressive_scale_reg_end', help='The final scale reg used in progressive training for scale regularization. ', metavar='FLOAT', type=click.FloatRange(min=0), default=0.01, required=False, show_default=True)

## Mask discriminator
@click.option('--use_mask', help='Whether to use mask in discriminators', metavar='BOOL',  type=bool, required=False, default=False)

@click.option('--old_init', help='Whether to use original initialization of decoder', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--ref_scale', help='The reference scale.', metavar='FLOAT', type=click.FloatRange(min=-100), default=-5, required=False, show_default=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    if opts.cfg in ('ffhq', ):
        c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, label_file="dataset.json",
                                                                  resolution=opts.resolution,
                                                                  gaussian_weighted_sampler=opts.gaussian_weighted_sampler,
                                                                  sample_std=opts.sample_std)
    else:
        c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, label_file="dataset.mat",
                                                                  resolution=opts.resolution,
                                                                  gaussian_weighted_sampler=opts.gaussian_weighted_sampler,
                                                                  sample_std=opts.sample_std)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.training_set_kwargs.bg_color = opts.bg_color
    c.training_set_kwargs.old_code = opts.old_code

    # progressive training
    c.progressive_training = opts.progressive_training
    if opts.progressive_training:
        lod_training_kimg = [float(i) for i in opts.lod_training_kimg]
        lod_transition_kimg = [float(i) for i in opts.lod_transition_kimg]
    else:
        lod_training_kimg = lod_transition_kimg = 0
    c.scheduler_kwargs = dnnlib.EasyDict(lod_initial_resolution=opts.lod_initial_resolution,
                                         lod_training_kimg=lod_training_kimg,
                                         lod_transition_kimg=lod_transition_kimg)

    ## progressive for tex
    c.G_kwargs.progressive_tex = opts.progressive_tex
    c.G_kwargs.tex_init_res = opts.tex_init_res
    c.G_kwargs.tex_final_res = opts.tex_final_res
    c.loss_kwargs.lod_transition_kimg_tex = opts.lod_transition_kimg_tex
    c.loss_kwargs.patch_cfg = c.patch_kwargs = dnnlib.EasyDict(min_scale=opts.patch_res/opts.resolution,
                                     max_scale=opts.patch_res/opts.resolution,
                                     patch_res=opts.patch_res,
                                     distribution='uniform',
                                     mbstd_group_size=opts.mbstd_group,
                                     hyper_mod=opts.hyper_mod)
    c.loss_kwargs.clamp = opts.clamp_scale_loss
    c.loss_kwargs.progressive_scale_reg_kimg = opts.progressive_scale_reg_kimg
    c.loss_kwargs.progressive_scale_reg_end = opts.progressive_scale_reg_end

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    c.G_kwargs.gaussian_model = opts.gaussian_model
    c.G_kwargs.rotate_gaussians = opts.rotate_gaussians
    c.G_kwargs.total_num_gaussians = opts.total_num_gaussians
    c.G_kwargs.decode_first = opts.decode_first
    c.G_kwargs.scale_act = opts.scale_act
    c.G_kwargs.num_shells = opts.num_shells
    c.G_kwargs.offset_len = opts.offset_len
    c.G_kwargs.max_scaling = opts.max_scaling
    c.G_kwargs.bg_color = c.training_set_kwargs.bg_color
    c.G_kwargs.lr_multiplier_color = opts.lr_multiplier_color
    c.G_kwargs.lr_multiplier_opacity = opts.lr_multiplier_opacity
    c.G_kwargs.lr_multiplier_scaling = opts.lr_multiplier_scaling
    c.G_kwargs.lr_multiplier_rotation = opts.lr_multiplier_rotation
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.D_kwargs.progressive_training = opts.progressive_training
    c.D_kwargs.ds_pg = opts.ds_pg
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = opts.snap_image
    c.network_snapshot_ticks = opts.snap_network
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.patch_from_start = opts.patch_from_start
    c.direct_max_res = opts.direct_max_res
    c.warmup_modules = opts.warmup_modules
    c.warmup_kimg = opts.warmup_kimg
    c.hand_from_start = opts.hand_from_start
    c.D_kwargs.use_mask = c.loss_kwargs.use_mask = opts.use_mask
    c.lr_mul_highres = opts.lr_mul_highres
    c.G_kwargs.old_init = opts.old_init

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = 'training.triplane.TriPlaneGenerator'
    c.G_kwargs.c_dim = 25
    c.D_kwargs.class_name = 'training.dual_discriminator.DualDiscriminator'
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = 'antialiased' # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    c.D_kwargs.disc_c_noise = opts.disc_c_noise # Regularization for discriminator pose conditioning

    if c.training_set_kwargs.resolution == 512:
        sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
    elif c.training_set_kwargs.resolution == 256:
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif c.training_set_kwargs.resolution == 128:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
    else:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
        # assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"

    if opts.sr_module != None:
        sr_module = opts.sr_module

    rendering_options = {
        'image_resolution': opts.resolution ,#c.training_set_kwargs.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not opts.gen_pose_cond, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale': opts.c_scale, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': opts.sr_noise_mode, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': opts.density_reg, # strength of density regularization
        'density_reg_p_dist': opts.density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': opts.reg_type, # for experimenting with variations on density regularization
        'decoder_lr_mul': opts.decoder_lr_mul, # learning rate multiplier for decoder
        'sr_antialias': True,
    }

    c.patch_kwargs['half_width'] = False
    if opts.cfg.lower() in ('shhq', 'deepfashion', 'aist'):
        c.G_kwargs.offset_len = (opts.offset_len, 1.25*opts.offset_len)  # increase the offset_len for inner layers
        c.patch_kwargs['half_width'] = True
        c.G_kwargs.base_shell_path = os.path.join(os.path.pardir, os.path.pardir, "assets", "smpl", "smpl_uv_no_hands_feet_ear.obj")
        c.G_kwargs.shrunk_ref_mesh = os.path.join(os.path.pardir, os.path.pardir, "assets", "smpl", "smpl_uv_no_hands_feet_ear_shrunk.obj")
        c.G_kwargs.bbox = ((-1.0, -1.2, -1.0), (1.0, 0.8, 1.0))
        c.loss_kwargs.ref_scale = opts.ref_scale

        c.G_kwargs.c_dim = 25
        c.D_kwargs.c_dim = 107  # 25 + 72 (global_orient + body_pose) + 10 (beta)

        c.D_kwargs.label_type = "smpl"
        if c.G_kwargs.gaussian_model != "volume":
            c.loss_kwargs.texture_mask = os.path.join(os.path.pardir, os.path.pardir, "assets", "smpl", "mask_uv.png")
    elif opts.cfg.lower() in ('ffhq',):
        c.patch_kwargs['half_width'] = False
        c.G_kwargs.base_shell_path = os.path.join(os.path.pardir, os.path.pardir, "assets", "flame", "flame_uv_no_back_close_mouth_no_subdivision.obj")
        c.G_kwargs.bbox = ((-0.4, -0.6, -0.4), (0.4, 0.6, 0.4))
        c.loss_kwargs.ref_scale = opts.ref_scale

        c.G_kwargs.c_dim = 25
        c.D_kwargs.c_dim = 81  # 25 + 6 (jaw_pose + global_orient) + 50 (expression)

        c.G_kwargs.smpl_scale = (4.0, 4.0, 3.6)
        c.G_kwargs.smpl_transl = (0.0, 0.0, 0.0)
        if c.G_kwargs.gaussian_model != "volume":
            c.loss_kwargs.texture_mask = os.path.join(os.path.pardir, os.path.pardir, "assets", "flame", "mask_uv.png")
    else:
        assert False, "Need to specify config"

    if c.G_kwargs.gaussian_model == 'volume':
        c.G_kwargs.c_dim = 25
        c.D_kwargs.c_dim = 25

    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.num_fp16_res = 0
    c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32 # Fade out the blur during the first N kimg.

    c.loss_kwargs.gpc_reg_prob = opts.gpc_reg_prob if opts.gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = opts.gpc_reg_fade_kimg
    c.loss_kwargs.dual_discrimination = True
    c.loss_kwargs.neural_rendering_resolution_initial = opts.neural_rendering_resolution_initial
    c.loss_kwargs.neural_rendering_resolution_final = opts.neural_rendering_resolution_final
    c.loss_kwargs.neural_rendering_resolution_fade_kimg = opts.neural_rendering_resolution_fade_kimg
    c.G_kwargs.sr_num_fp16_res = opts.sr_num_fp16_res
    c.loss_kwargs.decode_first = c.decode_first = opts.decode_first
    c.loss_kwargs.reg_weight = opts.reg_weight
    c.loss_kwargs.opacity_reg = opts.opacity_reg
    c.loss_kwargs.l1_loss_reg = opts.l1_loss_reg
    c.loss_kwargs.use_face_dist = c.use_face_dist = opts.use_face_dist
    c.loss_kwargs.use_hand_dist = c.use_hand_dist = opts.use_hand_dist
    c.loss_kwargs.use_foot_dist = c.use_foot_dist = opts.use_foot_dist
    c.loss_kwargs.use_patch_dist = c.use_patch_dist = opts.use_patch_dist
    c.loss_kwargs.face_dist_res = opts.face_dist_res
    c.loss_kwargs.hand_foot_dist_res = opts.hand_foot_dist_res
    c.loss_kwargs.face_weight = opts.face_weight
    c.loss_kwargs.hand_foot_weight = opts.hand_foot_weight

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=opts.cbase, channel_max=opts.cmax, fused_modconv_default='inference_only')

    c.loss_kwargs.style_mixing_prob = opts.style_mixing_prob

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.resume_opt = opts.resume_opt
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
            c.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run, cache_dir=opts.cache_dir)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
