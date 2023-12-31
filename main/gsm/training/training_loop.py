# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from training.crosssection_utils import sample_cross_section
from utils.graphics_utils import save_texures_to_images
from scene.gaussian_model_basic import save_gaussians_to_ply
from pytorch3d.io import save_ply
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = 15
    gh = 7

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, masks, labels = zip(*[training_set[i] for i in grid_indices])

    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------
class Warmup_schedule:
    def __init__(self, warmup_kimg, warmup_modules, loss_class):
        self.warmup_kimg = warmup_kimg
        self.warmup_modules = warmup_modules if warmup_modules is not None else []
        self.warmup_modules = [x.lower() for x in self.warmup_modules]
        if "d_face" in self.warmup_modules:
            self.face_weight = loss_class.face_weight
        if "d_hand" in self.warmup_modules or "d_foot" in self.warmup_modules:
            self.hand_foot_weight = loss_class.hand_foot_weight

    def __call__(self, cur_nimg, loss_class):
        cur_kimg = cur_nimg / 1000.0
        # Training phase.
        multiplier = 1.0
        if cur_kimg < self.warmup_kimg:
            multiplier = cur_kimg / self.warmup_kimg
        if "d_face" in self.warmup_modules:
            loss_class.face_weight = multiplier * self.face_weight
        if "d_hand" in self.warmup_modules or "d_foot" in self.warmup_modules:
            loss_class.hand_foot_weight = multiplier * self.hand_foot_weight


def training_schedule(
    cur_nimg,
    dataset_resolution,
    phase_idx,
    lod_initial_resolution  = 64,        # Image resolution used at the beginning.
    lod_training_kimg       = [400,1000],      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = [400,1000],      # Thousands of real images to show when fading in new layers.
):
    cur_kimg = cur_nimg / 1000.0
    # Training phase.
    if cur_kimg > (sum(lod_training_kimg[:(phase_idx+1)]) + sum(lod_transition_kimg[:(phase_idx+1)])):
        phase_idx  = min(phase_idx + 1, len(lod_training_kimg)-1)
    cur_lod_training_kimg = lod_training_kimg[phase_idx]
    cur_lod_transition_kimg = lod_transition_kimg[phase_idx]
    phase_dur = cur_lod_training_kimg + cur_lod_transition_kimg
    phase_kimg = (cur_kimg - (sum(lod_training_kimg[:phase_idx]) + sum(lod_transition_kimg[:phase_idx]))) if phase_idx > 0 else cur_kimg
    # Level-of-detail and resolution.
    lod = np.log2(dataset_resolution)
    lod -= np.floor(np.log2(lod_initial_resolution))
    lod -= phase_idx
    if cur_lod_transition_kimg > 0:
        lod -= max(phase_kimg - cur_lod_training_kimg, 0.0) / cur_lod_transition_kimg
    lod = max(lod, 0.0)
    resolution = 2 ** (np.log2(dataset_resolution) - int(np.floor(lod)))

    # phase_dur = lod_training_kimg + lod_transition_kimg
    # phase_idx = int(np.floor(cur_kimg / phase_dur)) if phase_dur > 0 else 0
    # phase_kimg = cur_kimg - phase_idx * phase_dur
    # # Level-of-detail and resolution.
    # lod = np.log2(dataset_resolution)
    # lod -= np.floor(np.log2(lod_initial_resolution))
    # lod -= phase_idx
    # if lod_transition_kimg > 0:
    #     lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    # lod = max(lod, 0.0)
    # resolution = 2 ** (np.log2(dataset_resolution) - int(np.floor(lod)))
    return lod, phase_idx
#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    scheduler_kwargs        = {},       # Options for scheduler of progressive training.
    patch_kwargs            = {},
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_opt              = None,     # Optimizer state dict to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    progressive_training    = False,    # Whether use progressive training?
    use_face_dist           = False,    # Whether to use face discriminator
    use_patch_dist          = False,    # Whether to use patch discriminator
    use_hand_dist           = False,    # Whether to use hand discriminator
    use_foot_dist           = False,    # Whether to use foot discriminator
    decode_first            = 'all',    # Whether to decode features before interpolating
    patch_from_start        = True,
    hand_from_start         = True,
    direct_max_res          = False,
    warmup_modules          = None,
    warmup_kimg             = 100,
    lr_mul_highres          = 1.0,
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed, weights=training_set.weights)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')

    common_kwargs = dict(img_resolution=training_set_kwargs.resolution, img_channels=training_set.num_channels)
    G_kwargs['device'] = device
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    with torch.no_grad():
        G_ema = copy.deepcopy(G).eval()

    if use_face_dist:
        D_face_kwargs = D_kwargs.copy()
        D_face_kwargs.update(dict(c_dim=0, img_resolution=loss_kwargs.face_dist_res, img_channels=training_set.num_channels))
        D_face = dnnlib.util.construct_class_by_name(**D_face_kwargs).train().requires_grad_(False).to(device)
    else:
        D_face = None

    if use_hand_dist or use_foot_dist:
        D_hand_kwargs = D_kwargs.copy()
        # Condition on one-hot label for foot / hand?
        D_hand_kwargs.update(dict(c_dim=2, img_resolution=loss_kwargs.hand_foot_dist_res, img_channels=training_set.num_channels))
        D_hand = dnnlib.util.construct_class_by_name(**D_hand_kwargs).train().requires_grad_(False).to(device)
    else:
        D_hand = None

    if use_patch_dist:
        D_patch_kwargs = D_kwargs.copy()
        D_patch_kwargs['class_name'] = 'training.dual_discriminator.PatchDiscriminator'
        D_patch_kwargs['ds_pg'] = False
        patch_d_kwargs = dict(img_resolution=patch_kwargs['patch_res'],
                              img_channels=training_set.num_channels, hyper_mod=patch_kwargs['hyper_mod'])
        D_patch = dnnlib.util.construct_class_by_name(**D_patch_kwargs, **patch_d_kwargs).train().requires_grad_(False).to(device)
    else:
        D_patch = None

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        load_list = [('G', G), ('D', D), ('G_ema', G_ema)]
        if use_face_dist:
            load_list.append(('D_face', D_face))
        if use_patch_dist and not patch_from_start:
            load_list.append(('D_patch', D_patch))
        if use_hand_dist and not hand_from_start:
            load_list.append(('D_hand', D_hand))

        for name, module in load_list:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)


    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    # x = 0
    for module in [G, D, G_ema, augment_pipe, D_face, D_hand, D_patch]:
        if module is not None:
            for names,param in misc.named_params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe,
                                               D_face=D_face, D_hand=D_hand, D_patch=D_patch, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            if name=='D':
                module_list = [module]
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                optimizer_list = [opt]
                if use_face_dist:
                    assert D_face is not None
                    module_list += [D_face]
                    opt_face = dnnlib.util.construct_class_by_name(params=D_face.parameters(), **opt_kwargs)
                    optimizer_list += [opt_face]
                if use_patch_dist:
                    assert D_patch is not None
                    opt_patch = dnnlib.util.construct_class_by_name(params=D_patch.parameters(), **opt_kwargs)
                    module_list += [D_patch]
                    optimizer_list += [opt_patch]
                if use_hand_dist:
                    assert D_hand is not None
                    opt_hand = dnnlib.util.construct_class_by_name(params=D_hand.parameters(), **opt_kwargs)
                    module_list += [D_hand]
                    optimizer_list += [opt_hand]
                phases += [dnnlib.EasyDict(name=name+'both', module=module_list, opt=optimizer_list, interval=1)]
            else:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            if name=='D':
                module_list = [module]
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                optimizer_list = [opt]
                if use_face_dist:
                    assert D_face is not None
                    module_list += [D_face]
                    opt_face = dnnlib.util.construct_class_by_name(params=D_face.parameters(), **opt_kwargs)
                    optimizer_list += [opt_face]
                if use_patch_dist:
                    assert D_patch is not None
                    opt_patch = dnnlib.util.construct_class_by_name(params=D_patch.parameters(), **opt_kwargs)
                    module_list += [D_patch]
                    optimizer_list += [opt_patch]
                if use_hand_dist or use_foot_dist:
                    assert D_hand is not None
                    opt_hand = dnnlib.util.construct_class_by_name(params=D_hand.parameters(), **opt_kwargs)
                    module_list += [D_hand]
                    optimizer_list += [opt_hand]
                phases += [dnnlib.EasyDict(name=name+'main', module=module_list, opt=optimizer_list, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module_list, opt=optimizer_list, interval=reg_interval)]
            else:
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]

    # Resume optimizer snapshot
    if (resume_opt is not None):
        checkpoint = torch.load(resume_opt)
        for phase in phases:
            if isinstance(phase.opt, list):
                for idx, optimizer in enumerate(phase.opt):
                    optimizer.load_state_dict(checkpoint[phase.name][idx])
            else:
                phase.opt.load_state_dict(checkpoint[phase.name])
        if rank == 0:
            print(f'Resuming optimizer state from "{resume_opt}"')


    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)

        c = torch.from_numpy(labels).to(device)
        if not G_kwargs.gaussian_model == 'volume':
            face = G_ema.crop_parts(c.clone(), torch.from_numpy(images.astype('float32')/255.0).to(device), output_size=loss.face_dist_res, parts=["face"])["face"].cpu().numpy()
            save_image_grid(face, os.path.join(run_dir, f'reals_face.png'), drange=[0,1], grid_size=grid_size)
            part_imgs = G_ema.crop_parts(c.clone(), torch.from_numpy(images.astype('float32')/255.0).to(device),
                                        output_size=loss.hand_foot_dist_res, parts=["left_foot", "right_foot", "right_hand", "left_hand"])
            left_foot = part_imgs["left_foot"].cpu().numpy()
            save_image_grid(left_foot, os.path.join(run_dir, f'reals_left_foot.png'), drange=[0,1], grid_size=grid_size)
            right_foot = part_imgs["right_foot"].cpu().numpy()
            save_image_grid(right_foot, os.path.join(run_dir, f'reals_left_foot.png'), drange=[0,1], grid_size=grid_size)
            left_hand = part_imgs["left_hand"].cpu().numpy()
            save_image_grid(left_hand, os.path.join(run_dir, f'reals_left_hand.png'), drange=[0,1], grid_size=grid_size)
            right_hand = part_imgs["right_hand"].cpu().numpy()
            save_image_grid(right_hand, os.path.join(run_dir, f'reals_right_hand.png'), drange=[0,1], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = c
        grid_c = grid_c.split(batch_gpu)

    # Export gaussian xyz
    if rank == 0:
        save_ply(os.path.join(run_dir, 'gaussian_init.ply'), G_ema.gaussians._xyz.detach().cpu())

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()

    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    phase_idx = 0
    warmup_scheduler = Warmup_schedule(warmup_kimg=warmup_kimg, warmup_modules=warmup_modules, loss_class=loss)
    while True:
        if progressive_training:
            if not direct_max_res:
                lod, phase_idx = training_schedule(cur_nimg=cur_nimg, dataset_resolution=training_set_kwargs.resolution, phase_idx=phase_idx, **scheduler_kwargs)
                if lod < 1:
                    lr_mul = lr_mul_highres
                else:
                    lr_mul = 1
            else:
                lod = 0
                lr_mul = 1
        else:
            lod = None
            lr_mul = 1

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            with torch.no_grad():
                phase_real_img, masks, phase_real_c = next(training_set_iterator)
                phase_real_img = phase_real_img.to(device).to(torch.float32) / 127.5 - 1
                phase_real_img  = phase_real_img.split(batch_gpu)
                phase_real_c = phase_real_c.to(device).split(batch_gpu)
                phase_real_mask = (masks.to(device).to(torch.float32)).split(batch_gpu)

            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            ##############
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Adjust loss weight
        warmup_scheduler(cur_nimg, loss)
        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            if phase.name in ['Dboth', 'Dreg', 'Dmain'] and isinstance(phase.module, list):
                for i in range(len(phase.opt)):
                    phase.opt[i].zero_grad(set_to_none=True)
                    phase.module[i].requires_grad_(True)
            else:
                phase.opt.zero_grad(set_to_none=True)
                phase.module.requires_grad_(True)

            for real_img, real_mask, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_mask, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_mask=real_mask, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg, lod=lod, lr_mul=lr_mul)
            if phase.name in ['Dboth', 'Dreg', 'Dmain'] and isinstance(phase.module, list):
                phase.module[i].requires_grad_(False)
            else:
                phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                if phase.name in ['Dboth', 'Dreg', 'Dmain'] and isinstance(phase.module, list):
                    len_idx = len(phase.module)
                    for idx in range(len_idx):
                        params_extended = [(name,param) for (name, param) in phase.module[idx].named_parameters() if param.numel() > 0 and param.grad is not None]
                        params = [x[1] for x in params_extended]
                        names = [x[0] for x in params_extended]
                        if len(params) > 0:
                            flat = torch.cat([param.grad.flatten() for param in params])
                            if num_gpus > 1:
                                torch.distributed.all_reduce(flat)
                                flat /= num_gpus
                            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                            grads = flat.split([param.numel() for param in params])
                            for param, grad, name in zip(params, grads, names):
                                param.grad = grad.reshape(param.shape)
                        phase.opt[idx].step()

                else:
                    params_extended = [(name,param) for (name, param) in phase.module.named_parameters() if param.numel() > 0 and param.grad is not None]
                    params = [x[1] for x in params_extended]
                    names = [x[0] for x in params_extended]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for param, grad, name in zip(params, grads, names):
                            param.grad = grad.reshape(param.shape)
                        # divide G scale branch by resolution / 256 TODO
                    phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            with torch.no_grad():
                ema_nimg = ema_kimg * 1000
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)
                G_ema.texture_res = G.texture_res
                G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        if progressive_training:
            fields += [f"lod {training_stats.report0('Progress/lod', lod):<4f}"]
            fields += [f"lr_mul {training_stats.report0('Progress/lr_mul', lr_mul):<4f}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
                images = torch.cat([o['image'].cpu() for o in out]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

                if use_face_dist:
                    c = torch.cat([c for c in grid_c])
                    gen_img = torch.cat([o['image'] for o in out])
                    face = G_ema.crop_parts(c.clone(), gen_img, output_size=loss.face_dist_res, parts=["face"])["face"].cpu().numpy()
                    save_image_grid(face, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_face.png'), drange=[-1,1], grid_size=grid_size)
                    part_imgs = G_ema.crop_parts(c.clone(), gen_img, output_size=loss.hand_foot_dist_res, parts=["left_foot", "right_foot", "right_hand", "left_hand"])
                    left_foot = part_imgs["left_foot"].cpu().numpy()
                    save_image_grid(left_foot, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_left_foot.png'), drange=[-1,1], grid_size=grid_size)
                    right_foot = part_imgs["right_foot"].cpu().numpy()
                    save_image_grid(right_foot, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_left_foot.png'), drange=[-1,1], grid_size=grid_size)
                    left_hand = part_imgs["left_hand"].cpu().numpy()
                    save_image_grid(left_hand, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_left_hand.png'), drange=[-1,1], grid_size=grid_size)
                    right_hand = part_imgs["right_hand"].cpu().numpy()
                    save_image_grid(right_hand, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_right_hand.png'), drange=[-1,1], grid_size=grid_size)

                #--------------------
                if decode_first == 'all':
                    save_texures_to_images(G_ema.gaussians, os.path.join(run_dir, f'textures_{cur_nimg//1000:06d}'))
                save_gaussians_to_ply(G_ema.gaussians, os.path.join(run_dir, f'gaussian{cur_nimg//1000:06d}.ply'))

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe), ('D_face', D_face), ('D_patch', D_patch), ('D_hand', D_hand)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')

                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
                state_dict = dict()
                for phase in phases:
                    if isinstance(phase.opt, list):
                        state_dict[phase.name] = [opt.state_dict() for opt in phase.opt]
                    else:
                        state_dict[phase.name] = phase.opt.state_dict()
                torch.save(
                    state_dict,
                    os.path.join(run_dir, f'optim-snapshot-{cur_nimg//1000:06d}.pth'))
                del state_dict

        # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        #     if rank == 0:
        #         print(run_dir)
        #         print('Evaluating metrics...')
        #     for metric in metrics:
        #         result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #         if rank == 0:
        #             metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #         stats_metrics.update(result_dict.results)
        # del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
