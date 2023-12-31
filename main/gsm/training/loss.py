# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from PIL import Image
import torchvision.transforms as transforms
from utils.loss_utils import extract_patches, sample_patch_params
import random


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, D_face=None, D_patch=None, D_hand=None, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0,
                 pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0,
                 blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64,
                 neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0,
                 gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',
                 decode_first='all', reg_weight=0.1, opacity_reg= 1, l1_loss_reg = True, texture_mask=None,
                 lod_transition_kimg_tex = 1000, use_face_dist=False, use_foot_dist=False, use_hand_dist=False,
                 face_dist_res=128, face_weight=1.0, hand_foot_dist_res=128, hand_foot_weight=1.0,
                 patch_cfg=None, use_patch_dist=False, ref_scale=None, clamp=False, use_mask=False, 
                 progressive_scale_reg_kimg=0, progressive_scale_reg_end=0.01):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.D_face             = D_face
        self.D_hand             = D_hand
        self.D_patch            = D_patch
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.decode_first = decode_first
        self.reg_weight = reg_weight
        self.opacity_reg = opacity_reg
        self.l1_loss_reg = l1_loss_reg
        self.lod_transition_kimg_tex = lod_transition_kimg_tex
        self.use_face_dist = use_face_dist
        self.use_foot_dist = use_foot_dist
        self.use_hand_dist = use_hand_dist
        self.face_dist_res = face_dist_res
        self.hand_foot_dist_res = hand_foot_dist_res
        self.face_weight = face_weight
        self.hand_foot_weight = hand_foot_weight
        self.patch_cfg = patch_cfg
        self.use_patch_dist = use_patch_dist
        self.clamp = clamp
        self.mask_image = None
        self.use_mask = use_mask
        self.progressive_scale_reg_kimg = progressive_scale_reg_kimg
        self.progressive_scale_reg_end = progressive_scale_reg_end
        print('texture_mask', texture_mask)
        if texture_mask is not None:
            with torch.no_grad():
                mask_image = Image.open(texture_mask)
                transform = transforms.Compose([
                        transforms.ToTensor(),  # Converts the image to a tensor with values between 0 and 1
                    ])
                self.mask_image =  transform(mask_image)

        with torch.no_grad():
            if decode_first == 'all' or decode_first == 'wo_color':
                self.ref_scale = ref_scale or -5.0 # -4.5  #4.5 # -5.2902 #-5.2902 # -3.5616 #
            else:
                self.ref_scale = ref_scale or 0.0001

        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c,  swapping_prob, neural_rendering_resolution, update_emas=False, lod=None, lr_mul=1):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, update_emas=update_emas, lod=lod, lr_mul=lr_mul)
        return gen_output, ws

    def run_D(self, img, c,  blur_sigma=0, blur_sigma_raw=0, update_emas=False, lod=None):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    img['mask']],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['mask'] = augmented_pair[:, img['image'].shape[1]:]

        logits = self.D(img, c, update_emas=update_emas, lod=lod)
        return logits

    def run_D_patch(self, img, c, patch_params=None, blur_sigma=0, blur_sigma_raw=0, update_emas=False, lod=None):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]

        logits = self.D_patch(img, c, update_emas=update_emas, lod=lod, patch_params=patch_params)
        return logits

    def run_D_face(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, lod=None):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        logits = self.D_face(img, c, update_emas=update_emas, lod=0)
        return logits

    def run_D_hand(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, lod=None):
        blur_size = np.floor(blur_sigma * 3)
        device = img["image"].device
        batch_size = img["image"].shape[0]
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
        c = torch.nn.functional.one_hot(torch.zeros(batch_size, dtype=torch.long, device=device, requires_grad=False), num_classes=2).to(device).float()
        logits = self.D_hand(img, c, update_emas=update_emas, lod=0)
        return logits

    def run_D_foot(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, lod=None):
        blur_size = np.floor(blur_sigma * 3)
        device = img["image"].device
        batch_size = img["image"].shape[0]
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        c = torch.nn.functional.one_hot(torch.ones(batch_size, dtype=torch.long, device=device, requires_grad=False), num_classes=2).to(device).float()
        logits = self.D_hand(img, c, update_emas=update_emas, lod=0)
        return logits

    def loss_opacity(self, opacity_map, texture_mask):
        """Inside the masked region of the texture, opacity should sum to 1.
        Args:
            texture_mask: (bs, 1, h, w) opacity after activation
            opacity_map: (bs, sh, h, w)
        returns:
            loss: float
        """
        texture_mask = texture_mask.to(opacity_map.device)
        texture_mask = filtered_resizing(texture_mask.unsqueeze(0), size=opacity_map.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).squeeze(0).repeat(opacity_map.shape[0], 1, 1)

        loss_map = torch.abs(torch.sum(opacity_map.squeeze(-1), dim=1, keepdim=False) - 1) * texture_mask
        return torch.sum(loss_map) / torch.sum(texture_mask)

    def loss_clamp_l2(self, source, target, mask=None, clamp=True):
        """
        Args:
            source: (bs, sh, h, w, c)
            target: float value
            mask: (bs, 1, h, w)
        Returns:
            float
        """
        if clamp:
            loss_map = torch.clamp((source - target), min=0)**2
        else:
            loss_map = (source - target)**2
        if mask is not None:
            mask = mask.to(source.device)
            texture_mask = filtered_resizing(mask.unsqueeze(0), size=source.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).repeat(source.shape[0], source.shape[1], 1, 1)
            return torch.sum(loss_map * texture_mask[..., None])/ torch.sum(texture_mask)
        else:
            return torch.mean(loss_map)

    def loss_clamp_l1(self, source, target_value, mask=None, clamp=False):
        """
        Args:
            source: (bs, sh, h, w, c)
            target_value: float value
            mask: (1, h, w)
        Returns:
            loss: float
        """
        if clamp:
            loss_map = torch.abs(torch.clamp(source - target_value, min=0))
        else:
            loss_map = torch.abs(source - target_value)
        if mask is not None:
            mask = mask.to(source.device)
            texture_mask = filtered_resizing(mask.unsqueeze(0), size=source.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).repeat(source.shape[0], source.shape[1], 1, 1)
            return torch.sum(loss_map * texture_mask[...,None]) / torch.sum(texture_mask)
        else:
            return torch.mean(loss_map)

    def extract_patches(self, img: torch.Tensor):
        patch_params = sample_patch_params(len(img), self.patch_cfg, device=img.device)
        img = extract_patches(img.clone(), patch_params, resolution=self.patch_cfg['patch_res']) # [batch_size, c, h_patch, w_patch]

        return img, patch_params

    def preprocess_image(self, images, lod=0, keep_original_res=True):
        """Pre-process images to support progressive training."""
        # Downsample to the resolution of the current phase (level-of-details).
        if lod != int(lod):
            lod = int(lod)
        for _ in range(int(lod)):
            images = torch.nn.functional.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
        # Transition from the previous phase (level-of-details) if needed.
        if lod != int(lod):
            downsampled_images = torch.nn.functional.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
            upsampled_images = torch.nn.functional.interpolate(
                downsampled_images, scale_factor=2, mode='nearest')
            alpha = lod - int(lod)
            images = images * (1 - alpha) + upsampled_images * alpha
        # Upsample back to the resolution of the model.
        if int(lod) == 0:
            return images
        if keep_original_res:
            return torch.nn.functional.interpolate(
                images, scale_factor=(2 ** int(lod)), mode='nearest')
        else:
            return images

    def accumulate_gradients(self, phase, real_img, real_mask, real_c, gen_z, gen_c, gain, cur_nimg, lod, lr_mul):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if lod is None:
            lod = 0
        real_img = self.preprocess_image(real_img, lod=lod, keep_original_res=True).detach()
        real_mask = self.preprocess_image(real_mask, lod=lod, keep_original_res=True).detach()

        # if self.blur_raw_target:
        #     blur_size = np.floor(blur_sigma * 3)
        #     if blur_size > 0:
        #         f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
        #         real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        # real_img = {'image': real_img, 'image_raw': real_img_raw}
        alpha_tex = min(cur_nimg / (self.lod_transition_kimg_tex * 1e3), 1)
        real_img = {'image': real_img, 'image_raw': 1, 'mask': real_mask}


        if self.progressive_scale_reg_kimg > 0:
            reg_weight_cur = self.reg_weight - min(cur_nimg / (self.progressive_scale_reg_kimg * 1e3), 1) * (self.reg_weight-self.progressive_scale_reg_end)

        else:
            reg_weight_cur = self.reg_weight

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                self.G.alpha = alpha_tex
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=None, lod=lod, lr_mul=lr_mul)
                if self.reg_weight > 0:
                    if self.decode_first == 'all' or self.decode_first == 'wo_color':
                        if self.l1_loss_reg:
                            scaling_loss = self.loss_clamp_l1(gen_img['scaling'], self.ref_scale, mask=self.mask_image, clamp=self.clamp)
                        else:
                            scaling_loss = self.loss_clamp_l2(gen_img['scaling'], self.ref_scale, mask=self.mask_image, clamp=self.clamp)

                        training_stats.report('Loss/G/loss_reg_scale', scaling_loss)
                        training_stats.report('Loss/G/scaling_max', torch.max(gen_img['scaling']))
                    else:
                        if self.l1_loss_reg:
                            scaling_loss = self.loss_clamp_l1(gen_img['scaling'], self.ref_scale, clamp=self.clamp)
                        else:
                            scaling_loss = self.loss_clamp_l2(gen_img['scaling'], self.ref_scale, clamp=self.clamp)

                        training_stats.report('Loss/G/loss_reg_scale', scaling_loss)
                        training_stats.report('Loss/G/scaling_max', torch.max(gen_img['scaling']))


                if self.opacity_reg > 0 and self.mask_image is not None:
                    opacity_loss = self.loss_opacity(gen_img['opacity'], self.mask_image).to(_gen_ws.device)
                    training_stats.report('Loss/G/opacity_loss', opacity_loss)

                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, lod=lod)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)

                if self.use_face_dist:
                    if self.use_mask:
                        g_face_input = self.G.crop_parts(gen_c.clone(), torch.cat((gen_img['image'].clone(), gen_img['mask'].clone()), dim=1), output_size=self.face_dist_res, parts=['face'])["face"]
                        gen_face_logits = self.run_D_face({"image": g_face_input[:,:3], 'mask': g_face_input[:,3:]}, gen_c, blur_sigma=blur_sigma, lod=lod)
                    else:
                        g_face_input = self.G.crop_parts(gen_c.clone(), gen_img['image'].clone(), output_size=self.face_dist_res, parts=['face'])["face"]
                        gen_face_logits = self.run_D_face({"image": g_face_input[:,:3]}, gen_c, blur_sigma=blur_sigma, lod=lod)
                    training_stats.report('Face_Loss/scores/fake', gen_face_logits)
                    training_stats.report('Face_Loss/signs/fake', gen_face_logits.sign())
                    loss_face_Gmain = torch.nn.functional.softplus(-gen_face_logits)
                    training_stats.report('Face_Loss/G/loss', loss_face_Gmain)

                if self.use_hand_dist:
                    if self.use_mask:
                        g_hand_input = self.G.crop_parts(gen_c.clone(), torch.cat((gen_img['image'].clone(), gen_img['mask'].clone()), dim=1), output_size=self.hand_foot_dist_res, parts=['left_hand', 'right_hand'])
                        g_hand_input_img = torch.cat([v[:,:3] for v in g_hand_input.values()])
                        g_hand_input_mask = torch.cat([v[:,3:] for v in g_hand_input.values()])
                        gen_hand_logits = self.run_D_hand({"image": g_hand_input_img, 'mask': g_hand_input_mask}, gen_c, blur_sigma=blur_sigma, lod=lod)
                    else:
                        g_hand_input = self.G.crop_parts(gen_c.clone(), gen_img['image'].clone(), output_size=self.hand_foot_dist_res, parts=['left_hand', 'right_hand'])
                        g_hand_input = torch.cat([v for v in g_hand_input.values()])
                        gen_hand_logits = self.run_D_hand({"image": g_hand_input}, gen_c, blur_sigma=blur_sigma, lod=lod)
                    training_stats.report('Hand_Loss/scores/fake', gen_hand_logits)
                    training_stats.report('Hand_Loss/signs/fake', gen_hand_logits.sign())
                    loss_hand_Gmain = torch.nn.functional.softplus(-gen_hand_logits)
                    training_stats.report('Hand_Loss/G/loss', loss_hand_Gmain)

                if self.use_foot_dist:
                    if self.use_mask:
                        g_foot_input = self.G.crop_parts(gen_c.clone(), torch.cat((gen_img['image'].clone(), gen_img['mask'].clone()), dim=1), output_size=self.hand_foot_dist_res, parts=['left_foot', 'right_foot'])
                        g_foot_input_img = torch.cat([v[:,:3] for v in g_foot_input.values()])
                        g_foot_input_mask = torch.cat([v[:,3:] for v in g_foot_input.values()])
                        gen_foot_logits = self.run_D_foot({"image": g_foot_input_img, 'mask': g_foot_input_mask}, gen_c, blur_sigma=blur_sigma, lod=lod)
                    else:
                        g_foot_input = self.G.crop_parts(gen_c.clone(), gen_img['image'].clone(), output_size=self.hand_foot_dist_res, parts=['left_foot', 'right_foot'])
                        g_foot_input = torch.cat([v for v in g_foot_input.values()])
                        gen_foot_logits = self.run_D_foot({"image": g_foot_input}, gen_c, blur_sigma=blur_sigma, lod=lod)
                    training_stats.report('Foot_Loss/scores/fake', gen_foot_logits)
                    training_stats.report('Foot_Loss/signs/fake', gen_foot_logits.sign())
                    loss_foot_Gmain = torch.nn.functional.softplus(-gen_foot_logits)
                    training_stats.report('Foot_Loss/G/loss', loss_foot_Gmain)

                if self.use_patch_dist:
                    if self.use_mask:
                        patch, patch_params = self.extract_patches(torch.cat((gen_img['image'], gen_img['mask']), dim=1))
                        g_patch_input = {'image': patch[:,:3],'mask': patch[:,3:]}
                    else:
                        patch, patch_params = self.extract_patches(gen_img['image'])
                        g_patch_input = {'image': patch}
                    gen_patch_logits = self.run_D_patch(g_patch_input, gen_c, patch_params=patch_params, blur_sigma=blur_sigma, lod=lod)
                    training_stats.report('Patch_Loss/scores/fake', gen_patch_logits)
                    training_stats.report('Patch_Loss/signs/fake', gen_patch_logits.sign())
                    loss_patch_Gmain = torch.nn.functional.softplus(-gen_patch_logits)
                    training_stats.report('Patch_Loss/G/loss', loss_patch_Gmain)


            with torch.autograd.profiler.record_function('Gmain_backward'):
                if self.use_face_dist:
                    loss_Gmain = loss_Gmain + loss_face_Gmain * self.face_weight
                if self.use_hand_dist:
                    loss_Gmain = loss_Gmain + loss_hand_Gmain.mean() * self.hand_foot_weight
                if self.use_foot_dist:
                    loss_Gmain = loss_Gmain + loss_foot_Gmain.mean() * self.hand_foot_weight
                if self.use_patch_dist:
                    loss_Gmain = loss_Gmain + loss_patch_Gmain
                if self.reg_weight > 0:
                    loss_Gmain = loss_Gmain + reg_weight_cur*scaling_loss
                if self.opacity_reg > 0 and self.mask_image is not None:
                    loss_Gmain = loss_Gmain + self.opacity_reg*opacity_loss

                loss_Gmain.mean().mul(gain).backward()

        if phase in ['Greg', 'Gboth']:
            pass

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                self.G.alpha = alpha_tex
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=None, update_emas=True, lod=lod, lr_mul=lr_mul)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, lod=lod)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)

                if self.use_face_dist:
                    if self.use_mask:
                        d_gen_face_input = self.G.crop_parts(gen_c.clone(), torch.cat((gen_img['image'].clone(), gen_img['mask'].clone()), dim=1), output_size=self.face_dist_res, parts=['face'])["face"]
                        gen_face_logits = self.run_D_face({"image": d_gen_face_input[:,:3], 'mask': d_gen_face_input[:,3:]}, gen_c, blur_sigma=blur_sigma, update_emas=True, lod=lod)
                    else:
                        d_gen_face_input = self.G.crop_parts(gen_c.clone(), gen_img['image'].clone(), output_size=self.face_dist_res, parts=['face'])["face"]
                        gen_face_logits = self.run_D_face({"image": d_gen_face_input}, gen_c, blur_sigma=blur_sigma, update_emas=True, lod=lod)
                    training_stats.report('Face_Loss/scores/fake', gen_face_logits)
                    training_stats.report('Face_Loss/signs/fake', gen_face_logits.sign())
                    loss_face_Dgen = torch.nn.functional.softplus(gen_face_logits)

                if self.use_foot_dist:
                    if self.use_mask:
                        d_gen_foot_input = self.G.crop_parts(gen_c.clone(), torch.cat((gen_img['image'].clone(), gen_img['mask'].clone()), dim=1), output_size=self.hand_foot_dist_res, parts=['left_foot', 'right_foot'])
                        d_gen_foot_input_img = torch.cat([v[:,:3] for v in d_gen_foot_input.values()])
                        d_gen_foot_input_mask = torch.cat([v[:,3:] for v in d_gen_foot_input.values()])
                        gen_foot_logits = self.run_D_foot({"image": d_gen_foot_input_img, 'mask': d_gen_foot_input_mask}, gen_c, blur_sigma=blur_sigma, update_emas=True, lod=lod)
                    else:
                        d_gen_foot_input = self.G.crop_parts(gen_c.clone(), gen_img['image'].clone(), output_size=self.hand_foot_dist_res, parts=['left_foot', 'right_foot'])
                        d_gen_foot_input = torch.cat([v for v in d_gen_foot_input.values()])
                        gen_foot_logits = self.run_D_foot({"image": d_gen_foot_input}, gen_c, blur_sigma=blur_sigma, update_emas=True, lod=lod)
                    training_stats.report('Foot_Loss/scores/fake', gen_foot_logits)
                    training_stats.report('Foot_Loss/signs/fake', gen_foot_logits.sign())
                    loss_foot_Dgen = torch.nn.functional.softplus(gen_foot_logits)

                if self.use_hand_dist:
                    if self.use_mask:
                        d_gen_hand_input = self.G.crop_parts(gen_c.clone(), torch.cat((gen_img['image'].clone(), gen_img['mask'].clone()),dim=1), output_size=self.hand_foot_dist_res, parts=['left_hand', 'right_hand'])
                        d_gen_hand_input_img = torch.cat([v[:,:3] for v in d_gen_hand_input.values()])
                        d_gen_hand_input_mask = torch.cat([v[:,3:] for v in d_gen_hand_input.values()])
                        gen_hand_logits = self.run_D_hand({"image": d_gen_hand_input_img, 'mask': d_gen_hand_input_mask}, gen_c, blur_sigma=blur_sigma, update_emas=True, lod=lod)
                    else:
                        d_gen_hand_input = self.G.crop_parts(gen_c.clone(), gen_img['image'].clone(), output_size=self.hand_foot_dist_res, parts=['left_hand', 'right_hand'])
                        d_gen_hand_input = torch.cat([v for v in d_gen_hand_input.values()])
                        gen_hand_logits = self.run_D_hand({"image": d_gen_hand_input}, gen_c, blur_sigma=blur_sigma, update_emas=True, lod=lod)
                    training_stats.report('Hand_Loss/scores/fake', gen_hand_logits)
                    training_stats.report('Hand_Loss/signs/fake', gen_hand_logits.sign())
                    loss_hand_Dgen = torch.nn.functional.softplus(gen_hand_logits)

                if self.use_patch_dist:
                    if self.use_mask:
                        patch, patch_params = self.extract_patches(torch.cat((gen_img['image'], gen_img['mask']), dim=1))
                        d_patch_input = {'image': patch[:,:3], 'mask': patch[:,3:]}
                    else:
                        patch, patch_params = self.extract_patches(gen_img['image'])
                        d_patch_input = {'image': patch}
                    gen_patch_logits = self.run_D_patch(d_patch_input, gen_c.clone(), patch_params=patch_params, blur_sigma=blur_sigma, lod=lod)
                    training_stats.report('Patch_Loss/scores/fake', gen_patch_logits)
                    training_stats.report('Patch_Loss/signs/fake', gen_patch_logits.sign())
                    loss_patch_Dgen = torch.nn.functional.softplus(gen_patch_logits)

            with torch.autograd.profiler.record_function('Dgen_backward'):
                if self.use_face_dist:
                    loss_Dgen = loss_Dgen + loss_face_Dgen * self.face_weight
                if self.use_hand_dist:
                    loss_Dgen = loss_Dgen + loss_hand_Dgen.mean() * self.hand_foot_weight
                if self.use_foot_dist:
                    loss_Dgen = loss_Dgen + loss_foot_Dgen.mean() * self.hand_foot_weight
                if self.use_patch_dist:
                    loss_Dgen = loss_Dgen + loss_patch_Dgen
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = None
                real_img_tmp_mask = real_img['mask'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'mask': real_img_tmp_mask}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, lod=lod)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                if self.use_face_dist:
                    if self.use_mask:
                        real_face = self.G.crop_parts(real_c.clone(), torch.cat((real_img_tmp['image'].clone(), real_img_tmp['mask'].clone()),dim=1), output_size=self.face_dist_res, parts=['face'])["face"]
                        real_face_img = real_face[:,:3].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_face_mask = real_face[:,3:].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_face_logits = self.run_D_face({"image": real_face_img, 'mask': real_face_mask}, real_c.clone(), blur_sigma=blur_sigma, lod=lod)
                    else:
                        real_face = self.G.crop_parts(real_c.clone(), real_img_tmp['image'].clone(), output_size=self.face_dist_res, parts=['face'])["face"]
                        real_face = real_face.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_face_logits = self.run_D_face({"image": real_face}, real_c.clone(), blur_sigma=blur_sigma, lod=lod)
                    training_stats.report('Face_Loss/scores/real', real_face_logits)
                    training_stats.report('Face_Loss/signs/real', real_face_logits.sign())

                    loss_face_Dreal = 0
                    if phase in ['Dmain', 'Dboth']:
                        loss_face_Dreal = torch.nn.functional.softplus(-real_face_logits)
                        training_stats.report('Face_Loss/D/loss', loss_face_Dgen + loss_face_Dreal)

                if self.use_hand_dist:
                    if self.use_mask:
                        real_hands = self.G.crop_parts(real_c.clone(), torch.cat((real_img_tmp['image'].clone(), real_img_tmp['mask'].clone()),dim=1), output_size=self.hand_foot_dist_res, parts=['left_hand', 'right_hand'])
                        real_hands = torch.cat([v for v in real_hands.values()])
                        real_hands_img = real_hands[:,:3].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_hands_mask = real_hands[:,3:].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_hand_logits = self.run_D_hand({"image": real_hands_img, 'mask': real_hands_mask}, real_c.clone(), blur_sigma=blur_sigma, lod=lod)
                    else:
                        real_hands = self.G.crop_parts(real_c.clone(), real_img_tmp['image'].clone(), output_size=self.hand_foot_dist_res, parts=['left_hand', 'right_hand'])
                        real_hands = torch.cat([v for v in real_hands.values()])
                        real_hands = real_hands.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_hand_logits = self.run_D_hand({"image": real_hands}, real_c.clone(), blur_sigma=blur_sigma, lod=lod)
                    training_stats.report('Hand_Loss/scores/real', real_hand_logits)
                    training_stats.report('Hand_Loss/signs/real', real_hand_logits.sign())

                    loss_hand_Dreal = torch.tensor(0.0).to(real_hand_logits.device)
                    if phase in ['Dmain', 'Dboth']:
                        loss_hand_Dreal = torch.nn.functional.softplus(-real_hand_logits)
                        training_stats.report('Hand_Loss/D/loss', loss_hand_Dgen + loss_hand_Dreal)

                if self.use_foot_dist:
                    if self.use_mask:
                        real_foot = self.G.crop_parts(real_c.clone(), torch.cat((real_img_tmp['image'].clone(), real_img_tmp['mask'].clone()),dim=1), output_size=self.hand_foot_dist_res, parts=['left_foot', 'right_foot'])
                        real_foot = torch.cat([v for v in real_foot.values()])
                        real_foot_img = real_foot[:,:3].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_foot_mask = real_foot[:,3:].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_foot_logits = self.run_D_foot({"image": real_foot_img, 'mask': real_foot_mask}, real_c.clone(), blur_sigma=blur_sigma, lod=lod)
                    else:
                        real_foot = self.G.crop_parts(real_c.clone(), real_img_tmp['image'].clone(), output_size=self.hand_foot_dist_res, parts=['left_foot', 'right_foot'])
                        real_foot = torch.cat([v for v in real_foot.values()])
                        real_foot = real_foot.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                        real_foot_logits = self.run_D_foot({"image": real_foot}, real_c.clone(), blur_sigma=blur_sigma, lod=lod)
                    training_stats.report('Foot_Loss/scores/real', real_foot_logits)
                    training_stats.report('Foot_Loss/signs/real', real_foot_logits.sign())

                    loss_foot_Dreal = torch.tensor(0.0).to(real_foot_logits.device)
                    if phase in ['Dmain', 'Dboth']:
                        loss_foot_Dreal = torch.nn.functional.softplus(-real_foot_logits)
                        training_stats.report('Foot_Loss/D/loss', loss_foot_Dgen + loss_foot_Dreal)

                if self.use_patch_dist:
                    if self.use_mask:
                        d_patch, patch_params = self.extract_patches(torch.cat((real_img_tmp['image'].clone(), real_img_tmp['mask'].clone()),dim=1))
                        d_real_patch_input = {'image': d_patch[:,:3], 'mask': d_patch[:,3:]}
                    else:
                        d_patch, patch_params = self.extract_patches(real_img_tmp['image'].clone())
                        d_real_patch_input = {'image': d_patch}
                    real_patch_logits = self.run_D_patch(d_real_patch_input, real_c.clone(), blur_sigma=blur_sigma, lod=lod, patch_params=patch_params)
                    training_stats.report('Patch_Loss/scores/real', real_patch_logits)
                    training_stats.report('Patch_Loss/signs/real', real_patch_logits.sign())

                    loss_patch_Dreal = 0
                    if phase in ['Dmain', 'Dboth']:
                        loss_patch_Dreal = torch.nn.functional.softplus(-real_patch_logits)
                        training_stats.report('Patch_Loss/D/loss', loss_patch_Dgen + loss_patch_Dreal)

                loss_Dr1 = 0.0
                loss_face_Dr1 = 0.0
                loss_patch_Dr1 = 0.0
                loss_handfoot_Dr1 = torch.tensor(0.0).to(real_logits.device)
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        if self.use_mask:
                            with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['mask']], create_graph=True, only_inputs=True)
                                r1_grads_image = r1_grads[0]
                                r1_grads_mask = r1_grads[1]
                            r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                        else:
                            with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                                r1_grads_image = r1_grads[0]
                            r1_penalty = r1_grads_image.square().sum([1,2,3]) #+ r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        if self.use_mask:
                            with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['mask']], create_graph=True, only_inputs=True)
                                r1_grads_image = r1_grads[0]
                                r1_grads_mask = r1_grads[1]
                            r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                        else:
                            with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                                r1_grads_image = r1_grads[0]
                            r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

                    if self.use_face_dist:
                        if self.dual_discrimination:
                            if self.use_mask:
                                with torch.autograd.profiler.record_function('r1_grads_f'), conv2d_gradfix.no_weight_gradients():
                                    r1_grads = torch.autograd.grad(outputs=[real_face_logits.sum()], inputs=[real_face_img, real_face_mask], create_graph=True, only_inputs=True)
                                    r1_grads_image = r1_grads[0]
                                    r1_grads_mask = r1_grads[1]
                                r1_penalty_f = r1_grads_image.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                            else:
                                with torch.autograd.profiler.record_function('r1_grads_f'), conv2d_gradfix.no_weight_gradients():
                                    r1_grads = torch.autograd.grad(outputs=[real_face_logits.sum()], inputs=[real_face], create_graph=True, only_inputs=True)
                                    r1_grads_image = r1_grads[0]
                                r1_penalty_f = r1_grads_image.square().sum([1,2,3])
                        else: # single discrimination
                            if self.use_mask:
                                with torch.autograd.profiler.record_function('r1_grads_f'), conv2d_gradfix.no_weight_gradients():
                                    r1_grads = torch.autograd.grad(outputs=[real_face_logits.sum()], inputs=[real_face_img, real_face_mask], create_graph=True, only_inputs=True)
                                    r1_grads_image = r1_grads[0]
                                    r1_grads_mask = r1_grads[1]
                                r1_penalty_f = r1_grads_image.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                            else:
                                with torch.autograd.profiler.record_function('r1_grads_f'), conv2d_gradfix.no_weight_gradients():
                                    r1_grads = torch.autograd.grad(outputs=[real_face_logits.sum()], inputs=[real_face], create_graph=True, only_inputs=True)
                                    r1_grads_image = r1_grads[0]
                                r1_penalty_f = r1_grads_image.square().sum([1,2,3])
                        loss_face_Dr1 = r1_penalty_f * (r1_gamma / 2)
                        training_stats.report('Face_Loss/r1_penalty', r1_penalty_f)
                        training_stats.report('Face_Loss/D/reg', loss_face_Dr1)

                    if self.use_hand_dist or self.use_foot_dist:
                        if self.use_hand_dist and self.use_foot_dist:
                            if random.random()<0.5:
                                D_hand_foot_logits = [real_hand_logits.sum()]
                                if self.use_mask:
                                    D_hand_foot_inputs = [real_hands_img, real_hands_mask]
                                else:
                                    D_hand_foot_inputs = [real_hands]
                            else:
                                D_hand_foot_logits = [real_foot_logits.sum()]
                                if self.use_mask:
                                    D_hand_foot_inputs = [real_foot_img, real_foot_mask]
                                else:
                                    D_hand_foot_inputs = [real_foot]
                        elif self.use_hand_dist:
                            D_hand_foot_logits = [real_hand_logits.sum()]
                            if self.use_mask:
                                D_hand_foot_inputs = [real_hands_img, real_hands_mask]
                            else:
                                D_hand_foot_inputs = [real_hands]
                        else:
                            D_hand_foot_logits = [real_foot_logits.sum()]
                            if self.use_mask:
                                D_hand_foot_inputs = [real_foot_img, real_foot_mask]
                            else:
                                D_hand_foot_inputs = [real_foot]
                        if self.dual_discrimination:
                            with torch.autograd.profiler.record_function('r1_grads_h'), conv2d_gradfix.no_weight_gradients():
                                r1_grads = torch.autograd.grad(outputs=D_hand_foot_logits, inputs=D_hand_foot_inputs, create_graph=True, only_inputs=True)
                                r1_grads_image = r1_grads[0]
                                if self.use_mask:
                                    r1_grads_mask = r1_grads[1]
                            if self.use_mask:
                                r1_penalty_h = r1_grads_image.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                            else:
                                r1_penalty_h = r1_grads_image.square().sum([1,2,3]) #+ r1_grads_image_raw.square().sum([1,2,3])
                        else: # single discrimination
                            with torch.autograd.profiler.record_function('r1_grads_h'), conv2d_gradfix.no_weight_gradients():
                                r1_grads = torch.autograd.grad(outputs=D_hand_foot_logits, inputs=D_hand_foot_inputs, create_graph=True, only_inputs=True)
                                r1_grads_image = r1_grads[0]
                                if self.use_mask:
                                    r1_grads_mask = r1_grads[1]
                            if self.use_mask:
                                r1_penalty_h = r1_grads_image.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                            else:
                                r1_penalty_h = r1_grads_image.square().sum([1,2,3]) #+ r1_grads_image_raw.square().sum([1,2,3])
                        loss_handfoot_Dr1 = r1_penalty_h * (r1_gamma / 2)
                        training_stats.report('HandFoot_Loss/r1_penalty', r1_penalty_h)
                        training_stats.report('HandFoot_Loss/D/reg', loss_handfoot_Dr1)

                    if self.use_patch_dist:
                        if self.dual_discrimination:
                            if self.use_mask:
                                with torch.autograd.profiler.record_function('r1_grads_p'), conv2d_gradfix.no_weight_gradients():
                                    r1_grads = torch.autograd.grad(outputs=[real_patch_logits.sum()], inputs=[d_patch[:,:3], d_patch[:,3:]], create_graph=True, only_inputs=True)
                                    r1_grads_image = r1_grads[0]
                                    r1_grads_mask = r1_grads[1]
                                r1_penalty_p = r1_grads_image.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                            else:
                                with torch.autograd.profiler.record_function('r1_grads_p'), conv2d_gradfix.no_weight_gradients():
                                    r1_grads = torch.autograd.grad(outputs=[real_patch_logits.sum()], inputs=[d_patch], create_graph=True, only_inputs=True)
                                    r1_grads_image = r1_grads[0]
                                r1_penalty_p = r1_grads_image.square().sum([1,2,3]) #+ r1_grads_image_raw.square().sum([1,2,3])
                        else: # single discrimination
                            if self.use_mask:
                                with torch.autograd.profiler.record_function('r1_grads_p'), conv2d_gradfix.no_weight_gradients():
                                    r1_grads = torch.autograd.grad(outputs=[real_patch_logits.sum()], inputs=[d_patch[:,:3], d_patch[:,3:]], create_graph=True, only_inputs=True)
                                    r1_grads_image = r1_grads[0]
                                    r1_grads_mask = r1_grads[1]
                                r1_penalty_p = r1_grads_image.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                            else:
                                with torch.autograd.profiler.record_function('r1_grads_p'), conv2d_gradfix.no_weight_gradients():
                                    r1_grads = torch.autograd.grad(outputs=[real_patch_logits.sum()], inputs=[d_patch], create_graph=True, only_inputs=True)
                                    r1_grads_image = r1_grads[0]
                                r1_penalty_p = r1_grads_image.square().sum([1,2,3])
                        loss_patch_Dr1 = r1_penalty_p * (r1_gamma / 2)
                        training_stats.report('Patch_Loss/r1_penalty', r1_penalty_p)
                        training_stats.report('Patch_Loss/D/reg', loss_patch_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                if self.use_face_dist:
                    loss_Dreal = loss_Dreal + loss_face_Dreal * self.face_weight
                    loss_Dr1 = loss_Dr1 + loss_face_Dr1 * self.face_weight
                if self.use_foot_dist or self.use_hand_dist:
                    if self.use_foot_dist:
                        loss_Dreal = loss_Dreal + loss_foot_Dreal.mean() * self.hand_foot_weight
                    if self.use_hand_dist:
                        loss_Dreal = loss_Dreal + loss_hand_Dreal.mean() * self.hand_foot_weight
                    loss_Dr1 = loss_Dr1 + loss_handfoot_Dr1.mean() * self.hand_foot_weight
                if self.use_patch_dist:
                    loss_Dreal = loss_Dreal + loss_patch_Dreal
                    loss_Dr1 = loss_Dr1 + loss_patch_Dr1
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
