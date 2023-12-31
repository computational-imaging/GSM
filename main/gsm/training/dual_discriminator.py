# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Discriminator architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import numpy as np
import torch
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from training.networks_stylegan2 import DiscriminatorBlock, MappingNetwork, DiscriminatorEpilogue, DiscriminatorBlockPatch
from training.networks_utils import ScalarEncoder1d
from torch_utils import misc
from utils.dataset_utils import parse_raw_labels, parse_raw_labels_ffhq

@persistence.persistent_class
class SingleDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        sr_upsample_factor  = 1,        # Ignored for SingleDiscriminator
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        img = img['image']

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------

def filtered_resizing(image_orig_tensor, size, f, filter_mode='antialiased'):
    if filter_mode == 'antialiased':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
    elif filter_mode == 'classic':
        ada_filtered_64 = upfirdn2d.upsample2d(image_orig_tensor, f, up=2)
        ada_filtered_64 = torch.nn.functional.interpolate(ada_filtered_64, size=(size * 2 + 2, size * 2 + 2), mode='bilinear', align_corners=False)
        ada_filtered_64 = upfirdn2d.downsample2d(ada_filtered_64, f, down=2, flip_filter=True, padding=-1)
    elif filter_mode == 'none':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False)
    elif type(filter_mode) == float:
        assert 0 < filter_mode < 1

        filtered = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        aliased  = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=False)
        ada_filtered_64 = (1 - filter_mode) * aliased + (filter_mode) * filtered

    return ada_filtered_64

#----------------------------------------------------------------------------

@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise        = 0,        # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        label_type          = 'smpl',
        final_res           = 4,
        is_down             = True,
        progressive_training= False,
        ds_pg               = False,
        use_mask            = False,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.final_res = final_res
        self.is_down = is_down
        self.ds_pg = ds_pg
        self.label_type = label_type
        self.use_mask = use_mask

        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution // final_res) + 2)
        # self.img_channels = img_channels
        if self.use_mask:
            self.img_channels = img_channels + 1
            assert self.img_channels == 4
        else:
            self.img_channels = img_channels
            assert self.img_channels == 3
        self.block_resolutions = [(2 ** (i - 2))*final_res  for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [final_res]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[final_res]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, with_rgb=progressive_training, ds_pg=ds_pg, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[final_res], cmap_dim=cmap_dim, resolution=final_res, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.disc_c_noise = disc_c_noise

    def parse_raw_labels(self, c):
        if self.label_type == "flame":
            labels = parse_raw_labels_ffhq(c)
            c = torch.concat([c[:, :25], labels["jaw_pose"], labels["expression"]], dim=-1)
        elif self.label_type == "smpl":
            labels = parse_raw_labels(c)
            c = torch.concat([c[:, :25], labels["global_orient"], labels["body_pose"], labels["betas"]], dim=-1)
        return c

    def forward(self, img, c, update_emas=False, lod=None, **block_kwargs):
        if c.shape[1] > 25:
            c_new = self.parse_raw_labels(c)
        else:
            c_new = c
        c_new =c_new[..., :self.c_dim]
        if self.use_mask:
            if img['mask'].shape[1] == 3:
                img_mask = torch.mean(img['mask'], dim=1, keepdim=True)
            else:
                assert img['mask'].shape[1] == 1
                # assert img['mask'].max()<=1 and img['mask'].min()>=0
                img_mask = img['mask']
            img = torch.cat((img['image'], img_mask), dim=1)
        else:
            img = img['image']

        if lod is None:
            lod = 0

        _ = update_emas # unused
        x = None

        if self.ds_pg:
            for res_log2 in range(self.img_resolution_log2, 2, -1):
                res = (2 ** (res_log2 - 2))*self.final_res
                cur_lod = self.img_resolution_log2 - res_log2

                if lod < cur_lod + 1:
                    block = getattr(self, f'b{res}')
                    if cur_lod <= lod < cur_lod + 1:
                        # print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, first')
                        x, img = block(x, img, alpha=1e4, **block_kwargs)
                    elif cur_lod -1 < lod < cur_lod:
                        alpha = lod -  np.floor(lod)
                        # print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, second')
                        x, img = block(x, img, alpha=alpha, **block_kwargs)
                    else:
                        # print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, third')
                        x, img = block(x, img, alpha=None, **block_kwargs)

                if self.is_down:
                    if lod > cur_lod:
                        # print(f'cur_lod:{cur_lod}, lod:{lod}, downsample!')
                        img = torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2, padding=0)
                else:
                    if cur_lod < lod < cur_lod + 1:
                        img = torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2, padding=0)
        else:
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0:
                c_new = c_new + torch.randn_like(c_new) * c_new.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c_new)

        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class PatchDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise        = 0,        # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        final_res           = 4,
        is_down             = True,
        progressive_training= False,
        ds_pg               = False,
        hyper_mod           = False,
        use_mask            = False,
        **unused_kwargs
    ):
        super().__init__()
        c_dim= 107
        self.c_dim = c_dim
        self.final_res = final_res
        self.is_down = is_down
        self.ds_pg = ds_pg
        self.hyper_mod = hyper_mod
        self.use_mask = use_mask

        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution // final_res) + 2)
        # self.img_channels = img_channels
        if self.use_mask:
            self.img_channels = img_channels + 1
            assert self.img_channels == 4
        else:
            self.img_channels = img_channels
            assert self.img_channels == 3
        self.block_resolutions = [(2 ** (i - 2))*final_res  for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [final_res]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[final_res]

        self.scalar_enc = ScalarEncoder1d(coord_dim=2, x_multiplier=1000.0, const_emb_dim=256)
        assert self.scalar_enc.get_dim() > 0

        if (self.c_dim == 0) and (self.scalar_enc is None):
            cmap_dim = 0

        if hyper_mod:
            hyper_mod_dim = 512
            self.hyper_mod_mapping = MappingNetwork(
                z_dim=0, c_dim=self.scalar_enc.get_dim(),
                w_dim=hyper_mod_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        else:
            self.hyper_mod_mapping = None
            hyper_mod_dim = 0

        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)
        total_conditioning_dim = c_dim + self.scalar_enc.get_dim()
        cur_layer_idx = 0
        for i, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            down = 2
            block = DiscriminatorBlockPatch(
                in_channels, tmp_channels, out_channels, resolution=res, first_layer_idx=cur_layer_idx, use_fp16=use_fp16,
                down=down, c_dim=hyper_mod_dim, hyper_mod=hyper_mod, ds_pg=ds_pg, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=total_conditioning_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[final_res], cmap_dim=cmap_dim, resolution=final_res, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.disc_c_noise = disc_c_noise

    def forward(self, img, c, patch_params=None, update_emas=False, lod=None, **block_kwargs):
        # image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
        # img = torch.cat([img['image'], image_raw], 1)
        c_new = c[:, :107].clone()
        img = img['image']

        if not self.scalar_enc is None:
            patch_params_cond = patch_params['offsets'] # [batch_size, 2]
            misc.assert_shape(patch_params_cond, [img.shape[0], 2])
            patch_scale_embs = self.scalar_enc(patch_params_cond) # [batch_size, fourier_dim]
            c_all = c_new if not self.hyper_mod else torch.cat([c_new, patch_scale_embs], dim=1) # [batch_size, c_dim + fourier_dim]

        if not self.scalar_enc is None and self.hyper_mod:
            hyper_mod_c = self.hyper_mod_mapping(z=None, c=patch_scale_embs) # [batch_size, 512]
        else:
            hyper_mod_c = None

        if lod is None:
            lod = 0

        _ = update_emas # unused
        x = None


        if self.ds_pg:

            for res_log2 in range(self.img_resolution_log2, 2, -1):
                res = (2 ** (res_log2 - 2))*self.final_res
                cur_lod = self.img_resolution_log2 - res_log2

                if lod < cur_lod + 1:
                    block = getattr(self, f'b{res}')
                    if cur_lod <= lod < cur_lod + 1:
                        # print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, first')
                        x, img = block(x, img, c=hyper_mod_c, alpha=1e4, **block_kwargs)
                    elif cur_lod -1 < lod < cur_lod:
                        alpha = lod -  np.floor(lod)
                        # print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, second')
                        x, img = block(x, img, c=hyper_mod_c, alpha=alpha, **block_kwargs)
                    else:
                        # print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, third')
                        x, img = block(x, img, c=hyper_mod_c, alpha=None, **block_kwargs)

                if self.is_down:
                    if lod > cur_lod:
                        # print(f'cur_lod:{cur_lod}, lod:{lod}, downsample!')
                        img = torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2, padding=0)
                else:
                    if cur_lod < lod < cur_lod + 1:
                        img = torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2, padding=0)
        else:
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                x = block(x, img, c=hyper_mod_c, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c_all = c_all + torch.randn_like(c_all) * c_all.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c_all)
        # print(cmap)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'



#----------------------------------------------------------------------------

@persistence.persistent_class
class DummyDualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        img_channels *= 2

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

        self.raw_fade = 1

    def forward(self, img, c, update_emas=False, **block_kwargs):
        self.raw_fade = max(0, self.raw_fade - 1/(500000/32))

        image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter) * self.raw_fade
        img = torch.cat([img['image'], image_raw], 1)

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------
