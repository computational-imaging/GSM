# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from typing import Dict
import yaml
import numpy as np
import torch
from deformer.smpl_deformer import smpl_init_kwargs, flame_init_kwargs
from scene.gaussian_volume import VolumeGaussianModel
from gaussian_renderer import render
from scene.gaussian_model_shells import ShellGaussianModel
from scene.gaussian_model_tets import TetGaussianModel
from torch import nn
from torch_utils import persistence
from torchvision.ops import roi_align
from training.dual_discriminator import filtered_resizing
from training.networks_stylegan2 import Attdecoder
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.ray_sampler import RaySampler
from training.volumetric_rendering.renderer import ImportanceRenderer
from utils.image_utils import get_bg_color
from utils.dataset_utils import create_new_camera, parse_raw_labels, parse_raw_labels_ffhq
from utils.graphics_utils import geom_transform_points_batched
from training.dual_discriminator import filtered_resizing
from torch_utils.ops import upfirdn2d


class PipelineParams():
        def __init__(self):
            self.convert_SHs_python = True
            self.compute_cov3D_python = False
            self.debug = False

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,
        img_channels,
        progressive_tex = False,
        tex_init_res =  128,
        tex_final_res = 512,
        decode_first       = 'all',
        gaussian_model     = "shells",
        num_shells         = 5,
        offset_len         = 0.1,
        max_scaling        = 0.02,
        total_num_gaussians= 80000,
        rotate_gaussians   = False,
        bbox               = None,
        scale_act          = 'exp',
        bg_color           = 'white',
        base_shell_path    = None,
        smpl_transl        = (0.0, 0.0, 0.0),
        smpl_scale         = (1.0, 1.0, 1.0),
        lr_multiplier_color=1.0,
        lr_multiplier_opacity=1.0,
        lr_multiplier_scaling=1.0,
        lr_multiplier_rotation=1.0,
        old_init            = False,
        scale_downsample = False,
        opacity_downsample = False,
        opacity_blur = 2,
        scale_blur = 2,
        device = 'cuda:0',          # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        shrunk_ref_mesh     = None,
        replace_faces       = None, # Arguments for editing.
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim if gaussian_model == "volume" else 25
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.progressive_tex = progressive_tex
        self.tex_init_res = tex_init_res
        self.tex_final_res = tex_final_res
        self.alpha = 0
        self.current_device = device
        self.scale_downsample = scale_downsample
        self.opacity_downsample = opacity_downsample

        if self.scale_downsample or self.opacity_downsample:

            self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device='cpu')
            self.filter_mode = 'antialiased'
            self.opacity_blur = opacity_blur
            self.scale_blur = scale_blur

        if gaussian_model == 'tets':
            GaussianModel = TetGaussianModel
            assert base_shell_path is not None
        elif gaussian_model == 'shells':
            GaussianModel = ShellGaussianModel
            assert base_shell_path is not None
        else:
            GaussianModel = VolumeGaussianModel
            assert bbox is not None
            num_shells = 3
            bbox = torch.tensor(bbox).to(self.current_device)

        img_color_channels = 0
        if decode_first == 'all':
            img_channels = 64
        elif decode_first == 'wo_color':
            img_channels = 64 * num_shells
        else:
            img_channels = 64 * num_shells
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=tex_final_res, img_channels=img_channels,
                                          mapping_kwargs=mapping_kwargs,
                                          decode_first=decode_first, num_shells=num_shells,
                                          lr_multiplier_color=lr_multiplier_color,
                                          lr_multiplier_opacity = lr_multiplier_opacity,
                                          lr_multiplier_scaling=lr_multiplier_scaling,
                                          lr_multiplier_rotation = lr_multiplier_rotation,
                                          old_init = old_init,
                                          **synthesis_kwargs)

        self.texture_res = tex_final_res
        self.rendering_kwargs = rendering_kwargs
        self.decode_first = decode_first

        self._last_planes = None
        if "smpl" in base_shell_path.lower():
            my_smpl_init_kwargs = smpl_init_kwargs.copy()
        elif "flame" in base_shell_path.lower():
            my_smpl_init_kwargs = flame_init_kwargs.copy()

        self._template_type = my_smpl_init_kwargs["model_type"]
        my_smpl_init_kwargs["gender"] = "neutral"

        NUM_SHELLS = num_shells
        self.num_shells = num_shells
        OFFSET_LEN = offset_len

        if decode_first == 'none':
            self.torgb_color = Attdecoder(img_channels, num_shells*3, lr_multiplier=lr_multiplier_color)
            self.torgb_opacity = Attdecoder(img_channels,num_shells*1, lr_multiplier=lr_multiplier_opacity)
            self.torgb_scaling = Attdecoder(img_channels, num_shells*3, lr_multiplier=lr_multiplier_scaling, bias_init=-4.5, init_small=True)
            self.torgb_rotation = Attdecoder(img_channels, num_shells*4, lr_multiplier=lr_multiplier_rotation)

        if decode_first == 'wo_color':
            self.torgb_color = Attdecoder(img_channels, num_shells*3, lr_multiplier=lr_multiplier_color)

        self.gaussians = GaussianModel(base_shell_path=base_shell_path,
                                       sh_degree=0,
                                       num_shells=NUM_SHELLS,
                                       offset_len=OFFSET_LEN,
                                       smpl_init_kwargs=my_smpl_init_kwargs,
                                       smpl_can_params=None,
                                       smpl_transl=smpl_transl,
                                       smpl_scale=smpl_scale,
                                       rotate_gaussians=rotate_gaussians,
                                       total_points=total_num_gaussians,
                                       bbox=bbox,
                                       max_scaling=max_scaling,
                                       device= self.current_device,
                                       shrunk_ref_mesh=shrunk_ref_mesh,
                                       scale_act=scale_act,
                                       replace_faces=replace_faces)
        self.pipeline = PipelineParams()
        self.background = get_bg_color(bg_color)

        if self._template_type == "smpl":
            parts_vertices_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "assets", "smpl", "parts_vertices.yaml")
            # self.smpl_mesh_head_verts_ids = torch.from_numpy(np.loadtxt(head_verts_path, dtype=np.int64, delimiter=',')).to(self.current_device)
            with open(parts_vertices_path, 'r') as f:
                data = yaml.safe_load(f)
                self.smpl_mesh_head_verts_ids = torch.from_numpy(np.array(data["face"], dtype=np.int64)).to(self.current_device)
                self.smpl_mesh_left_foot_verts_ids = torch.from_numpy(np.array(data["left_foot"], dtype=np.int64)).to(self.current_device)
                self.smpl_mesh_right_foot_verts_ids = torch.from_numpy(np.array(data["right_foot"], dtype=np.int64)).to(self.current_device)
                self.smpl_mesh_right_hand_verts_ids = torch.from_numpy(np.array(data["right_hand"], dtype=np.int64)).to(self.current_device)
                self.smpl_mesh_left_hand_verts_ids = torch.from_numpy(np.array(data["left_hand"], dtype=np.int64)).to(self.current_device)

    def parse_raw_labels(self, c):
        if self._template_type == "smpl":
            return parse_raw_labels(c)
        elif self._template_type == "flame":
            return parse_raw_labels_ffhq(c)
        else:
            raise ValueError("Invalid model type!")

    def get_part_bbox(self, smpl_parameters, cameras, part="face"):
        """Get the pixel space bounding box of the face.
        Returns:
            face_bbox: (B, 2, 2) tensor of pixel space square bounding boxes.
        """
        # From smpl_parameters get the head parameters
        smpl_output = self.gaussians.deformer.smplx_model(**smpl_parameters)
        expand = 0.2
        if part == "face":
            part_vertices = smpl_output.vertices[:, self.smpl_mesh_head_verts_ids]
            expand = 0.5
        elif part == "left_foot":
            part_vertices = smpl_output.vertices[:, self.smpl_mesh_left_foot_verts_ids]
        elif part == "right_foot":
            part_vertices = smpl_output.vertices[:, self.smpl_mesh_right_foot_verts_ids]
        elif part == "right_hand":
            part_vertices = smpl_output.vertices[:, self.smpl_mesh_right_hand_verts_ids]
        elif part == "left_hand":
            part_vertices = smpl_output.vertices[:, self.smpl_mesh_left_hand_verts_ids]
        else:
            raise ValueError("Invalid part name! Should be one of face, left_foot, right_foot, left_hand, right_hand")

        # This is in screen space coordinates from -1 to 1
        part_2D = geom_transform_points_batched(part_vertices, cameras.full_proj_transform)[...,:2]

        xy_min, xy_max = part_2D.min(dim=1)[0], part_2D.max(dim=1)[0]
        part_bbox = torch.stack([xy_min, xy_max], dim=1)  # B, 2, 2
        center = (xy_min + xy_max) / 2.0  # B, 2

        # Make sure the bounding box is square
        longer_side_half = torch.max(part_bbox[:, 1] - part_bbox[:, 0], dim=-1, keepdim=True)[0] / 2  # (B, 1, 1)
        part_bbox = torch.stack([center - longer_side_half, center + longer_side_half], dim=1)

        # Increase the region by percentage defined by `expand` (B, 1, 2)
        part_bbox = part_bbox + expand * (part_bbox[:, 1] - part_bbox[:, 0]).unsqueeze(1) * torch.tensor([[-1, 1]]).to(part_bbox.device).reshape(1, 2, 1)
        part_bbox = part_bbox.clamp(-1, 1)

        # In Pixel space
        part_bbox = (part_bbox + 1.0) / 2.0 * torch.tensor([cameras.image_width, cameras.image_height]).to(part_bbox.device)
        part_bbox = part_bbox.long()

        return part_bbox

    def crop_parts(self, c, image, output_size=128, parts=["face"]) -> Dict[str, torch.Tensor]:
        """Crop specific parts from the image. Potentially scale.
        Args:
            c: label arrays
            image: (B, C, H, W) tensor of images
            output_size: size of the output image
            parts: list of parts to crop. Should be one of face, left_foot, right_foot, left_hand, right_hand
        Returns:
            cropped_resized_images: dictionary (B, C, output_size, output_size) tensor of cropped and resized images
        """
        if self._template_type != "smpl":
            return None
        assert all([p in ("face", "right_hand", "right_foot", "left_hand", "left_foot") for p in parts]), "Invalid part name!"
        bs = c.shape[0]
        labels = self.parse_raw_labels(c)
        cameras = create_new_camera(labels, self.img_resolution, self.img_resolution, image.device)
        smpl_kwargs = {}
        smpl_kwargs["body_pose"] = labels["body_pose"].reshape(-1, labels["body_pose"].shape[-1]).to(image.device)
        smpl_kwargs["betas"] = labels["betas"].reshape(-1, labels["betas"].shape[-1]).to(image.device)
        smpl_kwargs["global_orient"] = labels["global_orient"].reshape(-1, labels["global_orient"].shape[-1]).to(image.device)
        smpl_kwargs["transl"] = torch.zeros_like(smpl_kwargs["body_pose"][:, :3]).to(image.device)

        part_bbox = []
        for part in parts:
            part_bbox += [self.get_part_bbox(smpl_kwargs, cameras, part)]

        part_bbox = torch.stack(part_bbox, dim=1).reshape(bs, -1, 2, 2)
        assert part_bbox.shape[1] == len(parts)
        indices = torch.arange(0, bs).view(bs, 1).expand(-1, part_bbox.shape[1]).float().to(part_bbox.device)
        rois = torch.cat((indices.reshape(-1, 1), part_bbox.reshape(-1, 4)), dim=1)

        cropped_resized_images = roi_align(image, rois, (output_size, output_size)).reshape(bs, len(parts), image.shape[1], output_size, output_size)
        output_dict = {}
        for i, part in enumerate(parts):
            output_dict[part] = cropped_resized_images[:, i]
        return output_dict

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        c = c[:, 0:self.c_dim]
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)


    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False,
                  cache_backbone=False, use_cached_backbone=False, lod=None,
                  perturb_pts=None, lr_mul=1.0, recompute_scaling=False, orient = None,
                  replace_idx = None, replace_color = None, replace_opacity = None,
                  replace_scale = None, replace_rotation = None, 
                  **synthesis_kwargs):
        """Returns rendering of range [-1, 1]"""
        if lod is None:
            lod = 0
        cur_img_resolution = int(self.img_resolution // (2**np.floor(lod)))

        c_clone = c.clone().detach()

        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            # backbone_outputs = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            if self.decode_first == 'all':
                color_, opacity_, scaling_, rotation_ = self.backbone.synthesis(ws, update_emas=update_emas, lr_mul=lr_mul, **synthesis_kwargs)
                if self.progressive_tex:
                    self.texture_res = int(np.rint(self.tex_init_res * (1 - self.alpha) + self.tex_final_res * self.alpha))
                    color_ = filtered_resizing(color_, size=self.texture_res, f=0, filter_mode='antialiased')
                    opacity_ = filtered_resizing(opacity_, size=self.texture_res, f=0, filter_mode='antialiased')
                    scaling_ = filtered_resizing(scaling_, size=self.texture_res, f=0, filter_mode='antialiased')
                    rotation_ = filtered_resizing(rotation_, size=self.texture_res, f=0, filter_mode='antialiased')

                if self.scale_downsample:
                    down_tensor = torch.nn.functional.avg_pool2d(scaling_, kernel_size=self.scale_blur, stride=self.scale_blur, padding=0)
                    scaling_ = filtered_resizing(down_tensor, size=scaling_.shape[-1], f=self.resample_filter, filter_mode=self.filter_mode)

                if self.opacity_downsample:
                    down_tensor = torch.nn.functional.avg_pool2d(opacity_, kernel_size=self.opacity_blur, stride=self.opacity_blur, padding=0)
                    opacity_ = filtered_resizing(down_tensor, size=opacity_.shape[-1], f=self.resample_filter, filter_mode=self.filter_mode)

                color_ = color_.reshape(ws.shape[0],-1,3,self.texture_res,self.texture_res).permute(0,1,3,4,2)
                opacity_ = opacity_.reshape(ws.shape[0],-1,1,self.texture_res,self.texture_res).permute(0,1,3,4,2)
                rotation_ = rotation_.reshape(ws.shape[0],-1,4,self.texture_res,self.texture_res).permute(0,1,3,4,2)
                scaling_ = scaling_.reshape(ws.shape[0],-1,3,self.texture_res,self.texture_res).permute(0,1,3,4,2)



                planes = torch.cat([color_, opacity_, scaling_,rotation_], dim = -1)
            elif self.decode_first == 'wo_color':
                planes_, opacity_, scaling_, rotation_ = self.backbone.synthesis(ws, update_emas=update_emas, lr_mul=lr_mul, **synthesis_kwargs)
                if self.progressive_tex:
                    self.texture_res = int(np.rint(self.tex_init_res * (1 - self.alpha) + self.tex_final_res * self.alpha))
                    planes_ = filtered_resizing(planes_, size=self.texture_res, f=0, filter_mode='antialiased')
                    opacity_ = filtered_resizing(opacity_, size=self.texture_res, f=0, filter_mode='antialiased')
                    scaling_ = filtered_resizing(scaling_, size=self.texture_res, f=0, filter_mode='antialiased')
                    rotation_ = filtered_resizing(rotation_, size=self.texture_res, f=0, filter_mode='antialiased')

                if self.scale_downsample:
                    down_tensor = torch.nn.functional.avg_pool2d(scaling_, kernel_size=self.scale_blur, stride=self.scale_blur, padding=0)
                    scaling_ = filtered_resizing(down_tensor, size=scaling_.shape[-1], f=self.resample_filter, filter_mode=self.filter_mode)

                if self.opacity_downsample:
                    down_tensor = torch.nn.functional.avg_pool2d(opacity_, kernel_size=self.opacity_blur, stride=self.opacity_blur, padding=0)
                    opacity_ = filtered_resizing(down_tensor, size=opacity_.shape[-1], f=self.resample_filter, filter_mode=self.filter_mode)

                planes_color = planes_.reshape(ws.shape[0],self.num_shells,64,self.texture_res,self.texture_res).permute(0,1,3,4,2)
                opacity_ = opacity_.reshape(ws.shape[0],-1,1,self.texture_res,self.texture_res).permute(0,1,3,4,2)
                rotation_ = rotation_.reshape(ws.shape[0],-1,4,self.texture_res,self.texture_res).permute(0,1,3,4,2)
                scaling_ = scaling_.reshape(ws.shape[0],-1,3,self.texture_res,self.texture_res).permute(0,1,3,4,2)

                planes = torch.cat([opacity_, scaling_,rotation_], dim = -1)
            else:
                planes = self.backbone.synthesis(ws, update_emas=update_emas, lr_mul=lr_mul, **synthesis_kwargs)
                if self.progressive_tex:
                    self.texture_res = int(np.rint(self.tex_init_res * (1 - self.alpha) + self.tex_final_res * self.alpha))
                    planes = filtered_resizing(planes, size=self.texture_res, f=0, filter_mode='antialiased')
                planes = planes.reshape(ws.shape[0],self.num_shells,64,self.texture_res,self.texture_res).permute(0,1,3,4,2)

        if cache_backbone:
            self._last_planes = planes

        perturb_pts = perturb_pts if perturb_pts is not None else self.rendering_kwargs.get('perturb_pts', False)
        if perturb_pts:
            self.gaussians.perturb_points()

        rendering = []
        scales = []
        rotations = []
        colors = []
        opacities = []
        masks = []

        for bs in range(len(planes)):
            if self.decode_first == 'all':
                self.gaussians.set_texture(planes[bs])
                self.gaussians.set_features_dc(planes[bs][..., :3] )
                self.gaussians.set_opacity(planes[bs][..., 3:4] )
                self.gaussians.set_scaling(planes[bs][..., 4:7])
                self.gaussians.set_rotation(planes[bs][..., 7:])
            elif self.decode_first == 'wo_color':
                self.gaussians.set_texture(planes[bs])
                self.gaussians.set_features_dc(planes_color[bs], self.torgb_color)
                self.gaussians.set_opacity(planes[bs][..., 0:1] )
                self.gaussians.set_scaling(planes[bs][..., 1:4])
                self.gaussians.set_rotation(planes[bs][..., 4:])
            else:
                self.gaussians.set_texture(planes[bs])
                self.gaussians.set_features_dc(planes[bs], self.torgb_color)
                self.gaussians.set_opacity(planes[bs], self.torgb_opacity)
                self.gaussians.set_scaling(planes[bs], self.torgb_scaling)
                self.gaussians.set_rotation(planes[bs], self.torgb_rotation)

            with torch.no_grad():
                labels = self.parse_raw_labels(c_clone[bs])
                camera = create_new_camera(labels, cur_img_resolution, cur_img_resolution, planes.device)
                smpl_params = {}
                if self._template_type == "flame":
                    smpl_params["expression"] = labels["expression"].to(device=planes.device).reshape(1, -1)
                    smpl_params["global_orient"] = labels["global_orient"].to(device=planes.device).reshape(1, -1)
                    smpl_params["jaw_pose"] = labels["jaw_pose"].to(device=planes.device).reshape(1, -1)
                    smpl_params["betas"] = labels["betas"].to(device=planes.device).reshape(1, -1)
                    smpl_params["global_orient"][:] = 0.0
                elif self._template_type == "smpl":
                    smpl_params["body_pose"] = labels["body_pose"].reshape(1, -1).to(planes.device)
                    smpl_params["betas"] = labels["betas"].reshape(1, -1).to(planes.device)
                    if orient == None:
                        smpl_params["global_orient"] = labels["global_orient"].reshape(1, -1).to(planes.device)
                    else:
                        smpl_params["global_orient"] =  orient.to(planes.device)
                    smpl_params["transl"] = torch.zeros_like(smpl_params["body_pose"][:, :3]).to(planes.device)
                else:
                    raise ValueError("Invalid model type!")

            out = render(camera, self.gaussians, self.pipeline, self.background.to(planes.device),
                         deformation_kwargs=smpl_params, recompute_scaling=recompute_scaling, device_ = planes.device,
                         replace_idx = replace_idx, replace_color = replace_color, replace_opacity = replace_opacity,
                         replace_scale = replace_scale, replace_rotation = replace_rotation)
            mask_ = render(camera, self.gaussians, self.pipeline, torch.zeros_like(self.background).to(planes.device),
                             override_color=torch.ones_like(self.gaussians._xyz).to(planes.device),
                             recompute_scaling=recompute_scaling,
                             deformation_kwargs=smpl_params, device_ = planes.device)["render"]

            rendering.append(out['render'])
            masks.append(mask_)
            scales.append(out['scales'])
            rotations.append(out['rotations'])
            opacities.append(out['opacities'])
            colors.append(out['colors'])


        rendering = torch.stack(rendering)
        rendering = (rendering - 0.5) / 0.5
        mask = torch.stack(masks)

        # get the image under the required resolution
        for _ in range(int(np.floor(lod)), 0 , -1):
            rendering = torch.nn.functional.interpolate(rendering, scale_factor=2, mode='nearest')
            mask = torch.nn.functional.interpolate(mask, scale_factor=2, mode='nearest')

        if self.decode_first == 'all':
            try : 
                  return {'image': rendering, 'scaling': planes[..., 4: 7],
                    'opacity': self.gaussians.opacity_activation(planes[..., 3: 4]), "mask": mask,
                    'replace_idx': self.gaussians.replace_idx, 'replace_scale': torch.stack(scales),
                    'replace_rotation': torch.stack(rotations), 'replace_opacity': torch.stack(opacities),
                    'replace_color': torch.stack(colors)}
            except:
                return {'image': rendering, 'scaling': planes[..., 4: 7],
                        'opacity': self.gaussians.opacity_activation(planes[..., 3: 4]), "mask": mask,
                        'replace_idx': replace_idx, 'replace_scale': torch.stack(scales),
                        'replace_rotation': torch.stack(rotations), 'replace_opacity': torch.stack(opacities),
                        'replace_color': torch.stack(colors)}
        elif self.decode_first == 'wo_color':
            return {'image': rendering, 'scaling': planes[..., 1: 4],
                    'opacity': self.gaussians.opacity_activation(planes[..., 0: 1]), "mask": mask}
        else:
            return {'image': rendering, 'scaling': torch.stack(scales), "mask": mask}

    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False,
                cache_backbone=False, use_cached_backbone=False, lod=None, perturb_pts=None, lr_mul=1.0, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, cache_backbone=cache_backbone,
                                use_cached_backbone=use_cached_backbone, lod=lod, perturb_pts=perturb_pts, lr_mul=lr_mul,
                                **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
