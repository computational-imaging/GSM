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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from simple_knn._C import distCUDA2


def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, override_color:torch.Tensor=None, recompute_scaling:bool=False,
           deformation_kwargs = None, replace_idx = None, replace_color = None, replace_opacity = None,
           replace_scale = None, replace_rotation = None, device_='cuda'
           ):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    deformation_kwargs = {} if deformation_kwargs is None else deformation_kwargs
    if not isinstance(pc.get_xyz, torch.Tensor):
        means3D = pc.get_xyz(cache_deformation_output=True, use_cached=False, **deformation_kwargs)
    else:
        means3D = pc.get_xyz

    if len(means3D.shape) == 3:
         means3D = means3D.squeeze(0)
    # print(means3D.shape)
    # ksk
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=means3D.requires_grad, device=device_) + 0
    if screenspace_points.requires_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    if replace_idx is not None and replace_opacity is not None:
        opacity[replace_idx==1] = replace_opacity[replace_idx==1]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if recompute_scaling:
            dist2 = torch.clamp(distCUDA2(means3D), min=0.0000001, max=0.0001)
            scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        else:
            scales = pc.get_scaling

        if replace_idx is not None and replace_scale is not None:
            scales[replace_idx==1] = replace_scale[replace_idx==1]

        if not isinstance(pc.get_rotation, torch.Tensor):
            rotations = pc.get_rotation(cache_deformation_output=True, use_cached=True, **deformation_kwargs)
        else:
            rotations = pc.get_rotation
        if rotations.ndim == 3:
            rotations = rotations.squeeze(0)
        
        if replace_idx is not None and replace_rotation is not None:
            rotations[replace_idx==1] = replace_rotation[replace_idx==1]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            # # colors_precomp = pc.get_my_features
            # colors_precomp = torch.clamp_min(pc.get_features[:, :3], 0.0)  # TODO: better filter
            colors_precomp = torch.clamp_min(pc.get_features, 0.0)
            # colors_precomp = torch.clamp_min(pc.get_features[:, 0], 0.0)  # TODO: better filter
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if replace_idx is not None and replace_color is not None:
        colors_precomp[replace_idx==1] = replace_color[replace_idx==1]

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    # print("(means2D)", means2D.shape, means2D.device)
    # print("(means3D)", means3D.shape, means3D.device)
    # print("(shs)", shs.shape, shs.device)
    # print("(colors_precomp)", colors_precomp.shape, colors_precomp.device)
    # print("(opacity)", opacity.shape, opacity.device)
    # print("(scales)", scales.shape, scales.device)
    # print("(rotations)", rotations.shape, rotations.device)
    # print("(cov3D_precomp)", cov3D_precomp.shape, cov3D_precomp.device)

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # print(screenspace_points.shape, screenspace_points.device)
    # print(means3D.shape, means3D.device)
    # # print(shs.shape, shs.device)
    # print(colors_precomp.shape, colors_precomp.device)
    # print(opacity.shape, opacity.device)
    # print(scales.shape, scales.device)
    # print(rotations.shape, rotations.device)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "scales": scales,
            "rotations": rotations,
            "colors": colors_precomp,
            "opacities": opacity}

def batch_render(viewpoint_camera_list, pc, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, override_color = None, deformation_kwargs = None,
           device_='cuda'
           ):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    from batch_diff_gaussian_rasterization import BatchGaussianRasterizationSettings, BatchGaussianRasterizer

    # B = len(viewpoint_camera_list)
    B = viewpoint_camera_list.world_view_transform.shape[0]
    deformation_kwargs = {} if deformation_kwargs is None else deformation_kwargs
    # means3D_list = []
    # means2D_list = []

    # for i in range(B):
    if not isinstance(pc.get_xyz, torch.Tensor):
            means3D = pc.get_xyz(**deformation_kwargs)
    else:
            means3D = pc.get_xyz

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=means3D.requires_grad, device=device_) + 0
        # screenspace_points = screenspace_points[None, ...].repeat(B, 1, 1)
    if screenspace_points.requires_grad:
            try:
                screenspace_points.retain_grad()
            except:
                pass
        # means3D_list.append(means3D)
        # means2D_list.append(screenspace_points)
    # means3D = torch.stack(means3D_list)
    # means2D = torch.stack(means2D_list)
    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera_list[0].FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera_list[0].FoVy * 0.5)
    print(viewpoint_camera_list.FoVx )
    print(viewpoint_camera_list.FoVy )
    tanfovx = math.tan(viewpoint_camera_list.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera_list.FoVy * 0.5)

    world_view_transform_list = viewpoint_camera_list.world_view_transform# torch.stack([c.world_view_transform for c in viewpoint_camera_list], axis=0)
    full_proj_transform_list = viewpoint_camera_list.full_proj_transform #torch.stack([c.full_proj_transform for c in viewpoint_camera_list], axis=0)
    camera_center_list =viewpoint_camera_list.camera_center #torch.stack([c.camera_center for c in viewpoint_camera_list], axis=0)
    raster_settings = BatchGaussianRasterizationSettings(
        image_height=int(viewpoint_camera_list.image_height),
        image_width=int(viewpoint_camera_list.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform_list,
        projmatrix=full_proj_transform_list,
        sh_degree=pc.active_sh_degree,
        campos=camera_center_list,
        prefiltered=False,
        gaussian_batched=True,
        debug=pipe.debug
    )

    batch_rasterizer = BatchGaussianRasterizer(raster_settings=raster_settings)
    # means3D = pc.get_xyz[None, ...].repeat(B, 1, 1)
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            # # colors_precomp = pc.get_my_features
            #colors_precomp = torch.clamp_min(pc.get_features[:, :3], 0.0)  # TODO: better filter
            # colors_precomp = torch.clamp_min(pc.get_features, 0.0) # TODO: better filter
            colors_precomp = torch.sigmoid(pc.get_features)
            # colors_precomp = colors_precomp[None,...].repeat(B, 1, 1)  # TODO: better filter
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    # print("(means2D)", means2D.shape, means2D.device)
    # print("(means3D)", means3D.shape, means3D.device)
    # # print("(shs)", shs.shape, shs.device)
    # print("(colors_precomp)", colors_precomp.shape, colors_precomp.device)
    # print("(opacity)", opacity.shape, opacity.device)
    # print("(scales)", scales.shape, scales.device)
    # print("(rotations)", rotations.shape, rotations.device)
    # # print("(cov3D_precomp)", cov3D_precomp.shape, cov3D_precomp.device)

    rendered_image, radii = batch_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # print(screenspace_points.shape, screenspace_points.device)
    # print(means3D.shape, means3D.device)
    # # print(shs.shape, shs.device)
    # print(colors_precomp.shape, colors_precomp.device)
    # print(opacity.shape, opacity.device)
    # print(scales.shape, scales.device)
    # print(rotations.shape, rotations.device)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}