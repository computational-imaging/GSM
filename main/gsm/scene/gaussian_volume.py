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

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from pytorch3d.ops import mesh_face_areas_normals
from utils.general_utils import inverse_sigmoid, mip_sigmoid
from utils.graphics_utils import (load_model)
from deformer import (deform_gaussians, get_shell_verts_from_base,
                      setup_deformer)
from utils import persistence


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Args:
        planes: axes (n_planes, 3, 3)
        coordinates: (N, M, 3)
    Returns:
        projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


@torch.no_grad()
def generate_points(n_points, box):
    """We sample points randomly in a bounding box
    Args:
        n_points: int, number of points to sample
        box: (2, 3) bounding box for min xyz and max xyz
    """
    min_xyz, max_xyz = box
    vertices = torch.rand(n_points, 3, device=min_xyz.device) * (max_xyz - min_xyz) + min_xyz
    return vertices


@torch.no_grad()
def setup_init_gaussians(n_points, box):
    """
    Args:
        vertices: (n, 3) vertices of the base mesh
        faces: (f, 3) faces of the base mesh
        uv_faces: (f, 2) uv faces of the base mesh
        tex_coord: (m, 2) texture coordinates of the base mesh
        num_shells: number of shells
        normal_scale: offset scale
    Returns:
        new_vertices (num_shell*n, 3), points_list (TetPoints)
    """
    new_vertices = generate_points(n_points, box)
    # print("Generated {} points in box {}".format(n_points, box[0].tolist() + box[1].tolist()))
    return new_vertices


def logistic(x, L, k, x0):
    return L / (1 + torch.exp(-k * (x - x0)))


@persistence.persistent_class
class VolumeGaussianModel(torch.nn.Module):
    def __init__(
        self,
        sh_degree: int,
        bbox,
        total_points=100000,
        max_scaling=0.02,
        scale_act="exp",
        device="cuda:0",
        *unsed_args,
        **unsed_kwargs,
    ):
        print("VolumeGaussian init")
        super().__init__()

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        with torch.no_grad():
            self.plane_axes = generate_planes()
            points = setup_init_gaussians(total_points, box=bbox)

            if scale_act == "exp":
                self.scaling_activation = torch.exp
                self.scaling_inverse_activation = torch.log
            elif scale_act == "softplus":
                self.scaling_activation = torch.nn.Softplus()
                self.scaling_inverse_activation = inverse_sigmoid
            elif scale_act == "sigmoid":
                self.scaling_activation = torch.sigmoid
                self.scaling_inverse_activation = inverse_sigmoid

            self.rgb_activation = mip_sigmoid

            self.opacity_activation = torch.sigmoid
            self.inverse_opacity_activation = inverse_sigmoid

            self.rotation_activation = torch.nn.functional.normalize

            self.register_buffer("_bbox", bbox.to(device))
            self.register_buffer("_xyz", points.reshape(-1, 3).contiguous().to(device))

            self.max_scaling = max_scaling

    @torch.no_grad()
    def perturb_points(self):
        """Perturb the points in the model"""
        # recompute points coordinates, point_uv with resampled baricentric coordinates
        self._xyz.data[:] = setup_init_gaussians(self._xyz.shape[0], box=self._bbox)

    @property
    def get_scaling(self):
        if self.mlp_sca is not None:
            sampled_points = self.lookup_texture_map(self._scaling.to(self._texture_planes.device))
            sampled_points = sampled_points.reshape(1, self._xyz.shape[0], 64)
            sampled_points = self.mlp_sca(sampled_points).reshape(-1, 3)
            return self.scaling_activation(sampled_points)  # remove the clamp
        else:
            sampled_points = self.lookup_texture_map(self._scaling.to(self._texture_planes.device))
            sampled_points = sampled_points.reshape(-1, 3)
            return torch.clamp(
                self.scaling_activation(sampled_points), max=self.max_scaling
            )

    @property
    def get_rotation(self):
        if self.mlp_rot is not None:
            sampled_points = self.lookup_texture_map(
                self._rotation.to(self._texture_planes.device),
            )
            sampled_points = sampled_points.reshape(1, self.get_xyz.shape[0], 64)
            sampled_points = self.mlp_rot(sampled_points).reshape(-1, 4)
            canonical_quat = self.rotation_activation(sampled_points.reshape(-1, 4))
        else:
            sampled_points = self.lookup_texture_map(
                self._rotation.to(self._texture_planes.device),
            )
            canonical_quat = self.rotation_activation(sampled_points.reshape(-1, 4))
        return canonical_quat

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        if self.mlp_dc is not None:
            stacked_features = (
                self._features_dc
            )  # torch.cat((self._features_dc, self._features_rest), dim=-1)
            num_features = 3
            sampled_points = self.lookup_texture_map(
                stacked_features.contiguous().to(self._texture_planes.device),
                )
            sampled_points = sampled_points.reshape(1, self.get_xyz.shape[0], 64)
            sampled_points = self.mlp_dc(sampled_points).reshape(-1, num_features)
            return self.rgb_activation(sampled_points)  # follow mip-nerf
        else:
            stacked_features = (
                self._features_dc
            )  # torch.cat((self._features_dc, self._features_rest), dim=-1)
            num_features = stacked_features.shape[-1]
            sampled_points = self.lookup_texture_map(
                stacked_features.contiguous().to(self._texture_planes.device),
            )

            sampled_points = sampled_points.reshape(-1, num_features)
            return self.rgb_activation(sampled_points)

    @property
    def get_opacity(self):
        if self.mlp_opa is not None:
            sampled_points = self.lookup_texture_map(
                self._opacity.to(self._texture_planes.device),
            )
            sampled_points = sampled_points.reshape(1, self.get_xyz.shape[0], 64)
            sampled_points = self.mlp_dc(sampled_points).reshape(-1, 1)
            return self.opacity_activation(sampled_points)
        else:
            sampled_points = self.lookup_texture_map(
                self._opacity.to(self._texture_planes.device),
                )
            sampled_points = sampled_points.reshape(-1, 1)
            return self.opacity_activation(sampled_points)

    def get_opacity_planes(
        self,
    ):
        return self.opacity_activation(self._opacity)

    def get_rgb_planes(
        self,
    ):
        return self.rgb_activation(self._features_dc)

    def get_rotation_planes(
        self,
    ):
        shp = self._rotation.shape
        return self.rotation_activation(self._rotation.reshape(-1, 4)).reshape(shp)

    def get_scaling_planes(
        self,
    ):
        return self.scaling_activation(self._scaling)

    def set_opacity(self, opacity, mlp_opa=None):
        self._opacity = opacity
        self.mlp_opa = mlp_opa

    def set_features_dc(self, features_dc, mlp_dc=None):
        self._features_dc = features_dc
        self.mlp_dc = mlp_dc

    def set_features_rest(self, features_rest):
        self._features_rest = features_rest

    def set_scaling(self, scaling, mlp_sca=None):
        self._scaling = scaling
        self.mlp_sca = mlp_sca

    def set_rotation(self, rotation, mlp_rot=None):
        self._rotation = rotation
        self.mlp_rot = mlp_rot

    def set_texture(self, texture):
        self._texture_planes = texture

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def lookup_texture_map(self, texture):
        """Use sampling to get texture value

        Args:
            texture: [B, H, W, C] texture color
        Returns:
            [B, P, C] texture color
        """
        # triplanes
        texture_planes = texture.permute(0, 3, 1, 2).reshape(1, 3, -1, texture.shape[1], texture.shape[2])
        N, n_planes, C, H, W = texture_planes.shape
        M, _ = self._xyz.shape
        texture_planes = texture_planes.view(N*n_planes, C, H, W)
        coordinates = (self._xyz - self._bbox[0]) / (self._bbox[1] - self._bbox[0])  # range (0, 1)
        coordinates = (coordinates - 0.5) * 2  # range (-1, 1)

        projected_coordinates = project_onto_planes(self.plane_axes.to(coordinates.device), coordinates[None]).unsqueeze(1)
        output_features = torch.nn.functional.grid_sample(texture_planes,
                                                          projected_coordinates.float(),
                                                          mode="bilinear",
                                                          padding_mode="zeros",
                                                          align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        # accumulate triplane textures
        output_features = output_features.mean(dim=1)
        return output_features
