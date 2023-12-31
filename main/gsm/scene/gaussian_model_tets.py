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
from torch import nn
from utils.general_utils import inverse_sigmoid, mip_sigmoid
from utils.graphics_utils import load_model

from deformer import deform_gaussians, get_shell_verts_from_base, setup_deformer
from utils import persistence


class TetPoints:
    """Build tets by treating the S shells of meshes as one mesh"""

    def __init__(self, coordinates=None, bari=None, tex_faces=None, faces=None):
        # N = num_shells * P
        # (N, 3) coordinates of points
        self.coordinates = [] if coordinates is None else coordinates
        # (N, 4) barycentric coordinates of tetrahedra
        self.bari = [] if bari is None else bari
        # (F, 4) tet uv vertex indices ranging from [0, num_shell*P_uv] (P_uv is the number of vertices in verts_uv)
        self.tex_faces = [] if tex_faces is None else tex_faces  # (N, 4)
        # (F, 4) tet vertex indices ranging from [0, num_shell*P] (P is the number of vertices in verts)
        self.faces = [] if faces is None else faces


@torch.no_grad()
def generate_points(
    num_shells, faces, vertices, tex_coord, uv_faces, n_points_per_shell
) -> TetPoints:
    num_verts = vertices.shape[1]
    num_uv_verts = tex_coord.shape[0]

    num_shells = vertices.shape[0]
    base_vertices = vertices[num_shells // 2]
    areas, _ = mesh_face_areas_normals(base_vertices, faces)

    vertices = vertices.reshape(-1, 3)

    point_list = TetPoints()
    for i in range(1, num_shells):
        print(f"generating shell points @ shell {i}")
        sample_face_idxs = areas.multinomial(
            n_points_per_shell // 3, replacement=True
        )  # (N, num_samples)

        face_one = faces + (i - 1) * num_verts
        uv_faceone = uv_faces + (i - 1) * num_uv_verts

        face_one = face_one[sample_face_idxs]
        uv_faceone = uv_faceone[sample_face_idxs]
        face_two = face_one + num_verts
        uv_facetwo = uv_faceone + num_uv_verts

        # tetrahedra_tri_faces = torch.concat((face_one[..., :3], face_two[..., :1]), dim=-1)
        # tetrahedra_faces = torch.concat((uv_faceone[..., :3], uv_facetwo[..., :1]), dim=-1)
        tetrahedra_tri_faces = torch.stack(
            [
                torch.concat((face_one[..., :3], face_two[..., :1]), dim=-1),
                torch.concat((face_two[..., :3], face_one[..., 1:2]), dim=-1),
                torch.concat((face_one[..., 1:3], face_two[..., [2, 0]]), dim=-1),
            ],
            dim=0,
        )
        tetrahedra_faces = torch.stack(
            [
                torch.concat((uv_faceone[..., :3], uv_facetwo[..., :1]), dim=-1),
                torch.concat((uv_facetwo[..., :3], uv_faceone[..., 1:2]), dim=-1),
                torch.concat((uv_faceone[..., 1:3], uv_facetwo[..., [2, 0]]), dim=-1),
            ],
            dim=0,
        )
        # 3, F, 4, 3
        tetrahedra = vertices[tetrahedra_tri_faces]

        bari_tets = torch.rand(
            (3, sample_face_idxs.shape[0], 4), device=vertices.device
        )
        bari_tets = bari_tets / bari_tets.sum(dim=-1, keepdim=True)
        points = (tetrahedra * bari_tets[..., None]).sum(dim=-2)
        # from pytorch3d.io import save_ply
        # save_ply(f"dbg_points_{i}.ply", points.reshape(-1, 3))
        point_list.coordinates.append(points.reshape(-1, 3))
        point_list.bari.append(bari_tets.reshape(-1, 4))
        point_list.tex_faces.append(tetrahedra_faces.reshape(-1, 4))
        point_list.faces.append(tetrahedra_tri_faces.reshape(-1, 4))

    point_list.coordinates = torch.concat(point_list.coordinates, dim=0)
    point_list.bari = torch.concat(point_list.bari, dim=0)
    point_list.tex_faces = torch.concat(point_list.tex_faces, dim=0)
    point_list.faces = torch.concat(point_list.faces, dim=0)

    assert point_list.faces.shape == point_list.tex_faces.shape
    assert point_list.faces.shape[-1] == 4
    assert point_list.coordinates.shape[0] == point_list.faces.shape[0]
    # save_ply(f"dbg_points_all.ply", point_list.coordinates.reshape(-1, 3))
    return vertices, point_list


# @torch.no_grad()
# def generate_points(num_shells, faces, vertices, tex_coord, uv_faces,
#                     n_points_per_tri=None,
#                     tex_size=128,
#                     **kwargs)->TetPoints:
#     """Populate TetPoints in shells
#     Args:
#         num_shells: integer number of shells
#         faces: (f, 3) faces of the mesh (shared among all shells)
#         vertices: (..., n, 3) vertices of the mesh
#         tex_coord: (..., m, 2) texture coordinates of the mesh
#         uv_faces: (f, 2) uv faces of the mesh
#     Returns:
#         point_list: TetPoints
#     """
#     num_verts = vertices.shape[1]
#     num_uv_verts = tex_coord.shape[0]

#     if n_points_per_tri is None:
#         area_grid = (1/tex_size) * (1/tex_size)
#         areas = calculate_2D_triangle_area(tex_coord[uv_faces])
#         n_points_per_tri = torch.clamp(torch.round(areas/area_grid/3), min=1.0, max=2).int()

#     total_points_per_shell = n_points_per_tri.sum()

#     vertices = vertices.to(faces.device).reshape(-1, 3)
#     tex_coord = tex_coord.to(faces.device).reshape(-1, 2).repeat(num_shells,1)

#     face_one = faces
#     uv_faceone = uv_faces

#     point_list = TetPoints()
#     for i in range(1, num_shells):
#         print("Generating points for shell {}".format(i))

#         face_two = face_one + num_verts
#         uv_facetwo = uv_faceone + num_uv_verts

#         # tetrahedra faces 3, F, 4
#         tetrahedra_tri_faces = torch.stack(
#             [
#                 torch.concat((face_one[..., :3], face_two[..., :1]), dim=-1),
#                 torch.concat((face_two[..., :3], face_one[..., 1:2]), dim=-1),
#                 torch.concat((face_one[..., 1:3], face_two[..., [2, 0]]), dim=-1),
#             ],
#             dim=0,
#         )
#         tetrahedra_faces = torch.stack(
#             [
#                 torch.concat((uv_faceone[..., :3], uv_facetwo[..., :1]), dim=-1),
#                 torch.concat((uv_facetwo[..., :3], uv_faceone[..., 1:2]), dim=-1),
#                 torch.concat((uv_faceone[..., 1:3], uv_facetwo[..., [2, 0]]), dim=-1),
#             ],
#             dim=0,
#         )
#         # 3, F, 4, 3
#         tetrahedra = vertices[tetrahedra_tri_faces]

#         points, bari_tets = generate_random_point_in_tetrahedron(tetrahedra, n_points_per_tri)
#         point_to_uv_face = torch.repeat_interleave(tetrahedra_faces, n_points_per_tri, dim=-2)
#         point_to_face = torch.repeat_interleave(tetrahedra_tri_faces, n_points_per_tri, dim=-2)

#         point_list.coordinates.append(points.reshape(-1, 3))
#         point_list.bari.append(bari_tets.reshape(-1, 4))
#         point_list.tex_faces.append(point_to_uv_face.reshape(-1, 4))
#         point_list.faces.append(point_to_face.reshape(-1, 4))

#         face_one = face_two
#         uv_faceone = uv_facetwo

#     point_list.coordinates = torch.concat(point_list.coordinates, dim=0)
#     point_list.bari = torch.concat(point_list.bari, dim=0)
#     point_list.tex_faces = torch.concat(point_list.tex_faces, dim=0)
#     point_list.faces = torch.concat(point_list.faces, dim=0)

#     assert point_list.faces.shape == point_list.tex_faces.shape
#     assert point_list.faces.shape[-1] == 4
#     assert point_list.coordinates.shape[0] == point_list.faces.shape[0]
#     assert total_points_per_shell * (num_shells - 1) * 3 == point_list.coordinates.shape[0]
#     return vertices, point_list


@torch.no_grad()
def setup_init_gaussians(
    vertices,
    faces,
    uv_faces,
    tex_coord,
    num_shells=None,
    normal_scale=None,
    n_points_per_shell=None,
    tex_size=128,
    shrunk_ref_mesh=None,
):
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
    if vertices.ndim == 2:
        print("SETUP_INIT_GAUSSIANS: Create shells from base mesh.")
        assert normal_scale is not None and num_shells is not None
        vertices = get_shell_verts_from_base(
            vertices[None],
            faces,
            normal_scale,
            num_shells,
            shrunk_ref_mesh=shrunk_ref_mesh,
        )[0]
    else:
        num_shells = vertices.shape[0]
    new_vertices, points_list = generate_points(
        num_shells,
        faces,
        vertices,
        tex_coord,
        uv_faces,
        n_points_per_shell=n_points_per_shell,
    )
    print(
        "Generated {} points on {} shells".format(
            len(points_list.coordinates), num_shells
        )
    )
    return new_vertices, points_list


def logistic(x, L, k, x0):
    return L / (1 + torch.exp(-k * (x - x0)))


@persistence.persistent_class
class TetGaussianModel(torch.nn.Module):
    def __init__(
        self,
        base_shell_path,
        sh_degree: int = 0,
        num_shells=5,
        offset_len=0.1,
        res=128,
        smpl_init_kwargs=None,
        smpl_can_params=None,
        smpl_scale=torch.ones((3,)),
        smpl_transl=torch.zeros((3,)),
        total_points=100000,
        max_scaling=0.01,
        scale_act="exp",
        rotate_gaussians=False,
        shrunk_ref_mesh=None,
        use_base_shell_for_correspondence=False,
        device="cuda:0",
        *unused_args,
        **unused_kwargs,
    ):
        print("TetGaussianModel init")
        super().__init__()

        shell_base_verts, shell_faces, faces_uvs, vertex_uvs = load_model(
            base_shell_path, device
        )

        # scale
        if isinstance(smpl_scale, torch.Tensor):
            smpl_scale = smpl_scale.to(device)
        else:
            smpl_scale = torch.tensor(smpl_scale, dtype=torch.float32, device=device)
        smpl_transl = smpl_transl.to(device)

        # Create deformer, will set self.shell_verts and some other attributes
        deformer = setup_deformer(
            self,
            shell_base_verts=shell_base_verts,
            shell_faces=shell_faces,
            num_shells=num_shells,
            offset_len=offset_len,
            smpl_init_kwargs=smpl_init_kwargs,
            smpl_can_params=smpl_can_params,
            smpl_scale=smpl_scale,
            smpl_transl=smpl_transl,
            shrunk_ref_mesh=shrunk_ref_mesh,
            use_base_shell_for_correspondence=use_base_shell_for_correspondence,
            device=device,
        )
        self.deformer = deformer
        self.register_buffer("smpl_scale", smpl_scale)
        self.register_buffer("transl", smpl_transl)
        self.register_buffer("shell_verts", self._shell_verts)
        self.register_buffer("smpl_shell_verts", self._smpl_shell_verts)
        self.register_buffer("point_verts_weights", self._point_verts_weights)
        self.register_buffer("point_verts_idxs", self._point_verts_idxs)

        assert self.shell_verts.shape[0] == num_shells
        assert shell_faces.ndim == 2 and shell_faces.shape[1] == 3
        assert (
            faces_uvs.ndim == 2
            and faces_uvs.shape[1] == 3
            and faces_uvs.shape[0] == shell_faces.shape[0]
        )
        assert vertex_uvs.ndim == 2 and vertex_uvs.shape[1] == 2

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._rotate_gaussians = rotate_gaussians

        with torch.no_grad():
            _, points = setup_init_gaussians(
                self.shell_verts,
                shell_faces,
                faces_uvs,
                vertex_uvs,
                n_points_per_shell=total_points // (num_shells - 1),
            )

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
            self.num_shells = num_shells

            self.register_buffer(
                "tex_faces", points.tex_faces.unsqueeze(-1)
            )  # (P, 4, 1)
            self.register_buffer(
                "stacked_tex",
                vertex_uvs.unsqueeze(0).repeat(num_shells, 1, 1).contiguous(),
            )  # (S, P_uv, 2)
            self.register_buffer("bari", points.bari.reshape(-1, 4, 1).contiguous())
            self.register_buffer("faces", points.faces.contiguous())  # (P, 4)
            self.register_buffer("_xyz", points.coordinates)

            self.max_scaling = max_scaling  # 0.04 #0.03

            assert self.bari.shape[0] == self._xyz.shape[0]
            assert (
                self.stacked_tex.shape[0] == num_shells
                and self.stacked_tex.shape[2] == 2
            )
            self.interp_dim = self.bari.shape[1]
            assert self.interp_dim == 4

    @torch.no_grad()
    def perturb_points(self):
        """Perturb the points in the model"""
        # recompute points coordinates, point_uv with resampled baricentric coordinates
        self.bari.data[:] = torch.rand_like(self.bari)
        self.bari.data = self.bari.data / self.bari.data.sum(dim=-2, keepdim=True)
        shell_verts = self.shell_verts.reshape(-1, 3)
        self._xyz.data[:] = (shell_verts[self.faces] * self.bari).sum(dim=-2)

    @property
    def get_scaling(self):
        if self.mlp_sca is not None:
            sampled_points = self.lookup_texture_map(
                self._scaling.flip(1).to(self._texture_planes.device),
                self.stacked_tex.reshape(self.num_shells, -1, 1, 2).to(
                    self._texture_planes.device
                ),
            )
            sampled_points = sampled_points.reshape(1, -1, self.num_shells * 64)
            sampled_points = self.mlp_sca(sampled_points).reshape(-1, 3)
            features_ = sampled_points[
                self.tex_faces.to(self._texture_planes.device)
            ].reshape(-1, 4, 3)
            up_ = (features_ * self.bari.to(self._texture_planes.device)).sum(dim=1)
            # return torch.clamp(torch.nn.ReLU()(up_.reshape(-1,3)), max = self.max_th)
            # return logistic(up_.reshape(-1,3), L=self.max_th, k=0.125, x0=0)
            return self.scaling_activation(up_.reshape(-1, 3))  # remove the clamp

        else:
            sampled_points = self.lookup_texture_map(
                self._scaling.flip(1).to(self._texture_planes.device),
                self.stacked_tex.reshape(self.num_shells, -1, 1, 2).to(
                    self._texture_planes.device
                ),
            )
            sampled_points = sampled_points.reshape(-1, 3)
            features_ = sampled_points[
                self.tex_faces.to(self._texture_planes.device)
            ].reshape(-1, 4, 3)
            up_ = (features_ * self.bari.to(self._texture_planes.device)).sum(dim=1)
            return torch.clamp(
                self.scaling_activation(up_.reshape(-1, 3)), max=self.max_scaling
            )

    def get_rotation(
        self, use_cached=False, cache_deformation_output=False, **smpl_kwargs
    ):
        if self.mlp_rot is not None:
            sampled_points = self.lookup_texture_map(
                self._rotation.flip(1).to(self._texture_planes.device),
                self.stacked_tex.reshape(self.num_shells, -1, 1, 2).to(
                    self._texture_planes.device
                ),
            )
            sampled_points = sampled_points.reshape(1, -1, self.num_shells * 64)
            sampled_points = self.mlp_rot(sampled_points).reshape(-1, 4)
            features_ = sampled_points[
                self.tex_faces.to(self._texture_planes.device)
            ].reshape(-1, 4, 4)
            up_ = (features_ * self.bari.to(self._texture_planes.device)).sum(dim=1)
            canonical_quat = self.rotation_activation(up_.reshape(-1, 4))
        else:
            sampled_points = self.lookup_texture_map(
                self._rotation.flip(1).to(self._texture_planes.device),
                self.stacked_tex.reshape(self.num_shells, -1, 1, 2).to(
                    self._texture_planes.device
                ),
            )
            sampled_points = sampled_points.reshape(-1, 4)
            features_ = sampled_points[
                self.tex_faces.to(self._texture_planes.device)
            ].reshape(-1, 4, 4)
            up_ = (features_ * self.bari.to(self._texture_planes.device)).sum(dim=1)
            canonical_quat = self.rotation_activation(up_.reshape(-1, 4))

        if not self._rotate_gaussians:
            return canonical_quat
        if (
            use_cached
            and hasattr(self, "_cached_deformation_output")
            and self._cached_deformation_output is not None
        ):
            deformation_output = self._cached_deformation_output
        else:
            deformation_output = deform_gaussians(self, self.deformer, smpl_kwargs)
        if cache_deformation_output:
            self._cached_deformation_output = deformation_output
        weighted_transform_mat = deformation_output.weighted_transform_mat
        batch_size = weighted_transform_mat.shape[0]
        canonical_quat = canonical_quat[None].expand(batch_size, -1, -1)
        rot_mat = weighted_transform_mat[..., :3, :3]
        quat = matrix_to_quaternion(rot_mat)
        result_quaternion = quaternion_multiply(canonical_quat, quat)
        return self.rotation_activation(result_quaternion.reshape(-1, 4)).reshape(
            batch_size, -1, 4
        )

    def get_xyz(self, cache_deformation_output=False, use_cached=False, **smpl_kwargs):
        if (
            use_cached
            and hasattr(self, "_cached_deformation_output")
            and self._cached_deformation_output is not None
        ):
            deformation_output = self._cached_deformation_output
        else:
            deformation_output = deform_gaussians(self, self.deformer, smpl_kwargs)

        if cache_deformation_output:
            self._cached_deformation_output = deformation_output
        return deformation_output.deformed_pnts + self.transl

    @property
    def get_features(self):
        if self.mlp_dc is not None:
            stacked_features = (
                self._features_dc
            )  # torch.cat((self._features_dc, self._features_rest), dim=-1)
            num_features = 3
            sampled_points = self.lookup_texture_map(
                stacked_features.flip(1).contiguous().to(self._texture_planes.device),
                self.stacked_tex.reshape(self.num_shells, -1, 1, 2).to(
                    self._texture_planes.device
                ),
            )
            sampled_points = sampled_points.reshape(1, -1, self.num_shells * 64)
            sampled_points = self.mlp_dc(sampled_points).reshape(-1, num_features)

            features_ = sampled_points[
                self.tex_faces.to(self._texture_planes.device)
            ].reshape(-1, 4, num_features)

            up_ = (features_ * self.bari.to(self._texture_planes.device)).sum(dim=1)
            features_total = up_.reshape(-1, num_features)
            features_total = self.rgb_activation(features_total)  # follow mip-nerf
            return features_total
        else:
            stacked_features = (
                self._features_dc
            )  # torch.cat((self._features_dc, self._features_rest), dim=-1)
            num_features = stacked_features.shape[-1]
            sampled_points = self.lookup_texture_map(
                stacked_features.flip(1).contiguous().to(self._texture_planes.device),
                self.stacked_tex.reshape(self.num_shells, -1, 1, 2).to(
                    self._texture_planes.device
                ),
            )

            sampled_points = sampled_points.reshape(-1, num_features)

            features_ = sampled_points[
                self.tex_faces.to(self._texture_planes.device)
            ].reshape(-1, 4, num_features)

            up_ = (features_ * self.bari.to(self._texture_planes.device)).sum(dim=1)
            features_total = up_.reshape(-1, num_features)
            features_total = self.rgb_activation(features_total)

            return features_total

    @property
    def get_opacity(self):
        if self.mlp_opa is not None:
            sampled_points = self.lookup_texture_map(
                self._opacity.flip(1).to(self._texture_planes.device),
                self.stacked_tex.reshape(self.num_shells, -1, 1, 2).to(
                    self._texture_planes.device
                ),
            )
            sampled_points = sampled_points.reshape(1, -1, self.num_shells * 64)
            sampled_points = self.mlp_dc(sampled_points).reshape(-1, 1)
            features_ = sampled_points[
                self.tex_faces.to(self._texture_planes.device)
            ].reshape(-1, 4, 1)

            up_ = (features_ * self.bari.to(self._texture_planes.device)).sum(dim=1)
            return self.opacity_activation(up_.reshape(-1, 1))
        else:
            sampled_points = self.lookup_texture_map(
                self._opacity.flip(1).to(self._texture_planes.device),
                self.stacked_tex.reshape(self.num_shells, -1, 1, 2).to(
                    self._texture_planes.device
                ),
            )
            sampled_points = sampled_points.reshape(-1, 1)
            features_ = sampled_points[
                self.tex_faces.to(self._texture_planes.device)
            ].reshape(-1, 4, 1)

            up_ = (features_ * self.bari.to(self._texture_planes.device)).sum(dim=1)

            return self.opacity_activation(up_.reshape(-1, 1))

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

    def lookup_texture_map(self, texture, texc):
        """Use sampling to get texture value

        Args:
            texture: [B, H, W, 4] texture color
            texc: [B, H, W, 2] texture uv
        Returns:
            [B, H, W, 4] texture color
        """
        # map to [-1, 1] and change to [B, 2, W, H]
        out = F.grid_sample(
            texture.permute(0, 3, 1, 2),
            2 * texc - 1,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        # map back to [B, H, W, 4]
        return out.permute(0, 2, 3, 1)
