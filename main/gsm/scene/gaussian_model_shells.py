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
import torch.nn.functional as F
from pytorch3d.transforms import (matrix_to_quaternion, quaternion_multiply)
from pytorch3d.ops import mesh_face_areas_normals
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, mip_sigmoid
from utils.graphics_utils import (calculate_2D_triangle_area,
                                  generate_random_point_in_triangle,
                                  load_model)
from scene.gaussian_model_tets import TetGaussianModel
from deformer import (deform_gaussians, get_shell_verts_from_base,
                      setup_deformer)
from utils import persistence
import tqdm


class Points:
    def __init__(self, coordinates=None, tri_bari=None, tex_faces=None, faces=None, point_uv=None):
        self.coordinates = [] if coordinates is None else coordinates
        self.tri_bari = [] if tri_bari is None else tri_bari
        self.tex_faces = [] if tex_faces is None else tex_faces
        self.faces = [] if faces is None else faces
        self.point_uv = [] if point_uv is None else point_uv


@torch.no_grad()
def generate_points(num_shells, faces, vertices, tex_coord, uv_faces,
                    n_points_per_shell=4000)->Points:
    num_verts = vertices.shape[1]
    num_uv_verts = tex_coord.shape[0]

    num_shells = vertices.shape[0]
    base_vertices = vertices[num_shells//2]
    areas, _ = mesh_face_areas_normals(base_vertices, faces)

    point_list =  Points()
    g1 = torch.Generator(device=vertices.device)
    g1.manual_seed(0)
    for i in range(num_shells):
        print(f"generating shell points @ shell {i}")
        sample_face_idxs = areas.multinomial(
                n_points_per_shell, replacement=True, generator=g1
            )  # (N, num_samples)

        uvw = torch.rand((sample_face_idxs.shape[0], 3), device=vertices.device)
        uvw = uvw / uvw.sum(dim=-1, keepdim=True)
        points = torch.sum(vertices[i][faces[sample_face_idxs]] * uvw[..., None], dim=-2)
        point_list.coordinates.append(points)
        point_list.tri_bari.append(uvw)
        point_list.faces.append(faces[sample_face_idxs] + num_verts*i)
        point_to_uv_face = uv_faces[sample_face_idxs]
        point_list.tex_faces.append(uv_faces[sample_face_idxs] + num_uv_verts * i)
        point_list.point_uv.append(torch.sum(tex_coord[point_to_uv_face] * uvw[..., None], dim=-2))
        # from pytorch3d.io import save_ply
        # save_ply(f"dbg_shell_{i}.ply", vertices[i], faces)
        # save_ply(f"dbg_points_{i}.ply", points)

    point_list.tri_bari = torch.concat(point_list.tri_bari, dim=0)
    point_list.faces = torch.concat(point_list.faces, dim=0)
    point_list.coordinates = torch.concat(point_list.coordinates, dim=0)
    point_list.tex_faces = torch.concat(point_list.tex_faces, dim=0)
    point_list.point_uv = torch.concat(point_list.point_uv, dim=0)

    assert point_list.tri_bari.shape[-1] == 3
    assert n_points_per_shell * num_shells == point_list.coordinates.shape[0]
    print("Generated {} points".format(point_list.coordinates.shape[0]))
    return vertices, point_list


# @torch.no_grad()
# def generate_points(num_shells, faces, vertices, tex_coord, uv_faces,
#                     n_points_per_tri=None, tex_size=128,
#                     **kwargs)->Points:
#     """Populate Points in shells
#     Args:
#         num_shells: integer number of shells
#         faces: (f, 3) faces of the mesh (shared among all shells)
#         vertices: (shells, n, 3) vertices of the shell meshes
#         tex_coord: (m, 2) texture coordinates of the mesh
#         uv_faces: (f, 2) uv faces of the mesh
#     Returns:
#         point_list: Points
#     """
#     num_verts = vertices.shape[1]
#     num_uv_verts = tex_coord.shape[0]

#     if n_points_per_tri is None:
#         area_grid = (1/tex_size) * (1/tex_size)
#         areas = calculate_2D_triangle_area(tex_coord[uv_faces])
#         n_points_per_tri = torch.clamp(areas/area_grid, min=0.25, max=2).ceil().int()

#     total_points_per_shell = n_points_per_tri.sum()

#     point_to_uv_face = torch.repeat_interleave(uv_faces, n_points_per_tri, dim=0)
#     point_to_face = torch.repeat_interleave(faces, n_points_per_tri, dim=0)
#     point_list = Points()
#     for i in range(num_shells):
#         print(f"generating shell points @ shell {i}")
#         # P, 3
#         points, uvws = generate_random_point_in_triangle(vertices[i][faces], n_points_per_triangle=n_points_per_tri)
#         point_list.coordinates.append(points)
#         point_list.tri_bari.append(uvws)
#         point_list.faces.append(point_to_face + num_verts*i)
#         point_list.tex_faces.append(point_to_uv_face + num_uv_verts * i)
#         # (P, 3, 2) * (P, 3, 1)
#         point_list.point_uv.append(torch.sum(tex_coord[point_to_uv_face] * uvws[..., None], dim=1))

#     point_list.tri_bari = torch.concat(point_list.tri_bari, dim=0)
#     point_list.faces = torch.concat(point_list.faces, dim=0)
#     point_list.coordinates = torch.concat(point_list.coordinates, dim=0)
#     point_list.tex_faces = torch.concat(point_list.tex_faces, dim=0)
#     point_list.point_uv = torch.concat(point_list.point_uv, dim=0)

#     assert point_list.tri_bari.shape[-1] == 3
#     assert total_points_per_shell * num_shells == point_list.coordinates.shape[0]
#     print("Generated {} points".format(point_list.coordinates.shape[0]))
#     return vertices, point_list


@torch.no_grad()
def setup_init_gaussians(vertices, faces, uv_faces, tex_coord,
                        num_shells=None, normal_scale=None,
                        n_points_per_shell=None,
                        shrunk_ref_mesh=None):
    """
    Args:
        vertices: (n, 3) vertices of the base mesh or (shell, n, 3)
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
        vertices = get_shell_verts_from_base(vertices[None], faces, normal_scale, num_shells, shrunk_ref_mesh=shrunk_ref_mesh)[0]
    else:
        num_shells = vertices.shape[0]
    new_vertices, points_list = generate_points(num_shells, faces, vertices, tex_coord, uv_faces,
                                                n_points_per_shell=n_points_per_shell)

    return new_vertices, points_list


@persistence.persistent_class
class ShellGaussianModel(TetGaussianModel):
    def __init__(self,
                 base_shell_path,
                 sh_degree: int=0,
                 num_shells=5,
                 offset_len=0.1,
                 res=128,
                 smpl_init_kwargs=None,
                 smpl_can_params=None,
                 smpl_scale=torch.ones((3,)),
                 smpl_transl=torch.zeros((3,)),
                 total_points=100000,
                 max_scaling=0.01,
                 scale_act='exp',
                 rotate_gaussians=False,
                 shrunk_ref_mesh=None,
                 use_base_shell_for_correspondence=False,
                 device="cuda:0",
                 *unused_args,
                 **unused_kwargs):
        print("ShellGaussianModel init")
        torch.nn.Module.__init__(self)

        shell_base_verts, shell_faces, faces_uvs, vertex_uvs  = load_model(base_shell_path, device)

        # scale
        if isinstance(smpl_scale, torch.Tensor):
            smpl_scale = smpl_scale.to(device)
        else:
            smpl_scale = torch.tensor(smpl_scale, dtype=torch.float32, device=device)

        if isinstance(smpl_transl, torch.Tensor):
            smpl_transl = smpl_transl.to(device)
        else:
            smpl_transl = torch.tensor(smpl_transl, dtype=torch.float32, device=device)

        # Create deformer, will set self.shell_verts
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
        # self.register_buffer("point_verts_weights", self._point_verts_weights)
        # self.register_buffer("point_verts_idxs", self._point_verts_idxs)
        self.point_verts_weights = self._point_verts_weights
        self.point_verts_idxs = self._point_verts_idxs
        self.total_points = total_points
        self.shell_faces = shell_faces

        assert self.shell_verts.shape[0] == num_shells
        assert shell_faces.ndim == 2 and shell_faces.shape[1] == 3
        assert faces_uvs.ndim == 2 and faces_uvs.shape[1] == 3 and faces_uvs.shape[0] == shell_faces.shape[0]
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
                n_points_per_shell=total_points//num_shells,
            )

            if scale_act == 'exp':
                self.scaling_activation = torch.exp
                self.scaling_inverse_activation = torch.log
            elif scale_act == 'softplus':
                self.scaling_activation = torch.nn.Softplus()
                self.scaling_inverse_activation = inverse_sigmoid
            elif scale_act == 'sigmoid':
                self.scaling_activation = torch.sigmoid
                self.scaling_inverse_activation = inverse_sigmoid

            self.rgb_activation = mip_sigmoid

            self.opacity_activation = torch.sigmoid
            self.inverse_opacity_activation = inverse_sigmoid

            self.rotation_activation = torch.nn.functional.normalize
            self._num_shells = num_shells

            self.register_buffer("tex_faces", points.tex_faces.contiguous())  # (P, 3) indices to uv coordinates
            self.register_buffer("stacked_tex", vertex_uvs.unsqueeze(0).repeat(num_shells, 1, 1).contiguous())  # verts_uv (S, P_uv, 2)
            self.register_buffer("point_uv", points.point_uv.reshape(-1, 2, 1).contiguous()), # (P, 2, 1)
            self.register_buffer("bari", points.tri_bari.reshape(-1,3,1).contiguous())  # (P, 3, 1)
            self.register_buffer("faces", points.faces.contiguous())  # (P, 3)
            self.register_buffer("_xyz", points.coordinates)  # (P, 3)

            self.max_scaling = max_scaling #0.04 #0.03

            assert self.bari.shape[0] == self._xyz.shape[0]
            assert self.stacked_tex.shape[0] == num_shells and self.stacked_tex.shape[2] == 2
            self.interp_dim = self.bari.shape[1]
            assert self.interp_dim == 3

    def get_face_index(self, replace_faces=None):
        n_points_per_shell=self.total_points//self._num_shells
        index_list = []
        replace_idx = []
        num_verts = self.shell_verts.shape[1]
        for i in tqdm.tqdm(range(self._num_shells)):
            per_shell_idx = []
            for j in tqdm.tqdm(range(n_points_per_shell)):
                indices = torch.nonzero(torch.all(self.shell_faces == (self.faces[i*n_points_per_shell+j]-num_verts*i), dim=1), as_tuple=False)
                per_shell_idx.append(indices[0])
            per_shell_idx = torch.cat((per_shell_idx))
            if replace_faces is not None:
                ind_faces = torch.zeros(10728, device=self.shell_verts.device)
                ind_faces[replace_faces] = 1
                ind_ones = torch.gather(ind_faces, dim=0, index=per_shell_idx)
                replace_idx.append(ind_ones)
        
        replace_idx = torch.concat(replace_idx, dim=0)
        # np.savetxt('replace_idx.txt',replace_idx.cpu().numpy())
        self._replace_idx = replace_idx
    
    @property
    def replace_idx(self):
        return self._replace_idx

    @property
    def num_shells(self):
        return self._num_shells

    @torch.no_grad()
    def perturb_points(self):
        """Perturb the points in the model"""
        # recompute points coordinates, point_uv with resampled baricentric coordinates
        self.bari.data = torch.rand_like(self.bari)
        assert self.bari.shape[-2] == 3
        self.bari.data[:] = self.bari.data / self.bari.data.sum(dim=-2, keepdim=True)
        shell_verts = self.shell_verts.reshape(-1, 3)
        self._xyz.data[:] = (shell_verts[self.faces] * self.bari).sum(dim=-2)
        stacked_tex = self.stacked_tex.reshape(-1, 2)
        self.point_uv.data[:] = (stacked_tex[self.tex_faces] * self.bari).sum(dim=-2).unsqueeze(-1)

    @property
    def get_scaling(self):
        sampled_points = self.lookup_texture_map(self._scaling.flip(1),
                                                 self.point_uv.reshape(self._num_shells,-1,1,2).to(self._scaling.device),
                                                )
        if self.mlp_sca is not None:
            sampled_points = sampled_points.reshape(1, -1, self._num_shells*64)
            sampled_points = self.mlp_sca(sampled_points).reshape(-1, 3)
            return self.scaling_activation(sampled_points) # remove the clamp
        else:
            return torch.clamp(self.scaling_activation(sampled_points.reshape(-1,3)), max=self.max_scaling)


    def get_rotation(self, cache_deformation_output=False, use_cached=False, **smpl_kwargs):
        sampled_points = self.lookup_texture_map(self._rotation.flip(1),
                                                 self.point_uv.reshape(self._num_shells,-1,1,2).to(self._rotation.device),
                                                 )
        if self.mlp_rot is not None:
            sampled_points = sampled_points.reshape(1, -1, self._num_shells*64)
            sampled_points = self.mlp_rot(sampled_points).reshape(-1, 4)
        canonical_quat = self.rotation_activation(sampled_points.reshape(-1,4))
        if not self._rotate_gaussians:
            return canonical_quat
        if use_cached and hasattr(self, "_cached_deformation_output") and self._cached_deformation_output is not None:
            deformation_output = self._cached_deformation_output
        else:
            deformation_output = deform_gaussians(
                self, self.deformer, smpl_kwargs
            )
        if cache_deformation_output:
            self._cached_deformation_output = deformation_output
        weighted_transform_mat = deformation_output.weighted_transform_mat
        batch_size = weighted_transform_mat.shape[0]
        canonical_quat = canonical_quat[None].expand(batch_size, -1, -1)
        rot_mat = weighted_transform_mat[..., :3, :3]
        quat = matrix_to_quaternion(rot_mat)
        result_quaternion = quaternion_multiply(canonical_quat, quat)
        return self.rotation_activation(result_quaternion.reshape(-1, 4)).reshape(batch_size, -1, 4)

    @property
    def get_features(self):
        stacked_features = self._features_dc #torch.cat((self._features_dc, self._features_rest), dim=-1)
        num_features = stacked_features.shape[-1]
        sampled_points = self.lookup_texture_map(stacked_features.flip(1).contiguous().to(self._features_dc.device),
                                                 self.point_uv.reshape(self._num_shells,-1,1,2).to(self._features_dc.device),
                                                 )
        if self.mlp_dc is not None:
            num_features = 3
            sampled_points = sampled_points.reshape(1, -1, self._num_shells*64)
            sampled_points = self.mlp_dc(sampled_points).reshape(-1, num_features)

        sampled_points = torch.sigmoid(sampled_points.reshape(-1, num_features))*(1 + 2*0.001) - 0.001 # follow mip-nerf
        return sampled_points

    @property
    def get_opacity(self):
        sampled_points = self.lookup_texture_map(self._opacity.flip(1),
                                                 self.point_uv.reshape(self._num_shells,-1,1,2).to(self._opacity.device),
                                                 )
        if self.mlp_opa is not None:
            sampled_points = sampled_points.reshape(1, -1, self._num_shells*64)
            sampled_points = self.mlp_opa(sampled_points).reshape(-1, 1)
        return self.opacity_activation(sampled_points.reshape(-1, 1))