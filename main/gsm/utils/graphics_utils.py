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

import math
import os
from typing import NamedTuple, Union

import numpy as np
import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    PointsRasterizationSettings,
    rasterize_meshes,
    rasterize_points,
)
from utils.system_utils import mkdir_p


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points_batched(points, transf_matrix):
    """transf_matrix is Bx4x4 matrix"""
    # convert to homogeneous coordinates
    B, P, _ = points.shape
    ones = torch.ones((B, P, 1), dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=-1)
    points_out = torch.matmul(points_hom, transf_matrix)

    denom = points_out[..., 3:] + 0.0000001
    return points_out[..., :3] / denom


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getWorld2ViewTensor(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    Rt = torch.zeros((4, 4), dtype=torch.float32)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt


def getWorld2ViewBatched(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    """Batched version of getWorld2View2"""
    B = R.size(0)  # Get the batch size

    # Create an identity matrix with batch dimension
    Rt = (
        torch.zeros((4, 4), dtype=torch.float32, device=R.device)
        .unsqueeze(0)
        .repeat(B, 1, 1)
    )

    # Transpose R along the batch dimension and assign it to Rt
    Rt[:, :3, :3] = R.permute(0, 2, 1)

    # Assign t to the first three elements of the last column of Rt
    t_expanded = t.unsqueeze(-1)
    Rt[:, :3, 3:4] = t_expanded

    # Set the bottom-right element of Rt to 1.0
    Rt[:, 3, 3] = 1.0

    # Compute the inverse of Rt for each batch element
    C2W = torch.inverse(Rt)

    # Extract the camera centers for each batch element
    cam_center = C2W[:, :3, 3]

    # Apply translation and scaling adjustments
    cam_center = (cam_center + translate.unsqueeze(0).repeat(B, 1)) * scale

    # Update the first three elements of the last column of C2W
    C2W[:, :3, 3:4] = cam_center.unsqueeze(-1)

    # Compute the final transformation matrix by taking the inverse of C2W
    Rt = torch.inverse(C2W)

    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY, shiftX=0.0, shiftY=0.0):
    """Returns 4x4 projection matrix from view frustum parameters."""
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[0, 2] = shiftX
    P[1, 2] = (top + bottom) / (top - bottom)
    P[1, 2] = shiftY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrixTensor(znear, zfar, fovX, fovY, shiftX=0.0, shiftY=0.0):
    """Returns [..., 4,4] projection matrix from view frustum parameters.
    Args:
        znear: [...] Tensor
        zfar: [...]
        fovX: [...]
        fovY: [...]
        shiftX: [...]
        shiftY:  [...]
    Returns:
        [..., 4, 4] Tensor
    """
    # Calculate the tangent values using PyTorch's operations to handle tensors.
    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))

    # Calculate frustum boundaries. The operations are automatically broadcasted.
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # Create a tensor for the projection matrix. We need to ensure it has the correct shape
    # based on the input tensors. Here, we are creating a zero tensor of the right shape.
    # The shape is derived from the input tensors, with an added 4x4 at the end for the matrix itself.
    matrix_shape = znear.shape + (4, 4)
    P = torch.zeros(matrix_shape, dtype=znear.dtype, device=znear.device)

    z_sign = 1.0

    # Fill in the values. We are using PyTorch's operations to ensure compatibility with tensors.
    # The ellipsis is used to automatically handle the arbitrary number of dimensions.
    P[..., 0, 0] = 2.0 * znear / (right - left)
    P[..., 1, 1] = 2.0 * znear / (top - bottom)
    P[..., 0, 2] = (right + left) / (
        right - left
    ) + shiftX  # modified based on the assumed intention
    P[..., 1, 2] = (top + bottom) / (
        top - bottom
    ) + shiftY  # modified based on the assumed intention
    P[..., 3, 2] = z_sign
    P[..., 2, 2] = z_sign * zfar / (zfar - znear)
    P[..., 2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def calculate_2D_triangle_area(vertices):
    """Given [..., 3, 2] vertices, calculate the area of the triangle.
    Returns:
        [...,] Vector
    """
    # Calculate the cross product of two edges of the triangle
    y2y3 = vertices[..., 1, 1] - vertices[..., 2, 1]
    y3y1 = vertices[..., 2, 1] - vertices[..., 0, 1]
    y1y2 = vertices[..., 0, 1] - vertices[..., 1, 1]
    # (1/2) [x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2)]
    # Calculate the area as half the magnitude of the cross product
    area = (
        vertices[..., 0, 0] * y2y3
        + vertices[..., 1, 0] * y3y1
        + vertices[..., 2, 0] * y1y2
    )
    return 0.5 * area.abs()


def generate_random_point_in_tetrahedron(
    vertices, n_points_per_triangle: Union[torch.Tensor, int] = 1
):
    """
    Args:
        vertices: (..., tri, 4, 3)
        n_points_per_triangle: torch.Int (tri,)
    Returns:
        point: (..., P, 3)
        uvw: (..., P, 4)
    """
    # interleave the vertices to match n_points_per_triangle
    vertices = torch.repeat_interleave(vertices, n_points_per_triangle, dim=-3)
    # Generate random barycentric coordinates
    uvw = torch.rand(vertices.shape[:-2] + (4,)).to(vertices.device)

    # Ensure that the sum of barycentric coordinates is <= 1 (..., 3, 1)
    total = torch.sum(uvw, dim=-1, keepdim=True)
    uvw /= total
    point = torch.sum(uvw[..., None] * vertices, dim=-2)
    assert point.shape[-1] == 3
    assert uvw.shape[-1] == 4
    return point, uvw


def generate_random_point_in_triangle(
    vertices, n_points_per_triangle: Union[torch.Tensor, int] = 1
):
    """
    Args:
        vertices: (tri, 3, 3)
        n_points_per_triangle: torch.Int (tri,)
    Returns:
        point: (P, 3)
        uvw: (P, 3)
    """
    # interleave the vertices to match n_points_per_triangle
    vertices = torch.repeat_interleave(vertices, n_points_per_triangle, dim=-3)
    # Generate random barycentric coordinates
    uvw = torch.rand(vertices.shape[:-2] + (3,)).to(vertices.device)

    # Ensure that the sum of barycentric coordinates is <= 1 (..., 3, 1)
    total = torch.sum(uvw, dim=-1, keepdim=True)
    uvw /= total
    point = torch.sum(uvw[..., None] * vertices, dim=-2)
    assert point.shape[-1] == 3
    assert uvw.shape[-1] == 3
    return point, uvw


def plot_texture_hit(tex_faces, uv_verts, tex_bari, tex_res):
    """plot number of points per texel
    Args:
        tex_faces: (F, 3 or 4)
        uv_verts: (n_shell, V, 2)
        tex_bari: (P, 3 or 4)
        tex_res: texture resolution
    """
    # get xyz's uv
    interp_dim = tex_faces.shape[1]
    uv_verts.clamp_(0, 1)
    n_tex = uv_verts.shape[0]
    uv_verts_with_shidx = torch.concat(
        [
            torch.arange(n_tex, device=uv_verts.device)
            .reshape(n_tex, 1, 1)
            .to(torch.float32)
            .expand(-1, uv_verts.shape[1], -1),
            uv_verts,
        ],
        dim=-1,
    )  # (n_shell, V, 3)
    point_buv = torch.sum(
        tex_bari[..., None] * uv_verts_with_shidx.reshape(-1, 3)[tex_faces], dim=1
    )  # F, 3
    point_uv = point_buv[..., 1:]
    point_uv = (point_uv * (tex_res - 1) + 0.5).long()
    point_buv = torch.cat([point_buv[..., 0:1].long(), point_uv], dim=-1)
    count_map = torch.zeros(n_tex, tex_res, tex_res, device=point_uv.device)
    indices = (
        point_buv[:, 0] * (tex_res * tex_res)
        + point_buv[:, 1] * tex_res
        + point_buv[:, 2]
    )
    counts = torch.bincount(indices)
    count_map.view(-1)[: len(counts)] = counts
    count_map = count_map.permute(0, 2, 1)  # y coordinate is the first dimension
    return count_map


def save_texture_hit(tex_heatmap, output_dir):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        1, tex_heatmap.shape[0], figsize=(3 * tex_heatmap.shape[0], 3)
    )
    for i in range(len(axs)):
        im = axs[i].imshow(
            tex_heatmap[i], vmin=tex_heatmap.min(), vmax=tex_heatmap.max(), cmap="hot"
        )
        # axs[i].axis('off')
        axs[i].set_xticks([0, tex_heatmap[i].shape[1]])
        axs[i].set_yticks([0, tex_heatmap[i].shape[0]])
    cb_ax = fig.add_axes([0.91, 0.124, 0.04, 0.754])
    fig.colorbar(im, orientation="vertical", cax=cb_ax)
    fig.savefig(os.path.join(output_dir, "tex_hit.png"))
    plt.close()


@torch.no_grad()
def save_texures_to_images(gaussians, path, prefix=""):
    mkdir_p(path)
    scaling_img = (
        gaussians.get_scaling_planes().detach().cpu().numpy()
    )  # (shell, res, res, 3)
    rgb_img = gaussians.get_rgb_planes().detach().cpu().numpy()
    rot_img = (
        gaussians.get_rotation_planes().detach().cpu().numpy()
    )  # (shell, res, res, 4)
    opacity_img = (
        gaussians.get_opacity_planes().detach().cpu().numpy()
    )  # (shell, res, res, 1)

    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.titlesize": "small"})
    rgb_img = np.clip(rgb_img, 0, 1)
    fig, axs = plt.subplots(1, rgb_img.shape[0], figsize=(rgb_img.shape[0] * 3, 3))
    if rgb_img.shape[0] == 1: axs = [axs]
    for i in range(rgb_img.shape[0]):
        im = axs[i].imshow(rgb_img[i], norm="linear", interpolation="bilinear")
        axs[i].set_title("Shell {} DC".format(i))
        axs[i].set_xticks([0, rgb_img[i].shape[1]])
        axs[i].set_xticks([0, rgb_img[i].shape[1]])

    cb_ax = fig.add_axes([0.95, 0.124, 0.01, 0.754])
    fig.colorbar(im, orientation="vertical", cax=cb_ax)
    fig.savefig(os.path.join(path, f"{prefix}dc.png"))
    plt.clf()

    fig, axs = plt.subplots(
        1, opacity_img.shape[0], figsize=(opacity_img.shape[0] * 3, 3)
    )
    if opacity_img.shape[0] == 1: axs = [axs]
    for i in range(opacity_img.shape[0]):
        im = axs[i].imshow(
            opacity_img[i],
            vmin=opacity_img.min(),
            vmax=opacity_img.max(),
            norm="linear",
            cmap="gray",
            interpolation="bilinear",
        )
        axs[i].set_title("Shell {} opc".format(i))
        axs[i].axis("off")
    cb_ax = fig.add_axes([0.95, 0.124, 0.01, 0.754])
    fig.colorbar(im, orientation="vertical", cax=cb_ax)
    fig.savefig(os.path.join(path, f"{prefix}opacity.png"))
    plt.clf()

    fig, axs = plt.subplots(
        scaling_img.shape[-1],
        scaling_img.shape[0],
        figsize=(scaling_img.shape[0] * 3, scaling_img.shape[-1] * 3),
    )
    axs = axs.reshape(scaling_img.shape[-1], scaling_img.shape[0])
    for row in range(scaling_img.shape[-1]):
        for col in range(scaling_img.shape[0]):
            im = axs[row, col].imshow(
                scaling_img[col, ..., row],
                vmin=scaling_img.min(),
                vmax=scaling_img.max(),
                interpolation="bilinear",
                norm="linear",
                cmap="gray",
            )
            axs[row, col].set_title("Shell {} scaling {}".format(col, row))
            axs[row, col].axis("off")
    cb_ax = fig.add_axes([0.91, 0.124, 0.04, 0.754])
    fig.colorbar(im, orientation="vertical", cax=cb_ax)
    fig.savefig(os.path.join(path, f"{prefix}scaling.png"))
    plt.clf()


    fig, axs = plt.subplots(
        rot_img.shape[-1],
        rot_img.shape[0],
        figsize=(rot_img.shape[0] * 3, rot_img.shape[-1] * 3),
    )
    axs = axs.reshape(rot_img.shape[-1], rot_img.shape[0])
    for row in range(rot_img.shape[-1]):
        for col in range(rot_img.shape[0]):
            im = axs[row, col].imshow(
                rot_img[col, ..., row],
                vmin=rot_img.min(),
                vmax=rot_img.max(),
                norm="linear",
                cmap="gray",
                interpolation="bilinear",
            )
            axs[row, col].set_title("Shell {} rotation {}".format(col, row))
            axs[row, col].axis("off")
    cb_ax = fig.add_axes([0.91, 0.124, 0.04, 0.754])
    fig.colorbar(im, orientation="vertical", cax=cb_ax)
    fig.savefig(os.path.join(path, f"{prefix}rotation.png"))
    plt.clf()

    plt.close("all")


def load_model(base_shell_path, device):
    shell_base_verts, _faces, _aux = load_obj(base_shell_path, load_textures=False)
    shell_base_verts = shell_base_verts.to(device)
    shell_faces = _faces.verts_idx.to(device)
    faces_uvs = _faces.textures_idx.to(device)
    vertex_uvs = _aux.verts_uvs.to(device)
    return shell_base_verts, shell_faces, faces_uvs, vertex_uvs


def face_idx_of_vertices(vertex_indices, faces):
    """Given a list of vertex indices (N,), return the faces indices that contain them."""
    faces = faces.unsqueeze(0).expand(len(vertex_indices), -1, -1)  # (N, F, 3)
    vertex_indices = vertex_indices.reshape(-1, 1, 1).expand(-1, -1, 3)  # (N, 1, 3)
    faces_matched = torch.any(torch.any(faces == vertex_indices, dim=-1), dim=0)  # (N, F)
    return torch.nonzero(faces_matched)


@torch.no_grad()
def test_part_visibility(geometry, cameras, vertex_indices, device):
    """Use pytorch3d rasterizer to test if a given set of vertices are visible.
    Args:
        geometry: Meshes or Pointclouds. If mesh, suppose they share the same topology.
        cameras: pytorch3d cameras
        vertex_indices: List[Tensor] indices of vertices to test. Each tensor is a part to test
    Returns:
        visibility: List[Tensor(B,)] True/False as long as 10% of the vertices are visible
    """
    is_mesh = isinstance(geometry, Meshes)
    img_size = 128
    world2cam = cameras.world_view_transform.transpose(1, 2)
    intrinsics = cameras.projection_matrix.transpose(1, 2)
    # flip camera xy-axis because pytorch3d +x is left and +y is up
    intrinsics[:, [0, 1], [0, 1]] *= -1
    cameras = PerspectiveCameras(
        device=device,
        R=world2cam[:, :3, :3],
        T=world2cam[:, :3, 3],
        K=intrinsics,
        in_ndc=True,
    )
    if is_mesh:
        raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        pix_to_face = rasterize_meshes(
            geometry,
            img_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            clip_barycentric_coords=True,
            cull_backfaces=False,
            cull_to_frustum=True,
        )[0]
        # suppose sharing the same faces
        face_idx_of_vertices(vertex_indices, geometry.faces_list()[0])
    else:
        pix_to_pnts = rasterize_points(geometry, img_size=img_size, radius=0.01, points_per_pixel=1)[0]
    pass
