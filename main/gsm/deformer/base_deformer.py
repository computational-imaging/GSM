import os
from typing import Optional

import numpy as np
import torch
from pytorch3d.ops.knn import knn_gather

from smplx.lbs import (
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)
from deformer.util import DeformerOutputs


class Deformer:
    def to(self, device):
        self.smplx_model = self.smplx_model.to(device)
        return self

    @property
    def v_template(self):
        return self.smplx_model.v_template + self.smplx_model.transl

    def run(
        self,
        v_template: Optional[torch.Tensor] = None,
        deform_using_k_verts: bool = False,
        src_verts: Optional[torch.Tensor] = None,
        bary_coords: Optional[torch.Tensor] = None,
        point_face_idxs: Optional[torch.Tensor] = None,
        point_verts_weights: Optional[torch.Tensor] = None,
        point_verts_idxs: Optional[torch.Tensor] = None,
        **smpl_kwargs
    ):
        """deformation function"""
        raise NotImplementedError

    def _deform_from_closest_face(
        self, bary_coords, point_face_idxs, transform_mat, src_verts
    ):
        """Use barycentric coordinates query skinning weights then deform"""
        src_batch_size, num_points = bary_coords.shape[:2]

        # compute skinning weights for given points using barycentric coordinates
        lbs_weights_packed = self.smplx_model.lbs_weights  # (V, J+1)
        lbs_weights_tris = lbs_weights_packed[
            self.smplx_model.faces_tensor
        ]  # (F, 3, J+1)
        assert lbs_weights_tris.shape == (
            self.smplx_model.faces_tensor.shape[0],
            3,
            self.smplx_model.lbs_weights.shape[-1],
        )
        lbs_weights_pnts = lbs_weights_tris[point_face_idxs]  # (N, P, 3, J+1)
        lbs_weights_pnts = torch.einsum("npkj,npk->npj", lbs_weights_pnts, bary_coords)
        assert lbs_weights_pnts.shape == (
            src_batch_size,
            num_points,
            self.smplx_model.lbs_weights.shape[-1],
        )

        # perform lbs
        num_joints = self.smplx_model.J_regressor.shape[0]
        W = lbs_weights_pnts
        T = torch.matmul(W, transform_mat.view(src_batch_size, num_joints, 16)).view(
            src_batch_size, -1, 4, 4
        )
        homogen_coord = torch.ones(
            [src_batch_size, src_verts.shape[1], 1],
            dtype=src_verts.dtype,
            device=src_verts.device,
        )
        v_posed_homo = torch.cat([src_verts, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        deformed_pnts = v_homo[:, :, :3, 0]
        return deformed_pnts

    def deform_from_closest_points(
        self, point_verts_weights, point_verts_idxs, transform_mat,
        src_verts=None
    ):
        """Interpolate the skinning weights from the k closest vertices
        Args:
            point_verts_weights: (num_shells, P, k) interpolating weights from each P point
                to its k closest template vertices
            point_verts_idxs: (num_shells, P, k) indices of the k nearest vertices
            transform_mat: (n, num_shells, J, 4, 4)
            src_verts: (n, num_shells, P, 3)
        Returns:
            deformed_pts: (n, num_shells, P, 3)
            T: (n, num_shells, P, 4, 4)
        """
        num_shells, num_points = point_verts_weights.shape[:2]


        # compute skinning weights for given points using barycentric coordinates
        lbs_weights_packed = self.smplx_model.lbs_weights  # (V, J+1)
        lbs_weights_pnts = torch.stack(
            [lbs_weights_packed[idxs] for idxs in point_verts_idxs]
        )  # (bs, P, K, J+1)
        lbs_weights_pnts = torch.einsum(
            "npkj,npk->npj", lbs_weights_pnts, point_verts_weights
        )
        assert lbs_weights_pnts.shape == (
            num_shells,
            num_points,
            self.smplx_model.lbs_weights.shape[-1],
        )

        # perform lbs
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.smplx_model.J_regressor.shape[0]
        W = lbs_weights_pnts
        T = torch.matmul(W, transform_mat.view(-1, num_shells, num_joints, 16)).view(
            -1, num_shells, num_points, 4, 4
        )
        deformed_pnts = None
        if src_verts is not None:
            num_verts = src_verts.shape[-2]
            assert num_verts == num_points
            homogen_coord = torch.ones_like(
                # [src_batch_size, src_verts.shape[1], 1],
                src_verts.reshape(-1, num_shells, num_verts, 3)[..., :1],
                dtype=src_verts.dtype,
                device=src_verts.device,
            )
            v_posed_homo = torch.cat([src_verts.reshape(-1, num_shells, num_verts, 3), homogen_coord], dim=-1)
            v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

            deformed_pnts = v_homo[..., :3, 0]

        return deformed_pnts, T