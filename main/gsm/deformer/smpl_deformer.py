import os
import sys
from typing import Optional, Union

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_gather
import smplx

from deformer.base_deformer import Deformer as BaseDeformer
from deformer.util import PointMeshCorrespondence, get_shell_verts_from_base, DeformerOutputs


model_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))), "assets"
)
smpl_init_kwargs = {
    "model_path": model_root,
    "model_type": "smpl",
    "num_betas": 10,
    "ext": ".npz",
}
flame_init_kwargs = {
    "model_path": model_root,
    "model_type": "flame",
    "num_betas": 100,
    "use_face_contour": True,
    "num_expression_coeffs": 50,
    "ext": "pkl",
}


class Deformer(BaseDeformer):
    def __init__(
        self, scale: Union[float, torch.Tensor] = 1.0, transl=None, **smpl_init_kwargs
    ):
        self.smplx_model = smplx.create(**smpl_init_kwargs)
        # scale
        if isinstance(scale, torch.Tensor):
            scale = scale.to(self.smplx_model.v_template.device)
        else:
            scale = torch.tensor(scale, dtype=torch.float32, device=self.smplx_model.v_template.device)

        self.smplx_model.v_template.data *= scale
        self.smplx_model.shapedirs.data *= scale.reshape(1, -1, 1)
        if hasattr(self.smplx_model, "expr_dirs"):
            self.smplx_model.expr_dirs.data *= scale.reshape(1, -1, 1)
        self.smplx_model.posedirs.data = scale.reshape(
            1, 1, -1
        ) * self.smplx_model.posedirs.data.reshape(
            -1, self.smplx_model.v_template.shape[0], 3
        )
        self.smplx_model.posedirs.data = self.smplx_model.posedirs.data.reshape(-1, self.smplx_model.v_template.shape[0]*3)
        # translation
        if transl is not None:
            self.smplx_model.transl.data += transl.reshape(3).to(device=self.smplx_model.transl.device)

        self.smplx_model.requires_grad_(False)

    def run(
        self,
        v_template: Optional[torch.Tensor] = None,
        deform_using_k_verts: bool = False,
        point_mesh_correspondence: Optional[PointMeshCorrespondence] = None,
        # bary_coords: Optional[torch.Tensor] = None,
        # point_face_idxs: Optional[torch.Tensor] = None,
        point_verts_weights: Optional[torch.Tensor] = None,
        point_verts_idxs: Optional[torch.Tensor] = None,
        **smpl_kwargs,
    ):
        """
        Similar to FLAME.forward, but allows using correspondence to deform.

        First, run the forward pass of the body model using (optional custom) v_template, which are v_template inflated.
        Second, use the correspondence to map the vpose (before applying rigging) to the vpose of the custom mesh.
        Third, use point_verts_weights and point_verts_idxs to deform the custom mesh using skinning weights.
        Args:
            v_template: (V, 3) or (num_shell, P, 3) optional template (default self.smplx_model.v_template)
            bary_coords: (num_shell or 1, P, 3) FloatTensor of barycentric coordinates per point on template faces
            point_face_idxs: (num_shell or 1, P) LongTensor of face indices per point on template faces
            point_verts_weights: (num_shell or 1, P, k) FloatTensor of weights per point on template vertices
            point_verts_idxs: (num_shell or 1, P, k) LongTensor of vertex indices per point on template vertices
            src_verts: (num_shell or 1, P, 3) optional src vertices, if not given, compute src_verts from precomputed correspondence
            smpl_kwargs: kwargs for body_model.forward can be (B, C), B can be different to num_shell
        Returns:
            smpl outputs, deformed outputs including points (B, N, 3) and (B, N, 4, 4) transform matrix (if using point_verts_weights)
        """
        # v_template can be the base template or a shell template
        _smpl_v_template = self.smplx_model.v_template.clone()

        batch_size_v = 1
        batch_size = 1
        if (body_pose := smpl_kwargs.get("betas")) is not None:
            batch_size = body_pose.shape[0]
        if v_template is not None:
            if v_template.shape[-2] != self.smplx_model.v_template.shape[0]:
                raise ValueError(
                    "v_template should have shape (V, 3), where V is the number of vertices in the parametric template"
                )

            # make sure if both v_template and betas are batched, they have the same batch size
            if v_template.ndim == 3:
                batch_size_v = v_template.shape[0]
                v_template = v_template.unsqueeze(dim=0)  # (batch_size, batch_size_v, V, 3)
                v_template = v_template.expand(batch_size, -1, -1, -1)
                v_template = v_template.reshape(-1, v_template.shape[-2], 3)
                for k in smpl_kwargs.keys():
                    smpl_kwargs[k] = smpl_kwargs[k].unsqueeze(1).expand(-1, batch_size_v, -1)
                    smpl_kwargs[k] = smpl_kwargs[k].reshape(-1, smpl_kwargs[k].shape[-1])

                    assert smpl_kwargs[k].shape[0] == batch_size * batch_size_v

            self.smplx_model.v_template.data = v_template

        ######## Start of LBS ########
        try:
            # Try to be general to support different smplx template
            # Run forward pass with custom v_template
            output = self.smplx_model(**smpl_kwargs)
            v_posed = output.v_posed
            # v_posed = _smpl_v_template.reshape(1, -1, 3).expand(smpl_kwargs["betas"].shape[0], -1, -1)
            A = output.transform_mat.reshape(batch_size, batch_size_v, -1, 4, 4)
            deformed_verts = output.vertices
        except Exception as e:
            self.smplx_model.v_template.data = _smpl_v_template
            raise e
        else:
            self.smplx_model.v_template.data = _smpl_v_template
        ####### End of LBS #########

        ####### Beginning of deformation #######
        batch_size = deformed_verts.shape[0]
        # compute src_verts from current template and correspondence
        if point_mesh_correspondence is not None:
            src_verts = point_mesh_correspondence.backproject(
                Meshes(
                    v_posed,
                    self.smplx_model.faces_tensor[None].expand(
                        v_posed.shape[0], -1, -1
                    ),
                )
            )

        # # Use barycentric coordinates and skinning weights to deform src_pnts
        # if not deform_using_k_verts and bary_coords is not None:
        #     src_batch_size, num_points = bary_coords.shape[:2]
        #     if src_batch_size != batch_size:
        #         assert (batch_size % src_batch_size == 0) or (
        #             batch_size == 1
        #         ), "batch size mismatch!"
        #         assert bary_coords.shape[:2] == point_face_idxs.shape[:2]

        #     deformed_pnts = self.deform_from_closest_points(
        #         bary_coords, point_face_idxs, A, src_verts
        #     )

        # Use point_verts_weights and point_verts_idxs to deform src_pnts
        if deform_using_k_verts and point_verts_weights is not None:
            src_batch_size, num_points = point_verts_weights.shape[:2]
            if src_batch_size != batch_size:
                assert (batch_size % src_batch_size == 0) or (
                    batch_size == 1
                ), "Expect batch_size to be a multiple of point_verts_weights.shape[0] or 1"
                assert point_verts_weights.shape[:2] == point_verts_idxs.shape[:2]

            deformed_pnts, T = self.deform_from_closest_points(
                point_verts_weights, point_verts_idxs, A, src_verts
            )
            deformed_pnts = deformed_pnts.reshape(batch_size, num_points, 3)
        else:
            deformed_pnts = deformed_verts
            src_batch_size = 1
            num_points = deformed_pnts.shape[1]
            T = None

        # apply translation because it's not part of v_posed
        transl = smpl_kwargs.get("transl")
        if smpl_kwargs.get("transl") is None:
            transl = self.smplx_model.transl
        if transl is not None:
            deformed_pnts += transl.unsqueeze(dim=-2)

        ####### End of deformation #######
        deformed_pnts = deformed_pnts.reshape(-1, src_batch_size, num_points, 3)

        return output, DeformerOutputs(deformed_pnts=deformed_pnts, weighted_transform_mat=T)


if __name__ == "__main__":
    TEMPLATE_SCALE = 1.0
    TEMPLATE_TRANS = [0, 0, 0]
    NUM_SHELLS = 5
    ASSET_DIR = "assets"
    PATH_BASE = os.path.join(ASSET_DIR, "smpl", "smpl_uv_hands_feet.sculpted.ear.obj")
    from deformer.util import (
        get_shell_verts_from_base,
        weights_from_k_closest_verts,
        PointMeshCorrespondence,
    )
    from pytorch3d.structures import Meshes
    from pytorch3d.io import load_obj, save_obj

    device = torch.device("cuda:0")

    # Load base mesh (custom SMPL)
    shell_base_verts, shell_faces, aux = load_obj(PATH_BASE, load_textures=False)
    shell_base_verts = shell_base_verts * TEMPLATE_SCALE + torch.tensor(
        [TEMPLATE_TRANS]
    )
    shell_faces = shell_faces.verts_idx.to(device=device)
    shell_base_verts = shell_base_verts[None].to(device=device)

    # Initialize 3DMM (e.g. SMPL or FLAME)
    deformer = Deformer(
        **smpl_init_kwargs, scale=TEMPLATE_SCALE, transl=torch.tensor(TEMPLATE_TRANS)
    )

    deformer.smplx_model = deformer.smplx_model.to(device)
    smpl_template_verts = deformer.smplx_model().vertices

    # Offset both to create shells
    shell_vertices = get_shell_verts_from_base(
        shell_base_verts,
        shell_faces,
        offset_len=0.03,
        num_shells=NUM_SHELLS,
    )
    shell_v_template = get_shell_verts_from_base(
        smpl_template_verts,
        deformer.smplx_model.faces_tensor,
        offset_len=0.03,
        num_shells=NUM_SHELLS,
    )

    # Compute correspondences for skinning weight (base shell to base shell)
    # and static deformation (all shells to shells)
    shell_vnormals = Meshes(shell_base_verts, shell_faces[None]).verts_normals_padded()
    smpl_template_vnormals = Meshes(
        smpl_template_verts, deformer.smplx_model.faces_tensor[None]
    ).verts_normals_padded()
    point_verts_weights, point_verts_idxs = weights_from_k_closest_verts(
        shell_base_verts,
        smpl_template_verts,
        k=5,
        points_normals=shell_vnormals,
        verts_normals=smpl_template_vnormals,
        normal_weight=0.1,
    )
    mesh_shells_template = Meshes(
        shell_v_template[0],
        deformer.smplx_model.faces_tensor[None].expand(NUM_SHELLS, -1, -1),
    )
    shells_correspondence = PointMeshCorrespondence(
        shell_vertices[0], meshes=mesh_shells_template
    )

    # Finally deform the shells on the custom SMPL based
    batch_size = 3
    shp_params = torch.zeros(batch_size, deformer.smplx_model.num_betas, device=device)
    pose_params = torch.zeros(
        batch_size, (deformer.smplx_model.NUM_BODY_JOINTS + 1) * 3, device=device
    )
    _, shell_deformed_verts = deformer.run(
        betas=shp_params,
        global_orient=pose_params[:, :3],
        body_pose=pose_params[:, 3:],
        deform_using_k_verts=True,
        v_template=shell_v_template[0],  # NUM_SHELLS, N_points, 3
        point_verts_idxs=point_verts_idxs.expand(NUM_SHELLS, -1, -1),
        point_verts_weights=point_verts_weights.expand(NUM_SHELLS, -1, -1),
        point_mesh_correspondence=shells_correspondence,
    )

    # shell_deformed_verts (num_shells, num_verts, 3) is then used drive Gaussian's deformation-