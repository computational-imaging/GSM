import os
import smplx
import torch
from deformer.smpl_deformer import Deformer
from deformer.util import (DeformerOutputs, PointMeshCorrespondence,
                           get_shell_verts_from_base,
                           interpolate_mesh_from_bary_coords,
                           weights_from_k_closest_verts)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (matrix_to_quaternion,
                                  quaternion_to_matrix)


def setup_deformer(
    gaussians,
    shell_base_verts,
    shell_faces,
    num_shells=5,
    offset_len=0.1,
    smpl_init_kwargs=None,
    smpl_can_params=None,
    smpl_scale=1.0,
    smpl_transl=None,
    shrunk_ref_mesh=None,
    use_base_shell_for_correspondence=False,
    device="cpu",
):
    # Create deformer
    with torch.no_grad():
        deformer = Deformer(scale=smpl_scale, transl=None, **smpl_init_kwargs)
        deformer.to(device)

        shell_base_verts *= smpl_scale

        # Set up correspondence to defaut neutral 3DMM template (TPose)
        _smpl_init_kwargs = smpl_init_kwargs.copy()
        _smpl_init_kwargs["gender"] = "neutral"
        _smplx_model = smplx.create(**_smpl_init_kwargs).to(device)
        _smpl_template_verts = _smplx_model.forward().vertices.detach()
        _smpl_template_verts *= smpl_scale

        # Deform the shell using closest k points on the smpl mesh.
        # Currently, the closest points are found between every shell (the custom template
        # and the smpl are both inflated/deflated).
        # TODO try find closest points between the shells and the base smpl mesh
        shell_vertices = get_shell_verts_from_base(
            shell_base_verts[None],
            shell_faces,
            offset_len=offset_len,
            num_shells=num_shells,
            shrunk_ref_mesh=shrunk_ref_mesh,
        )[
            0
        ]  # (num_shells, num_verts, 3)
        shrunk_smpl_mesh = None
        if shrunk_ref_mesh is not None:
            if "smpl" in shrunk_ref_mesh:
                shrunk_smpl_mesh = os.path.join(os.path.dirname(shrunk_ref_mesh), "smpl_uv_shrunk.obj")
            # TODO: add the same for flame
        _shell_v_template = get_shell_verts_from_base(
            _smpl_template_verts,
            deformer.smplx_model.faces_tensor,
            offset_len=offset_len,
            num_shells=num_shells,
            shrunk_ref_mesh=shrunk_smpl_mesh,
        )[
            0
        ]  # (num_shells, num_verts_template, 3)
        shell_vnormals = Meshes(
            shell_base_verts[None], shell_faces[None]
        ).verts_normals_padded()
        smpl_template_vnormals = Meshes(
            _smpl_template_verts, deformer.smplx_model.faces_tensor[None]
        ).verts_normals_padded()
        # SMPL template to Custom Shells
        # Skinning weights look up: the first option does not care about normals, and
        # assigns correspondences for all shells using the base pairs.
        # The second option uses normals and assigns correspondences for all shells
        # independently.
        point_verts_weights, point_verts_idxs = weights_from_k_closest_verts(
            shell_vertices,
            _smpl_template_verts.expand(num_shells, -1, -1),
            k=5,
            normal_weight=0.0,
        )
        # point_verts_weights, point_verts_idxs = weights_from_k_closest_verts(
        #     shell_base_verts[None],
        #     _smpl_template_verts,
        #     k=5,
        #     points_normals=shell_vnormals,
        #     verts_normals=smpl_template_vnormals,
        #     normal_weight=0.1,
        # )

        # This correspondence is used to deform the custom shells based on deformed smpl template
        # Compute correspondence for all shells using the base custom template and smpl template
        gaussians.shells_correspondence = PointMeshCorrespondence(
            shell_vertices,
            meshes=Meshes(
                _smpl_template_verts.expand(num_shells, -1, -1) if use_base_shell_for_correspondence else _shell_v_template, # _smpl_template_verts.expand(num_shells, -1, -1),
                deformer.smplx_model.faces_tensor[None].expand(num_shells, -1, -1),
            ),
        )
        gaussians.use_base_shell_for_correspondence = use_base_shell_for_correspondence
        # # test backprojection
        # shell_vertices_bp = gaussians.shells_correspondence.backproject(
        #     meshes=Meshes(
        #         _smpl_template_verts.expand(num_shells, -1, -1) if use_base_shell_for_correspondence else _shell_v_template, # _smpl_template_verts.expand(num_shells, -1, -1),
        #         deformer.smplx_model.faces_tensor[None].expand(num_shells, -1, -1),
        #     ))
        # from pytorch3d.io import save_ply
        # [save_ply(f"dbg_backproj_{i}.ply", shell_vertices_bp[i]) for i in range(shell_vertices_bp.shape[0])]
        # [save_ply(f"dbg_backproj_ref_{i}.ply", shell_vertices[i]) for i in range(shell_vertices_bp.shape[0])]
        # assert torch.allclose(shell_vertices, shell_vertices_bp, atol=1e-5), "Backprojection failed"

        # Create actual template shells using the gendered smpl template
        if smpl_init_kwargs["gender"] != "neutral":
            smpl_template_verts = deformer.smplx_model.forward().vertices.detach()
            shell_v_template = get_shell_verts_from_base(
                smpl_template_verts,
                deformer.smplx_model.faces_tensor,
                offset_len=offset_len,
                num_shells=num_shells,
                shrunk_smpl_mesh=shrunk_smpl_mesh
            )[
                0
            ]  # (num_shells, num_verts_template, 3)
        else:
            shell_v_template = _shell_v_template

        # Deform the shells to get the canonical pose (may be different to Tpose)
        gaussians._smpl_can_params = smpl_can_params
        shell_deformed_verts = shell_vertices
        if gaussians._smpl_can_params is not None:
            # deform the shells using the canonical smpl params
            for k, v in smpl_can_params.items():
                assert v.ndim == 2, "Must have batch dimension"
                if v.shape[0] == 1:
                    smpl_can_params[k] = v.expand(num_shells, -1)
            _, shell_deformed_verts = deformer.run(
                **smpl_can_params,
                deform_using_k_verts=True,
                v_template=shell_v_template,  # num_shells, N_points, 3
                point_verts_idxs=point_verts_idxs.expand(num_shells, -1, -1),
                point_verts_weights=point_verts_weights.expand(num_shells, -1, -1),
                point_mesh_correspondence=gaussians.shells_correspondence,
            )
            shell_deformed_verts = shell_deformed_verts[0]

        assert shell_vertices.shape == (num_shells, shell_base_verts.shape[0], 3)
        assert shell_deformed_verts.shape == (num_shells, shell_base_verts.shape[0], 3)
        assert shell_v_template.shape == (
            num_shells,
            deformer.smplx_model.v_template.shape[0],
            3,
        )
        assert point_verts_weights.shape[1] == shell_base_verts.shape[0]
        assert point_verts_idxs.shape[1] == shell_base_verts.shape[0]


        # Log buffers
        gaussians._tpose_shell_verts = shell_vertices.to(device) # (num_shells, num_verts, 3)
        gaussians._shell_verts = shell_deformed_verts.to(device) # (num_shells, num_verts, 3)
        gaussians._smpl_shell_verts = shell_v_template.to(device) # (num_shells, num_verts_tmp, 3)
        gaussians._point_verts_weights = point_verts_weights.to(device) # (1, num_verts, k) FloatTensor
        gaussians._point_verts_idxs = point_verts_idxs.to(device) # (1, num_verts, k) LongTensor
        gaussians._shell_faces = shell_faces.to(device)  # (F, 3)

        return deformer


@torch.no_grad()
def deform_gaussians(
    gaussian_model, deformer: Deformer, smpl_kwargs: dict
) -> torch.Tensor:
    """Set GaussianModel xyz's using smpl deformation model
    Args:
        shell_deformed_verts: (S, V, 3) FloatTensor shell vertices after deformation
        smpl_kwargs: dict of smpl params in batch with batch size = B
    Returns:
        new_xyz: (P, 3) FloatTensor after deformation
        smpl_out: SMPLOutput
    """
    with torch.no_grad():
        num_shells = gaussian_model.num_shells
        smpl_out, deformation_outputs = deformer.run(
            # betas=betas.expand(num_shells, -1),
            # global_orient=torch.zeros_like(global_orient).expand(num_shells, -1),
            # body_pose=body_pose.expand(num_shells, -1),
            # transl=torch.zeros_like(transl).expand(num_shells, -1),
            deform_using_k_verts=True,
            v_template=None if getattr(gaussian_model, "use_base_shell_for_correspondence", False) else gaussian_model.smpl_shell_verts,  # num_shells, N_points, 3
            point_verts_idxs=gaussian_model.point_verts_idxs.expand(num_shells, -1, -1),
            point_verts_weights=gaussian_model.point_verts_weights.expand(
                num_shells, -1, -1
            ),
            point_mesh_correspondence=gaussian_model.shells_correspondence,
            **smpl_kwargs,
        )
        shell_deformed_verts = deformation_outputs.deformed_pnts
        shell_weighted_transform_quat = matrix_to_quaternion(deformation_outputs.weighted_transform_mat[..., :3, :3])

        # # (B, S*V, 3)
        # shell_deformed_verts = shell_deformed_verts.reshape(-1, num_shells*gaussian_model.shell_verts.shape[-2], 3)
        # (B, P, 3)
        new_xyz = interpolate_mesh_from_bary_coords(
            shell_deformed_verts.reshape(
                -1, num_shells * gaussian_model.shell_verts.shape[-2], 3
            ),
            gaussian_model.faces.to(shell_deformed_verts.device),
            gaussian_model.bari.reshape(1, -1, gaussian_model.interp_dim)
            .expand(shell_deformed_verts.shape[0], -1, -1)
            .to(shell_deformed_verts.device),
        )
        weighted_transform_quat = interpolate_mesh_from_bary_coords(
        shell_weighted_transform_quat.reshape(
                -1, num_shells * gaussian_model.shell_verts.shape[-2], 4
            ),
            gaussian_model.faces.to(shell_deformed_verts.device),
            gaussian_model.bari.reshape(1, -1, gaussian_model.interp_dim)
            .expand(shell_deformed_verts.shape[0], -1, -1)
            .to(shell_deformed_verts.device),
        )
        weighted_transform_quat = torch.nn.functional.normalize(weighted_transform_quat, dim=-1)
        weighted_transform_mat = quaternion_to_matrix(weighted_transform_quat)
        return DeformerOutputs(deformed_pnts=new_xyz, weighted_transform_mat=weighted_transform_mat, deformed_shell_verts=shell_deformed_verts)