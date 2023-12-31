# import smplx
# import torch
# import numpy as np
# import sys
# import os

# current_directory = os.path.dirname(os.path.abspath(__file__))

# # Define the number of levels to go up
# levels_to_go_up = 3  # Adjust this number as needed

# # Navigate up the directory structure
# for _ in range(levels_to_go_up):
#     current_directory = os.path.abspath(os.path.join(current_directory, '..'))

# # Append the final parent directory to sys.path
# sys.path.append(current_directory)
# from pytorch3d.structures import Meshes
# from pytorch3d.io import load_obj
# from utils import persistence
# from scene.gaussian_model_tets import TetGaussianModel
# from scene.gaussian_model_shells import ShellGaussianModel

# from deformer.smpl_deformer import Deformer, smpl_init_kwargs
# from deformer.util import (
#     weights_from_k_closest_verts,
#     get_shell_verts_from_base,
#     PointMeshCorrespondence,
#     interpolate_mesh_from_bary_coords,
# )


# def deform_gaussians(
#     gaussian_model, deformer: Deformer, smpl_kwargs: dict
# ) -> torch.Tensor:
#     """Set GaussianModel xyz's using smpl deformation model
#     Args:
#         shell_deformed_verts: (S, V, 3) FloatTensor shell vertices after deformation
#         smpl_kwargs: dict of smpl params in batch with batch size = B
#     Returns:
#         new_xyz: (P, 3) FloatTensor after deformation
#         smpl_out: SMPLOutput
#     """
#     with torch.no_grad():
#         num_shells = gaussian_model.num_shells
#         smpl_out, shell_deformed_verts = deformer.run(
#             # betas=betas.expand(num_shells, -1),
#             # global_orient=torch.zeros_like(global_orient).expand(num_shells, -1),
#             # body_pose=body_pose.expand(num_shells, -1),
#             # transl=torch.zeros_like(transl).expand(num_shells, -1),
#             deform_using_k_verts=True,
#             v_template=gaussian_model.smpl_shell_verts,  # num_shells, N_points, 3
#             point_verts_idxs=gaussian_model.point_verts_idxs.expand(num_shells, -1, -1),
#             point_verts_weights=gaussian_model.point_verts_weights.expand(
#                 num_shells, -1, -1
#             ),
#             point_mesh_correspondence=gaussian_model.shells_correspondence,
#             **smpl_kwargs,
#         )
#         # # (B, S*V, 3)
#         # shell_deformed_verts = shell_deformed_verts.reshape(-1, num_shells*gaussian_model.shell_verts.shape[-2], 3)
#         # (B, P, 3)
#         new_xyz = interpolate_mesh_from_bary_coords(
#             shell_deformed_verts.reshape(
#                 -1, num_shells * gaussian_model.shell_verts.shape[-2], 3
#             ),
#             gaussian_model.faces.to(shell_deformed_verts.device),
#             gaussian_model.bari.reshape(1, -1, gaussian_model.interp_dim)
#             .expand(shell_deformed_verts.shape[0], -1, -1)
#             .to(shell_deformed_verts.device),
#         )
#         return new_xyz, smpl_out, shell_deformed_verts


# # def _load_base_model(base_shell_path, device):
# #     shell_base_verts, _faces, _aux = load_obj(base_shell_path, load_textures=False)
# #     shell_base_verts = shell_base_verts.to(device)
# #     shell_faces = _faces.verts_idx.to(device)
# #     faces_uvs = _faces.textures_idx.to(device)
# #     vertex_uvs = _aux.verts_uvs.to(device)
# #     return shell_base_verts, shell_faces, faces_uvs, vertex_uvs


# class DeformableTetGaussianModel(torch.nn.Module):
#     def __init__(
#         self,
#         base_shell_path,
#         sh_degree: int,
#         num_shells=5,
#         offset_len=0.1,
#         res=128,
#         smpl_init_kwargs=None,
#         smpl_can_params=None,
#         random_pts = True,
#         device="cpu",
#         scale_act='exp'
#     ):
#         print("DeformableTetGaussianModel init")

#         shell_base_verts, shell_faces, faces_uvs, vertex_uvs  = _load_base_model(base_shell_path, device)

#         # Create deformer, will set self.shell_verts
#         setup_deformer(
#             self,
#             shell_base_verts=shell_base_verts,
#             shell_faces=shell_faces,
#             num_shells=num_shells,
#             offset_len=offset_len,
#             smpl_init_kwargs=smpl_init_kwargs,
#             smpl_can_params=smpl_can_params,
#             device=device,
#         )
#         self._gaussians = TetGaussianModel(
#             shell_verts=self.shell_verts,
#             shell_faces=shell_faces,
#             faces_uvs=faces_uvs,
#             vertex_uvs=vertex_uvs,
#             sh_degree=sh_degree,
#             num_shells=num_shells,
#             res=res,
#             scale_act=scale_act
#         )
#         self.register_buffer("shell_verts", self._shell_verts.clone().to(device))
#         self.register_buffer("smpl_shell_verts", self._smpl_shell_verts.clone().to(device))
#         self.register_buffer("point_verts_weights", self._point_verts_weights.clone().to(device))
#         self.register_buffer("point_verts_idxs", self._point_verts_idxs.clone().to(device))

#     def get_xyz(self, **smpl_kwargs):
#         return deform_gaussians(self, self.deformer, smpl_kwargs)[0]


# class DeformableShellGaussianModel(torch.nn.Module):
#     def __init__(
#         self,
#         base_shell_path,
#         sh_degree: int,
#         num_shells=5,
#         offset_len=0.1,
#         res=128,
#         smpl_init_kwargs=None,
#         smpl_can_params=None,
#         random_pts = True,
#         device="cpu",
#         scale_act='exp'
#     ):
#         print("DeformableShellGaussianModel init")
#         super().__init__()
#         shell_base_verts, shell_faces, faces_uvs, vertex_uvs  = _load_base_model(base_shell_path, device)
#         # Create deformer, will set _shell_verts,
#         # _smpl_shell_verts, _point_verts_weights, _point_verts_idxs
#         setup_deformer(
#             self,
#             shell_base_verts=shell_base_verts,
#             shell_faces=shell_faces,
#             num_shells=num_shells,
#             offset_len=offset_len,
#             smpl_init_kwargs=smpl_init_kwargs,
#             smpl_can_params=smpl_can_params,
#             device=device,
#         )

#         super().__init__(
#             shell_verts=self._shell_verts,
#             shell_faces=shell_faces,
#             faces_uvs=faces_uvs,
#             vertex_uvs=vertex_uvs,
#             sh_degree=sh_degree,
#             num_shells=num_shells,
#             res=res,
#             scale_act=scale_act
#         )
#         self.register_buffer("shell_verts", self._shell_verts.clone().to(device))
#         self.register_buffer("smpl_shell_verts", self._smpl_shell_verts.clone().to(device))
#         self.register_buffer("point_verts_weights", self._point_verts_weights.clone().to(device))
#         self.register_buffer("point_verts_idxs", self._point_verts_idxs.clone().to(device))

#     def get_xyz(self, **smpl_kwargs):
#         with torch.no_grad():
#             return deform_gaussians(self, self.deformer, smpl_kwargs)[0]


# if __name__ == "__main__":
#     import torch
#     import numpy as np
#     import os
#     import json
#     from PIL import Image
#     import matplotlib.pyplot as plt
#     from pytorch3d.io import load_obj, save_obj
#     from pytorch3d.structures import Meshes
#     import sys
#     from deformer.smpl_deformer import smpl_init_kwargs
#     from torch import nn
#     from scene.cameras import Camera
#     from utils.graphics_utils import getWorld2View2, getProjectionMatrix
#     from gaussian_renderer import render

#     os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
#     sys.path.insert(0, os.path.join(os.getcwd(), ".."))

#     device = "cuda:0"

#     data_dir = os.path.join("data", "people_snapshot", "male-1-plaza")

#     NUM_SHELLS = 2
#     OFFSET_LEN = 0.03
#     RES = 1024

#     model_root = os.path.join("assets")
#     model_type = "smpl"
#     base_shell_path = os.path.join(
#         model_root, model_type, "smpl_uv_hands_feet.sculpted.ear.obj"
#     )

#     smpl_init_kwargs = smpl_init_kwargs.copy()
#     smpl_init_kwargs["gender"] = "male"
#     with open(os.path.join(data_dir, "cameras.json"), "r") as f:
#         loaded_json = json.load(f)

#     def query_smpl_param(frame_idx, batch_size=1):
#         body_pose = torch.tensor(
#             [
#                 loaded_json[f"{frame_idx+bid:04d}.png"]["body_pose"]
#                 for bid in range(batch_size)
#             ],
#             dtype=torch.float32,
#         ).to(device)
#         betas = torch.tensor(
#             [
#                 loaded_json[f"{frame_idx+bid:04d}.png"]["betas"]
#                 for bid in range(batch_size)
#             ],
#             dtype=torch.float32,
#         ).to(device)
#         global_orient = torch.tensor(
#             [
#                 loaded_json[f"{frame_idx+bid:04d}.png"]["global_orient"]
#                 for bid in range(batch_size)
#             ],
#             dtype=torch.float32,
#         ).to(device)
#         transl = torch.tensor(
#             [
#                 loaded_json[f"{frame_idx+bid:04d}.png"]["transl"]
#                 for bid in range(batch_size)
#             ],
#             dtype=torch.float32,
#         ).to(device)
#         return {
#             "betas": betas,
#             "body_pose": body_pose,
#             "global_orient": global_orient,
#             "transl": transl,
#         }

#     frame_idx = 0
#     batch_size = 4
#     gaussians = GaussianModel(
#         base_shell_path,
#         sh_degree=0,
#         num_shells=NUM_SHELLS,
#         offset_len=OFFSET_LEN,
#         res=RES,
#         smpl_init_kwargs=smpl_init_kwargs,
#         smpl_can_params=query_smpl_param(frame_idx=frame_idx, batch_size=1),
#         device=device,
#     )

#     gaussians.save_ply(os.path.join("./gaussians_init.ply"))

#     xyz_bk = gaussians._xyz.data.clone()
#     new_xyz = gaussians.get_xyz(
#         **query_smpl_param(frame_idx=frame_idx, batch_size=batch_size)
#     )
#     gaussians._xyz.data[:] = new_xyz[0]
#     gaussians.save_ply(os.path.join("./gaussians_{:02d}.ply").format(frame_idx))

#     frame_idx = 50
#     new_xyz = gaussians.get_xyz(
#         **query_smpl_param(frame_idx=frame_idx, batch_size=batch_size)
#     )
#     gaussians._xyz.data[:] = new_xyz[0]
#     gaussians.save_ply(os.path.join("./gaussians_{:02d}.ply").format(frame_idx))

#     gaussians._xyz.data[:] = xyz_bk

#     ######################## Render ########################
#     # how many frames in the sequence
#     num_frames = 0
#     while True:
#         if f"{num_frames:04d}.png" not in loaded_json:
#             break
#         num_frames = num_frames + 1
#     cam2world_loaded = np.zeros((num_frames, 4, 4))
#     for frame_idx in range(num_frames):
#         cam2world_loaded[frame_idx] = np.array(
#             loaded_json[f"{frame_idx:04d}.png"]["cam2world"]
#         ).reshape(4, 4)

#     world2cam = torch.tensor(cam2world_loaded, dtype=torch.float32).inverse()
#     world2cam = world2cam.numpy()

#     class Camera(nn.Module):
#         def __init__(
#             self,
#             R,
#             T,
#             FoVx,
#             FoVy,
#             # image, gt_alpha_mask,
#             img_width,
#             img_height,
#             trans=np.array([0.0, 0.0, 0.0]),
#             scale=1.0,
#             data_device="cuda",
#         ):
#             super(Camera, self).__init__()

#             self.R = R
#             self.T = T
#             self.FoVx = FoVx
#             self.FoVy = FoVy

#             try:
#                 self.data_device = torch.device(data_device)
#             except Exception as e:
#                 print(e)
#                 print(
#                     f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
#                 )
#                 self.data_device = torch.device("cuda")

#             self.image_width = img_width
#             self.image_height = img_height

#             self.zfar = 100.0
#             self.znear = 0.01

#             self.trans = trans
#             self.scale = scale

#             self.world_view_transform = (
#                 torch.tensor(getWorld2View2(R, T, trans, scale))
#                 .transpose(0, 1)
#                 .to(self.data_device)
#             )
#             self.projection_matrix = (
#                 getProjectionMatrix(
#                     znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
#                 )
#                 .transpose(0, 1)
#                 .to(self.data_device)
#             )
#             self.full_proj_transform = (
#                 self.world_view_transform.unsqueeze(0).bmm(
#                     self.projection_matrix.unsqueeze(0)
#                 )
#             ).squeeze(0)
#             self.camera_center = self.world_view_transform.inverse()[3, :3]

#     # img_width, img_height = Image.open(os.path.join(data_dir, "images", "0000.png")).size
#     FovX = loaded_json["fovx"]
#     FovY = loaded_json["fovy"]

#     bg_color = [0.5, 0.5, 0.5]
#     background = torch.tensor(
#         bg_color, dtype=torch.float32, device=gaussians.get_xyz().device
#     )
#     class PipelineParams:
#         def __init__(self):
#             self.convert_SHs_python = True
#             self.compute_cov3D_python = False
#             self.debug = False

#     pipeline = PipelineParams()
#     cam_id = 0
#     camera = Camera(
#         world2cam[cam_id][:3, :3].T,
#         world2cam[cam_id][:3, 3],
#         FovX,
#         FovY,
#         512,
#         512,
#         data_device=background.device,
#     )

#     # Set the DC texture manually
#     img = Image.open(os.path.join(model_root, model_type, "cube_uv.png"))
#     img = img.resize((RES, RES))
#     img = np.array(img).astype(np.float32) / 255.0

#     gaussians._features_dc.data[:] = (
#         torch.tensor(img)
#         .unsqueeze(0)
#         .to(gaussians._features_dc.device)
#     )
#     gaussians._opacity.data[:] = gaussians.inverse_opacity_activation(torch.ones_like(gaussians._opacity))

#     for cam_id in (0, 50):
#         smpl_params = query_smpl_param(frame_idx=cam_id)
#         out = render(
#             camera, gaussians, pipeline, background, deformation_kwargs=smpl_params
#         )
#         rendering = out["render"]

#     im = np.uint8(rendering.detach().permute(1, 2, 0).cpu().numpy() * 255)
#     Image.fromarray(im).save(os.path.join("./rendering_{:02d}.png").format(cam_id))
