import PIL
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
            PerspectiveCameras, RasterizationSettings,
            DirectionalLights, MeshRasterizer, MeshRenderer, HardFlatShader,
            TexturesVertex
)



def render_meshes(meshes, world2cam, intrinsics, img_size, device):
    """Render meshes using pytorch3d renderer
    Returns:
        (B, H, W, 3) uint8 image
    """
    # flip camera xy-axis because pytorch3d +x is left and +y is up
    intrinsics[:, [0, 1], [0, 1]] *= -1
    cameras = PerspectiveCameras(device=device, R=world2cam[:, :3, :3], T=world2cam[:, :3, 3],
                                 K=intrinsics, in_ndc=True)
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device,
            cameras=cameras,
            lights=DirectionalLights(device=device,
                                    ambient_color=((0.2, 0.2, 0.2),),
                                    diffuse_color=((0.6, 0.6, 0.8),),
                                    specular_color=((0.2, 0.2, 0.0),),
                                    direction=[[0.5,1,-1.5]])
        )
    )
    MESH_COLOR = [50, 168, 168]
    verts_rgb = torch.ones_like(meshes.verts_padded()).to(device)
    verts_rgb[...,0] = MESH_COLOR[0]/255
    verts_rgb[...,1] = MESH_COLOR[1]/255
    verts_rgb[...,2] = MESH_COLOR[2]/255
    meshes.textures = TexturesVertex(verts_features=verts_rgb)
    mesh_img = mesh_renderer(meshes, cameras=cameras)
    mesh_img = (mesh_img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return mesh_img


if __name__ == "__main__":
    import zipfile
    import scipy.io as sio
    import imageio
    from utils.dataset_utils import parse_raw_labels, create_new_camera
    import torch
    import smplx
    from deformer.smpl_deformer import smpl_init_kwargs

    device = "cuda:0"
    vtx_col = torch.tensor([[50, 168, 168]], dtype=torch.float32).to(device) / 255.0

    dataset_path = "/mnt/Storage/data/SHHQ.zip"
    zip_file = zipfile.ZipFile(dataset_path)
    data_image, data_label = sio.loadmat(zip_file.open('dataset.mat', 'r'))['labels'][0]
    with zip_file.open(data_image.item(), 'r') as f:
        image = np.array(PIL.Image.open(f))
    imageio.imwrite("visualization_image.png", image)
    labels = parse_raw_labels(torch.from_numpy(data_label.reshape(1, -1)).to(device=device, dtype=torch.float32))
    camera = create_new_camera(labels, image_width=256, img_height=256, device=device)

    smpl_params = {}
    smpl_params["body_pose"] = labels["body_pose"].reshape(1, -1).to(device)
    smpl_params["betas"] = labels["betas"].reshape(1, -1).to(device)
    smpl_params["global_orient"] = labels["global_orient"].reshape(1, -1).to(device)
    smpl_params["transl"] = torch.zeros_like(smpl_params["body_pose"][:, :3]).to(device)

    smpl_init_kwargs = smpl_init_kwargs.copy()
    smpl_init_kwargs["gender"] = "neutral"
    smpl_model = smplx.create(**smpl_init_kwargs).to(device=device)

    smpl_out = smpl_model(**smpl_params)

    world2view = camera.world_view_transform.transpose(1, 2)
    intrinsics = camera.projection_matrix.transpose(1, 2)
    img = render_meshes(Meshes(smpl_out.vertices, smpl_model.faces_tensor[None]), world2view, intrinsics,
                      img_size=(camera.image_height, camera.image_width), device=device,
                      )[0]

    imageio.imwrite("visualization_mesh.png", img)
    zip_file.close()
