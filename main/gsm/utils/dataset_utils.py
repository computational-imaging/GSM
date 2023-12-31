
import zipfile
import numpy as np
import torch
from utils.graphics_utils import getProjectionMatrixTensor, fov2focal, focal2fov
from scene.cameras import MiniCam


def parse_raw_labels_ffhq(label_arr: torch.Tensor):
    """Parse raw labels from FHHQ dataset
    see `notebooks/merge_ffhq_flame_to_cam_json.py` for details.
    """
    device = label_arr.device
    extrinsic = label_arr[..., :16].reshape(*label_arr.shape[:-1], 4, 4)
    intrinsics = label_arr[..., 16:25]
    fov_deg = 18.837
    FovX = FovY = np.deg2rad(fov_deg)
    intrinsics = intrinsics.reshape(*label_arr.shape[:-1], 3, 3)
    T = torch.inverse(extrinsic)[..., :3, 3]
    betas = label_arr[..., 25:125]
    global_orient = label_arr[..., 125:128]
    jaw_pose = label_arr[..., 128:131]
    expression = label_arr[..., 131:]
    assert expression.shape[-1] == 50
    return dict(
        cam2world=extrinsic,
        T=T,
        fovx=torch.tensor(FovX, device=device, dtype=torch.float32).reshape(*label_arr.shape[:-1], 1),
        fovy=torch.tensor(FovY, device=device, dtype=torch.float32).reshape(*label_arr.shape[:-1], 1),
        principle_points=intrinsics[..., :2, 2],
        betas=betas,
        expression=expression,
        global_orient=global_orient,
        jaw_pose=jaw_pose,
    )

def update_smpl_to_raw_labels(label_arr, global_orient=None, body_pose=None, betas=None):
    base_idx = 25
    label_arr = label_arr.clone()
    if global_orient is not None:
        label_arr[..., base_idx:3+base_idx] = global_orient
    if body_pose is not None:
        label_arr[..., base_idx+3:72+base_idx] = body_pose
    if betas is not None:
        label_arr[..., base_idx+72:82+base_idx] = betas
    return label_arr


def parse_raw_labels(label_arr: torch.Tensor, scale=1.0):
    cam2world = label_arr[..., :16].reshape(*label_arr.shape[:-1],4,4)
    cam2world[...,  :3, 3] /= scale
    T=torch.inverse(cam2world)[..., :3, 3]
    intrinsics = label_arr[...,  16:25]
    smpl_params = label_arr[...,  25:107]
    global_orient = smpl_params[..., :3]
    body_pose = smpl_params[..., 3:72]
    betas = smpl_params[..., 72:82]
    # Change intrinsics to range [-1, 1]
    intrinsics = intrinsics.reshape(*label_arr.shape[:-1], 3, 3)
    # intrinsics[0, [0, 1], [0, 1]] *= 2
    fovx = 2 * torch.arctan2(torch.tensor(1.0), 2*intrinsics[..., 0, 0])
    fovy = 2 * torch.arctan2(torch.tensor(1.0), 2*intrinsics[..., 1, 1])
    return dict(cam2world=cam2world,
                T=T,
                intrinsics=intrinsics,
                fovy=fovy,
                fovx=fovx,
                principle_points=intrinsics[..., :2, 2],
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas
                )



def create_new_camera(parsed_labels, image_width, img_height, device):
    """Create new camera(s) from parsed labels."""

    # Calculations should handle any shape of input. We avoid squeezing or unsqueezing
    # which assumes specific dimensions.
    world_view_transform = torch.inverse(parsed_labels["cam2world"])

    # Using ellipsis to handle broadcasting with any leading dimensions
    shiftX = 2 * parsed_labels["principle_points"][..., 0] - 1
    shiftY = 2 * parsed_labels["principle_points"][..., 1] - 1

    # We assume getProjectionMatrix is capable of handling broadcasting
    proj_matrix = getProjectionMatrixTensor(
        znear=torch.tensor(0.01, device=device).expand_as(shiftX),
        zfar=torch.tensor(100, device=device).expand_as(shiftX),
        fovX=parsed_labels["fovx"],
        fovY=parsed_labels["fovy"],
        shiftX=shiftX, shiftY=shiftY
    ).to(device)

    # We're using the new function 'permute_last_dims' here instead of directly calling 'permute'.
    world_view_transform_permuted = torch.transpose(world_view_transform, -1, -2).to(device)
    full_proj_transform_permuted = torch.transpose(proj_matrix @ world_view_transform, -1, -2).to(device)
    proj_matrix_permuted = torch.transpose(proj_matrix, -1, -2).to(device)

    # Note: the constructor of MiniCam should also support inputs with arbitrary shapes
    cam = MiniCam(
        width=image_width, height=img_height,
        fovx=parsed_labels["fovx"], fovy=parsed_labels["fovy"],
        znear=0.01,
        zfar=100,
        world_view_transform=world_view_transform_permuted,
        full_proj_transform=full_proj_transform_permuted,
    )

    # Set the projection matrix with correct permutation
    cam.projection_matrix = proj_matrix_permuted

    return cam


def get_zipfile(zip_path: str):
    zipfile = zipfile.ZipFile(zip_path)
    return zipfile

def open_file(zipfile:zipfile.ZipFile, fname):
    return zipfile.open(fname, 'r')