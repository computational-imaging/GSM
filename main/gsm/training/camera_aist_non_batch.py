import math
import os
import sys

import numpy as np
import torch
from scene.cameras import Camera, MiniCam
from torch import nn
from utils.general_utils import PILtoTorch
from utils.graphics_utils import (focal2fov, fov2focal, getProjectionMatrix,
                                  getWorld2View2)


def parse_raw_labels(label_arr):
    # scale = 1.0/3.23
    cam2world = label_arr[:16].reshape(4,4)
    # cam2world[:3, 3] /= scale
    try:
        T=torch.inverse(cam2world)[:3, 3]
    except:
        print(cam2world)
    intrinsics = label_arr[16:25]
    # intrinsics[0] = intrinsics[0]/2
    smpl_params = label_arr[25:107]
    global_orient = smpl_params[:3]
    body_pose = smpl_params[3:72]
    betas = smpl_params[72:82]
    # Change intrinsics to range [-1, 1]
    intrinsics = intrinsics.reshape(-1, 3, 3)
    # intrinsics[0, [0, 1], [0, 1]] *= 2
    fovx = 2 * np.arctan2(1.0, 2*intrinsics.cpu().numpy()[0, 0, 0])  #*15.4
    fovy = 2 * np.arctan2(1.0, 2*intrinsics.cpu().numpy()[0, 1, 1]) #*15.4
    return dict(cam2world=cam2world,
                T=T,
                intrinsics=intrinsics,
                fovy=fovy,
                fovx=fovx,
                principle_points=intrinsics[0, :2, 2],
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas
                )

def create_new_camera(parsed_labels, image_width, img_height, device):
    world_view_transform = torch.inverse(parsed_labels["cam2world"])
    shiftX = 2 * parsed_labels["principle_points"][0] - 1
    shiftY = 2 * parsed_labels["principle_points"][1] - 1
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100,
                                      fovX=parsed_labels["fovx"],
                                      fovY=parsed_labels["fovy"],
                                      shiftX=shiftX, shiftY=shiftY).to(device)

    cam = MiniCam(
        width=image_width, height=img_height,
        fovx=parsed_labels["fovx"], fovy=parsed_labels["fovy"],
        znear = 0.01,
        zfar = 100,
        world_view_transform=world_view_transform.permute(1,0).to(device),
        full_proj_transform=(proj_matrix @ world_view_transform).permute(1,0).to(device),
        # full_proj_transform=(proj_matrix @ world_view_transform).to(device),
    )
    # cam.projection_matrix = proj_matrix.permute(0,1).to(device)
    cam.projection_matrix = proj_matrix
    return cam
