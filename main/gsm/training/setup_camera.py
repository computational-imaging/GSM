import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the number of levels to go up
levels_to_go_up = 3  # Adjust this number as needed

# Navigate up the directory structure
for _ in range(levels_to_go_up):
    current_directory = os.path.abspath(os.path.join(current_directory, '..'))

# Append the final parent directory to sys.path
sys.path.append(current_directory)

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal,focal2fov
from torch import nn

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math
import torch


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# intrinsic = torch.tensor([[2.18050930e+00,  0.00000000e+00,  5.84026622e-01],  [0.00000000e+00,
#           2.18050930e+00,  3.29450915e-01],  [0.00000000e+00,  0.00000000e+00,
#           1.00000000e+00]])

# class Camera(nn.Module):
#     def __init__(self, R, T, FoVx, FoVy,mage_ht, image_wd,
#                  trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
#                  ):
#         super(Camera, self).__init__()

#         self.R = R
#         self.T = T
#         self.FoVx = FoVx
#         self.FoVy = FoVy

#         self.data_device = data_device


#         self.image_width = mage_ht
#         self.image_height =image_wd

   

#         self.zfar = 100.0
#         self.znear = 0.01

#         self.trans = trans
#         self.scale = scale = 1

#         self.world_view_transform = getWorld2View2(R.to(self.data_device), T.to(self.data_device), trans.to(self.data_device), scale).permute(0,2,1).to(self.data_device)

#         self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, bs = R.shape[0]).permute(0,2,1)

#         self.projection_matrix[:,:3,:3] = intrinsic.float()
#         self.projection_matrix[:,2,1] = -0.05 
#         self.projection_matrix = self.projection_matrix.to(self.data_device)
#         self.full_proj_transform = (self.world_view_transform.bmm(self.projection_matrix))
#         self.camera_center = self.world_view_transform.inverse()[:, 3, :3]

class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, image_ht, image_wd,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.data_device = data_device

        # try:
        #     self.data_device = torch.device(data_device)
        # except Exception as e:
        #     print(e)
        #     print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        #     self.data_device = torch.device("cuda")

        self.image_width = image_ht
        self.image_height =image_wd

   

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale 


        self.world_view_transform = getWorld2View2(R.to(self.data_device), T.to(self.data_device), trans.to(self.data_device), scale).permute(0,2,1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, bs = R.shape[0]).permute(0,2,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.bmm(self.projection_matrix))
        self.camera_center = self.world_view_transform.inverse()[:, 3, :3]
        