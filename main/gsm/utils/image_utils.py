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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_bg_color(bg_color:str):
    """Range between 0, 1"""
    if bg_color == 'white':
        background = torch.ones((3,), dtype=torch.float32)
    elif bg_color == "gray":
        background = 0.5 * torch.ones((3,), dtype=torch.float32)
    elif bg_color == "black":
        background = torch.zeros((3,), dtype=torch.float32)
    elif bg_color == "random":
        background = torch.rand((3,), dtype=torch.float32)
    else:
        raise ValueError("Invalid Color!")
    return background