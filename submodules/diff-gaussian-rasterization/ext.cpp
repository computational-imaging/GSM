/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Rasterizefwd(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means3D));
    return RasterizeGaussiansCUDA(background,means3D,  colors,
    opacity,
	scales,
	rotations,
 scale_modifier,
	cov3D_precomp,
	viewmatrix,
	projmatrix,
	 tan_fovx, 
	tan_fovy,
    image_height,
    image_width,
 sh,
degree,
	campos,
	prefiltered,
	debug );
  }
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 Rasterizeback(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(means3D));
  return RasterizeGaussiansBackwardCUDA(background,
	 means3D,
	 radii,
     colors,
	 scales,
	 rotations,
 scale_modifier,
	cov3D_precomp,
	viewmatrix,
    projmatrix,
 tan_fovx,
 tan_fovy,
  dL_dout_color,
	 sh, degree,
	campos,
	 geomBuffer, R,
binningBuffer,
	imageBuffer, debug);

  }

torch::Tensor __markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
    { const at::cuda::OptionalCUDAGuard device_guard(device_of(means3D));
      return markVisible(means3D,viewmatrix,projmatrix);
    }
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
//   m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
//   m.def("mark_visible", &markVisible);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &Rasterizefwd);
  m.def("rasterize_gaussians_backward", &Rasterizeback);
  m.def("mark_visible", &__markVisible);
}