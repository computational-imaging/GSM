from PIL import Image
import math

import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_batch, Meshes
from pytorch3d.renderer import (
            PerspectiveCameras, RasterizationSettings,
            DirectionalLights, MeshRasterizer, MeshRenderer, HardFlatShader,
            TexturesVertex,
            SoftSilhouetteShader,
)
from torch.utils.data import DataLoader

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from eg3d_dataset import ImageFolderDataset as EG3DDataset
from deformer.flame_deformer import FlameCustom, flame_config


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_cam2world_matrix(forward_vector, origin):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

class LookAtPoseSampler:

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = normalize_vecs(-camera_origins)
        forward_vectors = normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


def deform_with_MVC(cage, cage_deformed, cage_face, query, verbose=False):
    """
    cage (B,C,3)
    cage_deformed (B,C,3)
    cage_face (B,F,3) int64
    query (B,Q,3)
    """
    weights, weights_unnormed = mean_value_coordinates_3D(query, cage, cage_face, verbose=True)
    weights = weights.detach()
    deformed = torch.sum(weights.unsqueeze(-1)*cage_deformed.unsqueeze(1), dim=2)
    if verbose:
        return deformed, weights, weights_unnormed
    return deformed


def mean_value_coordinates_3D(query, vertices, faces, verbose=False):
    """
    Tao Ju et.al. MVC for 3D triangle meshes
    params:
        query    (B,P,3)
        vertices (B,N,3)
        faces    (B,F,3)
    return:
        wj       (B,P,N)
    """
    B, F, _ = faces.shape
    _, P, _ = query.shape
    _, N, _ = vertices.shape
    # u_i = p_i - x (B,P,N,3)
    uj = vertices.unsqueeze(1) - query.unsqueeze(2)
    # \|u_i\| (B,P,N,1)
    dj = torch.norm(uj, dim=-1, p=2, keepdim=True)
    uj = normalize(uj, dim=-1)
    # gather triangle B,P,F,3,3
    ui = torch.gather(uj.unsqueeze(2).expand(-1,-1,F,-1,-1),
                                   3,
                                   faces.unsqueeze(1).unsqueeze(-1).expand(-1,P,-1,-1,3))
    # li = \|u_{i+1}-u_{i-1}\| (B,P,F,3)
    li = torch.norm(ui[:,:,:,[1, 2, 0],:] - ui[:, :, :,[2, 0, 1],:], dim=-1, p=2)
    eps = 2e-5
    li = torch.where(li>=2, li-(li.detach()-(2-eps)), li)
    li = torch.where(li<=-2, li-(li.detach()+(2-eps)), li)
    # asin(x) is inf at +/-1
    # θi =  2arcsin[li/2] (B,P,F,3)
    theta_i = 2*torch.asin(li/2)
    assert(check_values(theta_i))
    # B,P,F,1
    h = torch.sum(theta_i, dim=-1, keepdim=True)/2
    # wi← sin[θi]d{i−1}d{i+1}
    # (B,P,F,3) ci ← (2sin[h]sin[h−θi])/(sin[θ_{i+1}]sin[θ_{i−1}])−1
    ci = 2*torch.sin(h)*torch.sin(h-theta_i)/(torch.sin(theta_i[:,:,:,[1, 2, 0]])*torch.sin(theta_i[:,:,:,[2, 0, 1]]))-1

    # NOTE: because of floating point ci can be slightly larger than 1, causing problem with sqrt(1-ci^2)
    # NOTE: sqrt(x)' is nan for x=0, hence use eps
    eps = 1e-5
    ci = torch.where(ci>=1, ci-(ci.detach()-(1-eps)), ci)
    ci = torch.where(ci<=-1, ci-(ci.detach()+(1-eps)), ci)
    # si← sign[det[u1,u2,u3]]sqrt(1-ci^2)
    # (B,P,F)*(B,P,F,3)

    si = torch.sign(torch.det(ui)).unsqueeze(-1)*torch.sqrt(1-ci**2)  # sqrt gradient nan for 0
    assert(check_values(si))
    # (B,P,F,3)
    di = torch.gather(dj.unsqueeze(2).squeeze(-1).expand(-1,-1,F,-1), 3,
                      faces.unsqueeze(1).expand(-1,P,-1,-1))
    assert(check_values(di))
    # if si.requires_grad:
    #     vertices.register_hook(save_grad("mvc/dv"))
    #     li.register_hook(save_grad("mvc/dli"))
    #     theta_i.register_hook(save_grad("mvc/dtheta"))
    #     ci.register_hook(save_grad("mvc/dci"))
    #     si.register_hook(save_grad("mvc/dsi"))
    #     di.register_hook(save_grad("mvc/ddi"))

    # wi← (θi −c[i+1]θ[i−1] −c[i−1]θ[i+1])/(disin[θi+1]s[i−1])
    # B,P,F,3
    # CHECK is there a 2* in the denominator
    wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*torch.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]])
    # if ∃i,|si| ≤ ε, set wi to 0. coplaner with T but outside
    # ignore coplaner outside triangle
    # alternative check
    # (B,F,3,3)
    # triangle_points = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))
    # # (B,P,F,3), (B,1,F,3) -> (B,P,F,1)
    # determinant = dot_product(triangle_points[:,:,:,0].unsqueeze(1)-query.unsqueeze(2),
    #                           torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0],
    #                                       triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1).unsqueeze(1), dim=-1, keepdim=True).detach()
    # # (B,P,F,1)
    # sqrdist = determinant*determinant / (4 * sqrNorm(torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0], triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1), keepdim=True))

    wi = torch.where(torch.any(torch.abs(si) <= 1e-5, keepdim=True, dim=-1), torch.zeros_like(wi), wi)
    # wi = torch.where(sqrdist <= 1e-5, torch.zeros_like(wi), wi)

    # if π −h < ε, x lies on t, use 2D barycentric coordinates
    # inside triangle
    inside_triangle = (math.pi-h).squeeze(-1)<1e-4
    # set all F for this P to zero
    wi = torch.where(torch.any(inside_triangle, dim=-1, keepdim=True).unsqueeze(-1), torch.zeros_like(wi), wi)
    # CHECK is it di https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf or li http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf
    wi = torch.where(inside_triangle.unsqueeze(-1).expand(-1,-1,-1,wi.shape[-1]), torch.sin(theta_i)*di[:,:,:,[2,0,1]]*di[:,:,:,[1,2,0]], wi)

    # sum over all faces face -> vertex (B,P,F*3) -> (B,P,N)
    wj = scatter_add(wi.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))

    # close to vertex (B,P,N)
    close_to_point = dj.squeeze(-1) < 1e-8
    # set all F for this P to zero
    wj = torch.where(torch.any(close_to_point, dim=-1, keepdim=True), torch.zeros_like(wj), wj)
    wj = torch.where(close_to_point, torch.ones_like(wj), wj)

    # (B,P,1)
    sumWj = torch.sum(wj, dim=-1, keepdim=True)
    sumWj = torch.where(sumWj==0, torch.ones_like(sumWj), sumWj)

    wj_normalised = wj / sumWj
    # if wj.requires_grad:
    #     saved_variables["mvc/wi"] = wi
    #     wi.register_hook(save_grad("mvc/dwi"))
    #     wj.register_hook(save_grad("mvc/dwj"))
    if verbose:
        return wj_normalised, wi
    else:
        return wj_normalised


def normalize(tensor, dim=-1):
    """normalize tensor in specified dimension"""
    return torch.nn.functional.normalize(tensor, p=2, dim=dim, eps=1e-12, out=None)


def check_values(tensor):
    """return true if tensor doesn't contain NaN or Inf"""
    return not (torch.any(torch.isnan(tensor)).item() or torch.any(torch.isinf(tensor)).item())


def scatter_add(src, idx, dim, out_size=None, fill=0.0):
    if out_size is None:
        out_size = list(src.size())
        dim_size = idx.max().item()+1
        out_size[dim] = dim_size
    return _scatter_add(src, idx, dim, out_size, fill)


class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, dim, out_size, fill=0.0):
        out = torch.full(out_size, fill, device=src.device, dtype=src.dtype)
        ctx.save_for_backward(idx)
        out.scatter_add_(dim, idx, src)
        ctx.mark_non_differentiable(idx)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, ograd):
        idx, = ctx.saved_tensors
        grad = torch.gather(ograd, ctx.dim, idx)
        return grad, None, None, None, None

_scatter_add = ScatterAdd.apply

def setup_diff_renderer(device, img_size, cameras):
    # Rasterization settings for silhouette rendering
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=img_size,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=50,
    )

    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )
    return renderer_silhouette


def setup_renderer(device, img_size, cameras):
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
                                    direction=[[0.5,1,1.5]])
        )
    )
    return mesh_renderer


def setup_cameras(device, img_size, pose=None, intrinsics=None):
    if intrinsics is None:
        focal_length = 4.2647 * 2
        intrinsics = torch.zeros((4, 4), device=device)
        intrinsics[[0, 1], [0, 1]] = -focal_length
        intrinsics[2, 3] = 1.0
        intrinsics[3, 2] = 1.0
        intrinsics = intrinsics[None]
    else:
        raise NotImplementedError

    if pose is None:
        yaw = 0.0
        pitch = 0.2
        pose = LookAtPoseSampler.sample(3.14/2 + yaw, 3.14/2 + pitch, torch.tensor([0, 0, 0.2]), radius=4.2)
    else:
        pose = pose.reshape(-1, 4, 4)

    world2cam = pose.clone()
    world2cam = world2cam.inverse()
    cameras = PerspectiveCameras(device=device, R=world2cam[:, :3, :3], T=world2cam[:, :3, 3],
                                 K=intrinsics, in_ndc=True, image_size=[[img_size, img_size]]*len(world2cam))
    return cameras


def detect_symmetry(vertices):
    """Find symmetry vertices along the x-axis
    Args:
        vertices: Tensor of shape (V, 3) containing the vertices of the mesh
    """
    # Calculate the difference between each pair of vertices
    diff = vertices[:, None, :] - vertices[None, :, :]

    # Check for symmetry across the x-axis
    symmetric_pairs = torch.argmin((torch.abs(diff[:, :, 1:])).sum(dim=-1) + (torch.abs(diff[:, :, 0] - 2 * vertices[:, 0:1])), dim=-1)

    assert symmetric_pairs.shape[0] == vertices.shape[0]
    return symmetric_pairs


def deform_cage_symmetrically(vertices, shp_d, symmetric_pairs=None):
    """Deform the vertices symmetrically indices and offset vector"""
    # Detect symmetric pairs
    if symmetric_pairs is None:
        symmetric_pairs = detect_symmetry(vertices)

    # Create a mask for vertices with positive x-values
    positive_x_mask = vertices[:, 0] >= 0

    # Apply deformation to vertices with positive x-values
    deformed_vertices = torch.where(positive_x_mask[:, None], vertices+shp_d, vertices)

    # Mirror the deformation to the other half
    deformed_vertices[symmetric_pairs[positive_x_mask]] = torch.stack([-deformed_vertices[:, 0], deformed_vertices[:, 1], deformed_vertices[:, 2]], dim=-1)[positive_x_mask]

    return deformed_vertices


def render_mesh_and_cage(mesh_renderer, cameras, meshes, cage_meshes):
    MESH_COLOR = [50/255.0, 168/255.0, 168/255.0]
    CAGE_COLOR = [242/255.0, 245/255.0, 66/255.0]
    meshes.textures = TexturesVertex(verts_features=torch.tensor([[MESH_COLOR]], dtype=torch.float32, device=meshes.device).expand(len(meshes), meshes.num_verts_per_mesh().max(), -1))
    cage_meshes.textures = TexturesVertex(verts_features=torch.tensor([[CAGE_COLOR]], dtype=torch.float32, device=cage_meshes.device).expand(len(cage_meshes), cage_meshes.num_verts_per_mesh().max(), -1))
    mesh_img = mesh_renderer(meshes, cameras=cameras)
    cage_img = mesh_renderer(cage_meshes, cameras=cameras)
    mesh_img = (mesh_img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    cage_img = (cage_img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return cage_img // 2 + mesh_img // 2


def render_meshes(mesh_renderer, cameras, meshes, lmk_2d=None):
    MESH_COLOR = [50/255.0, 168/255.0, 168/255.0]
    meshes.textures = TexturesVertex(verts_features=torch.tensor([[MESH_COLOR]], dtype=torch.float32, device=meshes.device).expand(len(meshes), meshes.num_verts_per_mesh().max(), -1))
    mesh_img = mesh_renderer(meshes, cameras=cameras)
    LMK_COLOR = [1.0, 0, 0, 1.0]
    if lmk_2d is not None:
        batch_idx = torch.arange(0, len(lmk_2d), device=meshes.device).unsqueeze(1).expand(-1, lmk_2d.shape[1]).flatten()
        mesh_img[batch_idx, lmk_2d[..., 1].long().flatten(), lmk_2d[..., 0].long().flatten()] = torch.tensor(LMK_COLOR, dtype=torch.float32, device=meshes.device)[None, None, None, :]
    mesh_img = (mesh_img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return mesh_img


def draw_landmarks(image, lmk_2d):
    """
    Args:
        img: numpy array
        lmk_2d: a list of (x, y) coordinates
    """
    LMK_COLOR = [255, 0, 0]
    # Define circle properties
    radius = 5
    color = (255, 0, 0)  # Red
    thickness = 1  # In pixels

    # Loop over the coordinates to draw circles and labels
    for index, (x, y) in enumerate(lmk_2d):
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.putText(image, str(index), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return image



def landmark_loss(lmk_2d, lmk_2d_gt):
    """Compute L2 loss for landmark projections"""
    return torch.mean(torch.sum((lmk_2d - lmk_2d_gt) ** 2, dim=-1))



if __name__ == "__main__":

    device = "cuda"
    img_size = 256
    cameras = setup_cameras(device)
    mesh_renderer = setup_renderer(device, img_size, cameras)
    diff_renderer = setup_diff_renderer(device, img_size, cameras)

    # Create Flame model
    config = flame_config
    flame_scale = 2.9
    flame_transl = torch.tensor([0, 0.04, 0.12])
    flame_server = FlameCustom(config, scale=flame_scale, transl=flame_transl)
    flame_server.to(device)
    template = flame_server.v_template + flame_server.transl
    template = template.unsqueeze(0)

    # Load cage
    cage_mesh = load_objs_as_meshes(["assets/flame_cage.obj"], load_textures=False)
    cage_mesh.scale_verts_(flame_scale)
    cage_mesh.offset_verts_(flame_transl)
    cage_verts = cage_mesh.verts_padded()
    cage_faces = cage_mesh.faces_padded()

    # Initialize cage delta
    cage_delta = torch.zeros_like(cage_verts)
    cage_delta[0, 0, 0] = 0.05

    # Mark cage symmetry vertices
    cage_symmetry_idx = detect_symmetry(cage_verts[0])

    # Deform cage symmetrically
    cage_verts_d = deform_cage_symmetrically(cage_verts[0], cage_delta[0], cage_symmetry_idx)

    template = deform_with_MVC(cage_verts, cage_verts_d[None], cage_faces, template, verbose=False)

    # Render for visualize (TODO)

    # Iterate through EG3D dataset and get the camera parameters, use the camera parameter to render silhouette
    # Rasterization settings for silhouette rendering
    dataset = EG3DDataset(path="/mnt/Storage/data/FFHQ/EG3D", use_labels=True, max_size=2000, xflip=False, img_subpath="images", mask_subpath="masks")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for phase_real_img, phase_real_mask, phase_real_c in dataloader:
        # Use label to get camera parameters
        pass

# Control the flame with cage deformation?
# Need to build a cage

# How to add symmetry constraint?