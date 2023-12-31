from typing import NewType, Optional, Union, Tuple, Iterable
from dataclasses import dataclass, fields
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.renderer.mesh.rasterize_meshes import kEpsilon


Tensor = NewType('Tensor', torch.Tensor)

@dataclass
class DeformerOutputs:
    deformed_pnts: Optional[Tensor] = None
    weighted_transform_mat: Optional[Tensor] = None  # (bs, n, 4, 4)
    deformed_shell_verts: Optional[Tensor] = None  # (bs, num_shells, v, 3)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


def area_of_triangle(v0, v1, v2):
    """Area of triangle given 3 vertices.
    Args:
        v0, v1, v2: (P, 3)
    Returns:
        area: (P,)
    """
    return torch.norm(torch.cross(v1 - v0, v2 - v0), dim=-1) / 2


def is_inside_triangles(p, v0, v1, v2, minarea=1e-5):
    """Is point p inside triangle v0, v1, v2?
    Args:
        p: (N, 3)
        v0, v1, v2: (F, 3)
    Returns:
        inside: (N, F) bool
    """
    bary_coords3d = barycentric_coordinates_3d(p, v0, v1, v2)
    inside = (bary_coords3d.min(dim=-1).values >= 0) & (bary_coords3d.max(dim=-1).values <= 1)
    area = area_of_triangle(v0, v1, v2)  # face area.
    if p.shape[0] != v0.shape[0]:
        inside = inside & (area >= minarea).reshape(1, -1).expand(p.shape[0], -1)
    else:
        inside = inside & (area >= minarea)
    return inside


def point_line_segment_dist(p, v0, v1):
    """Point to line distance.
    Args:
        p: (N, 3)
        v0, v1: (F, 3)
    Returns:
        distance_squared: (N, F)
    """

    # Reshape tensors for broadcasting
    p = p[:, None, :]
    v0 = v0[None, :, :]
    v1 = v1[None, :, :]

    v1v0 = v1 - v0
    l2 = torch.sum(v1v0 * v1v0, dim=-1)

    mask = l2 <= 1e-24  # 1, F
    t = torch.sum(v1v0 * (p - v0), dim=-1) / (l2 + kEpsilon)
    tt = torch.clamp(t, 0.0, 1.0)
    p_proj = v0 + tt[..., None] * v1v0

    distance_squared = torch.sum((p - p_proj) ** 2, dim=-1)

    # Handle the case where l2 <= kEpsilon
    mask = mask.expand(p.shape[0], -1)
    distance_squared[mask] = torch.sum((p - v1) ** 2, dim=-1)[mask]

    return distance_squared


def barycentric_coordinates_3d(p, v0, v1, v2):
    """
    Args:
        p: (N, 3,)
        v0, v1, v2: (F, 3,)
    Returns:
        bary_coords: (N, F, 3) or (N, 3) if F = N
    """
    if p.shape[0] != v0.shape[0]:
        p0 = (v1 - v0)[None]
        p1 = (v2 - v0)[None]
        p2 = p[:, None] - v0[None]
    else:
        p0 = v1 - v0
        p1 = v2 - v0
        p2 = p - v0

    d00 = torch.sum(p0*p0, dim=-1)
    d01 = torch.sum(p0*p1, dim=-1)
    d11 = torch.sum(p1*p1, dim=-1)
    d20 = torch.sum(p2*p0, dim=-1)
    d21 = torch.sum(p2*p1, dim=-1)

    denom = d00 * d11 - d01 * d01 + 1e-15
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    return torch.stack([w0, w1, w2], dim=-1)


def find_closest_face_idx(points, point_normals, verts, faces):
    """Find closest face considering point normals.

    Args:
        points: (bs, n, 3)
        point_normals: (bs, n, 3)
        verts: (bs, m, 3)
        faces: (f, 3)
    Returns:
        point_to_face_idxs: (bs, n) LongTensor of face indices
    """
    assert points.shape[0] == 1 or verts.shape[0] == 1 or points.shape[0] == verts.shape[0], "batch size mismatch!"
    point_normals = torch.nn.functional.normalize(point_normals, dim=-1, p=2, eps=1e-12)
    meshes = Meshes(verts=verts, faces=faces[None].expand(verts.shape[0], -1, -1))
    face_normals = meshes.faces_normals_list()
    # face_area_list = packed_to_list(meshes.faces_areas_packed(), meshes.num_faces_per_mesh().tolist())
    point_to_face_idxs = []
    for b_id in range(points.shape[0]):
        # normal distance
        # (n, 3), (3, f) -> (n, f), range (-1, 1) larger the better
        normal_similarity = torch.einsum("nk,fk->nf", point_normals[b_id], face_normals[b_id])

        # point to face euclidean distance
        # (n, 3), (f, 3) -> (n, f), range (0, inf) smaller the better
        tris = verts[b_id][faces] # (f, 3, 3)
        t = torch.einsum("nfk,fk->nf", (tris[:, 0][None]-points[b_id][:, None]), face_normals[b_id]) # (n, f)
        p0 = points[b_id][:, None] + t[..., None] * face_normals[b_id][None]  # projection to all triangles
        point_to_face = t * t
        is_inside = torch.stack([is_inside_triangles(p0[i], tris[:, 0], tris[:, 1], tris[:, 2], minarea=1e-5) for i in range(p0.shape[0])], dim=0)  # (n, f)
        e01 = point_line_segment_dist(points[b_id], tris[:,0], tris[:,1])  # (n, f)
        e02 = point_line_segment_dist(points[b_id], tris[:,0], tris[:,2])  # (n, f)
        e03 = point_line_segment_dist(points[b_id], tris[:,1], tris[:,2])  # (n, f)
        point_to_face[~is_inside] = torch.min(torch.min(e01, e02), e03)[~is_inside]

        # point_to_face[normal_similarity < 0] = 100.0  # set inverse faces to large distance

        # find closest face
        closest_face_idx = torch.argmin(point_to_face, dim=-1) # (n,)
        point_to_face_idxs.append(closest_face_idx)

    return torch.stack(point_to_face_idxs, dim=0)


def point_mesh_barycentric_coordinates(points, verts, faces, point_normals=None, return_packed=False):
    """
    Computes points' barycentric coordinates on the closest triangle in a mesh.
    Args:
        points: (bs, n, 3)
        verts: (bs, m, 3)
        faces: (f, 3)
        return_packed: bool, if True, return packed representation
    Returns:
        bary_coords: (bs, n, 3) FloatTensor of barycentric coordinates or (bs*n, 3) if return_packed
        idxs: (bs, n) LongTensor of face indices or (bs*n,) if return_packed
    """
    assert points.device == verts.device
    pcls = Pointclouds(points)
    points = pcls.points_packed()
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    meshes = Meshes(verts=verts, faces=faces[None].expand(verts.shape[0], -1, -1))
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    if point_normals is not None:
        idxs = find_closest_face_idx(pcls.points_padded(), point_normals, verts, faces)
        # packed representation for faces
        idxs = idxs + tris_first_idx.reshape(len(meshes), 1)
        idxs = idxs.reshape(-1)
    else:
        # Do not use pytorch3d's function, since it could return 0 idx depending on max_area
        idxs = find_closest_face_idx(pcls.points_padded(), pcls.points_padded(), verts, faces)
        # packed representation for faces
        idxs = idxs + tris_first_idx.reshape(len(meshes), 1)
        idxs = idxs.reshape(-1)

    # compute barycentric coordinates (P, 3)
    bary_coords = barycentric_coordinates_3d(points, tris[idxs, 0], tris[idxs, 1], tris[idxs, 2])
    assert bary_coords.isfinite().all() and not bary_coords.isnan().any()
    # from packed to per pointcloud representation
    if not return_packed:
        bary_coords = bary_coords.reshape(len(pcls), max_points, 3)
        idxs = idxs.reshape(len(pcls), max_points) - tris_first_idx.reshape(len(meshes), 1)
        assert idxs.min() >= 0

    return bary_coords, idxs


class PointMeshCorrespondence:
    """helper class for point to mesh correspondence"""
    bary_coords = None  # tensor of (bs, n, 3) FloatTensor of barycentric coordinates (per point)
    face_idx = None  # tensor of (bs, n) LongTensor of face indices (per point)
    offset_len = None  # scalar of (bs, n) FloatTensor of offset length (per point) along normal
    _mesh_num_faces = None  # int of number of faces in the mesh
    _mesh_num_verts = None  # int of number of vertices in the mesh

    def __init__(self, points, meshes=None, verts=None, faces=None, point_normals=None):
        """
        Args:
            points: (bs, n, 3)
            verts: (bs, m, 3)
            faces: (f, 3)
            meshes: Meshes object with equal batch size as points and same mesh topology
        """
        if meshes is None:
            assert verts is not None and faces is not None
            meshes = Meshes(verts=verts, faces=faces[None].expand(verts.shape[0], -1, -1))
        assert points.shape[0] == len(meshes)
        self._mesh_num_faces = meshes.num_faces_per_mesh().tolist()[0]
        self._mesh_num_verts = meshes.num_verts_per_mesh().tolist()[0]
        assert all([mesh_len == self._mesh_num_faces for mesh_len in meshes.num_faces_per_mesh().tolist()])
        assert all([mesh_len == self._mesh_num_verts for mesh_len in meshes.num_verts_per_mesh().tolist()])
        # Compute mesh normal
        face_normals = meshes.faces_normals_padded()
        self.bary_coords, self.face_idxs = point_mesh_barycentric_coordinates(
            points,
            meshes.verts_padded(),
            meshes.faces_list()[0],
            point_normals=point_normals,
            return_packed=False,
        )
        # projection points on the triangles
        points_proj = interpolate_mesh_from_bary_coords(meshes.verts_padded(), meshes.faces_list()[0], self.bary_coords, self.face_idxs)
        point_face_normals = face_normals.gather(1, self.face_idxs[..., None].expand(-1, -1, 3))
        self.offset_len = torch.einsum("bnl,bnl->bn", points - points_proj, point_face_normals)
        assert self.offset_len.isfinite().all()
        assert not self.offset_len.isnan().any()

    def backproject(self, meshes=None, verts=None, faces=None):
        """Project to 3D space using barycentric coordinates and offset length."""
        if meshes is None:
            assert verts is not None and faces is not None
            meshes = Meshes(verts=verts, faces=faces[None].expand(verts.shape[0], -1, -1))
        if len(meshes) == 1 and self.bary_coords.shape[0] != 1:
            meshes = meshes.extend(self.bary_coords.shape[0])

        device = meshes.device
        self.offset_len = self.offset_len.to(device)
        self.bary_coords = self.bary_coords.to(device)
        self.face_idxs = self.face_idxs.to(device)

        # Repeat cached bary_coords, offset_len, and face_idxs
        bary_coords = self.bary_coords
        offset_len = self.offset_len
        face_idxs = self.face_idxs
        if self.bary_coords.shape[0] != len(meshes):
            assert len(meshes) % self.bary_coords.shape[0] == 0
            num_repeats = int(len(meshes) / self.bary_coords.shape[0])
            bary_coords = self.bary_coords.repeat(num_repeats, 1, 1)
            offset_len = self.offset_len.repeat(num_repeats, 1)
            face_idxs = self.face_idxs.repeat(num_repeats, 1)
        # assert all([mesh_len == self._mesh_num_faces for mesh_len in meshes.num_faces_per_mesh().tolist()])
        # assert all([mesh_len == self._mesh_num_verts for mesh_len in meshes.num_verts_per_mesh().tolist()])
        points = interpolate_mesh_from_bary_coords(meshes.verts_padded(), meshes.faces_list()[0], bary_coords, face_idxs)
        points = points + offset_len[..., None] * meshes.faces_normals_padded().gather(1, face_idxs[..., None].expand(-1, -1, 3))
        return points


def interpolate_mesh_from_bary_coords(verts, faces, bary_coords, face_idxs=None):
    """Interpolate points from barycentric coordinates.
    Args:
        verts: (?, bs, m, k)
        faces: (f, k)
        bary_coords: (bs, n or f, k) FloatTensor of barycentric coordinates
        face_idxs: (bs, n, 1) LongTensor of face indices corresponding to bary_coords
                    If not given, bary_coords is assumed to be ordered by faces
    Returns:
        points: (?, bs, n, 3) FloatTensor of interpolated points
    """
    if face_idxs is None:
        face_idxs = torch.arange(faces.shape[0], device=faces.device, dtype=torch.long)[None, :, None]
        face_idxs = face_idxs.expand(bary_coords.shape[0], -1, -1)
    src_batch_size, num_points = face_idxs.shape[:2]
    point_verts_idxs = torch.cat(
        [faces[idxs] for idxs in face_idxs]
    ).reshape(src_batch_size, num_points, faces.shape[-1])
    # closest vertex locations per point
    points_verts = knn_gather(verts, point_verts_idxs)
    interpolated =  torch.einsum("npk,npkl->npl", bary_coords, points_verts)
    if interpolated.shape[0] != src_batch_size:
        interpolated = interpolated.reshape(-1, src_batch_size, num_points, interpolated.shape[-1])
    return interpolated


def get_shell_verts_from_base(
    template_verts: torch.Tensor,
    template_faces: torch.Tensor,
    offset_len: Union[float, Tuple[float, float]],
    num_shells: int,
    shrunk_ref_mesh: str=None,
):
    """
    Args:
        template_verts: (bs, n, 3)
        template_faces: (f, 3)
        offset_len: (ouside, inside) a positive number
    Returns:
        vertices: (bs, num_shells, n, 3)
    """
    if not isinstance(offset_len, Iterable):
        out_offset_len = in_offset_len = offset_len
    else:
        assert len(offset_len) == 2
        out_offset_len, in_offset_len = offset_len
    assert out_offset_len > 0 and in_offset_len > 0

    batch_size = template_verts.shape[0]
    mesh = Meshes(
        verts=template_verts, faces=template_faces[None].repeat(batch_size, 1, 1)
    )
    # bs, n, 3
    vertex_normal = mesh.verts_normals_padded()
    # only for inflating
    n_inflated_shells = num_shells//2 + 1
    linscale = torch.linspace(
        out_offset_len,
        0,
        n_inflated_shells,
        device=template_verts.device,
        dtype=template_verts.dtype,
    )
    offset = linscale.reshape(1,n_inflated_shells, 1, 1) * vertex_normal[:, None]
    # deflating
    if shrunk_ref_mesh:
        verts_shrunk, _faces, _aux = load_obj(shrunk_ref_mesh, load_textures=False)
        assert verts_shrunk.shape[0] == template_verts.shape[1]
        offset_in = verts_shrunk.to(template_verts.device) - template_verts
        linscale = torch.linspace(0, 1.0, num_shells - n_inflated_shells + 1, device=template_verts.device, dtype=template_verts.dtype)[1:]
        offset_in = linscale.reshape(1, -1, 1, 1) * offset_in
    else:
        linscale = torch.linspace(0, -in_offset_len, num_shells - n_inflated_shells + 1, device=template_verts.device, dtype=template_verts.dtype)[1:]
        offset_in = linscale.reshape(1, -1, 1, 1) * vertex_normal[:, None]

    offset = torch.cat([offset, offset_in], dim=1)

    verts = template_verts[:, None] + offset
    assert verts.isfinite().all()
    return verts


def find_k_closest_verts(points, verts, k, points_normals=None, verts_normals=None, normal_weight=0.1):
    """

    Args:
        points: (bs, n, 3)
        verts: (bs, m, 3)
    Returns:
        dist: (bs, n, k) FloatTensor of squared distances
        idxs: (bs, n, k) LongTensor of vertex indices
    """
    assert points.device == verts.device
    assert points.shape[0] == 1 or verts.shape[0] == 1 or points.shape[0] == verts.shape[0], "batch size mismatch!"
    if points_normals is None:
        points_normals = torch.zeros_like(points)
    if verts_normals is None:
        verts_normals = torch.zeros_like(verts)
    assert points_normals.shape == points.shape
    assert verts_normals.shape == verts.shape
    points = torch.cat([points, normal_weight*points_normals], dim=-1)
    verts = torch.cat([verts, normal_weight*verts_normals], dim=-1)
    knn_output = knn_points(points, verts, K=k)
    return knn_output.dists, knn_output.idx


def weights_from_k_closest_verts(points, verts, k, points_normals=None, verts_normals=None, normal_weight=0.1, p=1):
    """Compute weights from k closest vertices using "Shepard's Method".
    Args:
        points: (bs, n, 3)
        verts: (bs, m, 3)
        k: int
        points_normals: (bs, n, 3)
        verts_normals: (bs, m, 3)
        normal_weight: scalar to include normal
        p: int
    Returns:
        weights: (bs, n, k) FloatTensor of weights
        point_verts_idx: (bs, n, k) LongTensor of vertex indices
    """
    dists, point_verts_idx = find_k_closest_verts(points, verts, k, points_normals, verts_normals, normal_weight)
    dists = torch.clamp(dists, min=1e-8)
    weights = torch.pow(dists, -p)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, point_verts_idx