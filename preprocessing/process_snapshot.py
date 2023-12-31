import pickle
import os
import h5py
import sys
import numpy as np
import open3d as o3d
import json
import smplx
import cv2
import tqdm
import torch


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_KRTD(camera):
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1
    R = np.eye(3)
    T = np.zeros([3])
    D = camera['camera_k']
    return K, R, T, D


def get_o3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def get_smpl(base_smpl, betas, poses, trans):
    base_smpl.betas = betas
    base_smpl.pose = poses
    base_smpl.trans = trans
    vertices = np.array(base_smpl)

    faces = base_smpl.f
    mesh = get_o3d_mesh(vertices, faces)

    return vertices, mesh


def render_smpl(mesh, img, K, R, T):
    vertices = np.array(mesh.vertices)
    rendered_img = renderer.render_multiview(vertices, K[None], R[None],
                                             T[None, None], [img])[0]
    return rendered_img


def extract_image(data_path, img_dir):
    if len(os.listdir(img_dir)) >= 200:
        return

    cap = cv2.VideoCapture(data_path)

    ret, frame = cap.read()
    i = 0

    while ret:
        cv2.imwrite(os.path.join(img_dir, '{:04d}.png'.format(i)), frame)
        ret, frame = cap.read()
        i = i + 1

    cap.release()


def extract_mask(masks, mask_dir):
    if len(os.listdir(mask_dir)) >= len(masks):
        return

    for i in tqdm.tqdm(range(len(masks))):
        mask = masks[i].astype(np.uint8)
        # erode the mask
        border = 4
        kernel = np.ones((border, border), np.uint8)
        mask = cv2.erode(mask.copy(), kernel) * 255

        cv2.imwrite(os.path.join(mask_dir, '{:04d}.png'.format(i)), mask)


def combine_mask(image_dir, mask_dir, output_dir):
    if len(os.listdir(output_dir)) >= len(os.listdir(image_dir)):
        return

    for i in tqdm.tqdm(range(len(os.listdir(image_dir)))):
        image = cv2.imread(os.path.join(image_dir, '{:04d}.png'.format(i)))
        mask = cv2.imread(os.path.join(mask_dir, '{:04d}.png'.format(i)))
        mask = mask[..., 0:1]
        mask = np.tile(mask, (1, 1, 3))
        image = image * (mask / 255)
        cv2.imwrite(os.path.join(output_dir, '{:04d}.png'.format(i)), image)


data_root = '/mnt/Storage/data/People-Snapshot/people_snapshot_public'
videos = os.listdir(data_root)
videos = ['male-1-plaza']
debug = True

model_root = os.path.join("assets")
model_type = "smpl"

output_root = "data/people_snapshot"

smpl_female = smplx.create(model_root, model_type=model_type,
                                gender="female", use_face_contour=False,
                                num_betas=10,
                                num_expression_coeffs=10,
                                ext='.npz')

smpl_male = smplx.create(model_root, model_type=model_type,
                                gender="male", use_face_contour=False,
                                num_betas=10,
                                num_expression_coeffs=10,
                                ext='.npz')


for video in videos:
    gender = video.split('-')[0]
    if gender == "female":
        smpl = smpl_female
    else:
        smpl = smpl_male

    camera_path = os.path.join(data_root, video, 'camera.pkl')
    camera = read_pickle(camera_path)
    K, R, T, D = get_KRTD(camera)
    intrinsics = K.copy()
    intrinsics[0] /= (camera["width"]*1.0-1)
    intrinsics[1] /= (camera["height"]*1.0-1)

    # process video
    video_path = os.path.join(data_root, video, video + '.mp4')
    img_dir = os.path.join(output_root, video, 'images')
    os.system('mkdir -p {}'.format(img_dir))
    extract_image(video_path, img_dir)

    # process mask
    mask_path = os.path.join(data_root, video, 'masks.hdf5')
    masks = h5py.File(mask_path)['masks']
    mask_dir = os.path.join(output_root, video, 'masks')
    os.system('mkdir -p {}'.format(mask_dir))
    extract_mask(masks, mask_dir)

    # Combine image and mask
    combine_mask(img_dir, mask_dir, img_dir)

    smpl_path = os.path.join(data_root, video, 'reconstructed_poses.hdf5')
    smpl_data = h5py.File(smpl_path)
    betas = smpl_data['betas']
    pose = smpl_data['pose']
    trans = smpl_data['trans']

    pose = pose[len(pose) - len(masks):]
    trans = trans[len(trans) - len(masks):]
    betas = betas[:]

    # vertices_dir = os.path.join(data_root, video, 'vertices')
    # os.system('mkdir -p {}'.format(vertices_dir))

    num_img = len(os.listdir(img_dir))

    results = {}
    # field of view in grad
    results['fovx'] = 2*np.arctan2(camera['width'], 2.0 * camera['camera_f'][0])
    results['fovy'] = 2*np.arctan2(camera['height'], 2.0 * camera['camera_f'][1])
    results['cam_y_up'] = False
    body_beta = torch.tensor(betas).float()
    for frame_idx in tqdm.tqdm(range(num_img)):
        body_pose = torch.tensor(pose[frame_idx:frame_idx+1]).float()
        body_tran = torch.tensor(trans[frame_idx:frame_idx+1]).float()

        output = smpl(betas=body_beta[None],
                      expression=None,
                      body_pose=body_pose[:, 1:],
                      global_orient=body_pose[:, 0:1],
                      transl=body_tran,
                      return_verts=True)

        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3, 3:] = T.reshape(3, 1)

        if debug and frame_idx % 10 == 0 and frame_idx < 100:
            # rotating camera
            output_static = smpl(betas=body_beta[None],
                          expression=None,
                          body_pose=body_pose[:, 3:],
                          # global_orient=body_pose[:, 0:1],
                          # transl=body_tran,
                          return_verts=True)
            w2c_static = w2c @ output.transform_mat[0, 0].cpu().numpy()
            w2c_static[:3, 3:] += body_tran.cpu().numpy().reshape(3, 1)
            # plot joints on extracted image
            # read extracted images
            img = cv2.imread(os.path.join(img_dir, '{:04d}.png'.format(frame_idx)))

            # output joints projection to validate pose and transl
            joints = output_static.joints.detach().cpu()[0].numpy()

            joints = np.concatenate([joints, np.ones_like(joints[:, :1])], -1)
            cam_joints = joints @ w2c_static.T
            cam_joints = cam_joints[:, :3]
            image_joints = cam_joints @ K.T
            image_joints = image_joints / image_joints[:,-1:]
            image_joints = image_joints[:, :-1]

            for joint in image_joints:
                coord = tuple(joint.astype(np.int32).tolist())
                img = cv2.circle(img, coord , 3, (255, 0, 0), 4)

            # show to debug
            cv2.imshow(os.path.join(img_dir, '{:04d}.png'.format(frame_idx)), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        extrinsics = w2c

        results[f'{frame_idx:04d}.png'] = {
            'cam2world': np.linalg.inv(extrinsics).flatten().tolist(),
            'body_pose': body_pose[0,3:].tolist(),
            'global_orient': body_pose[0,:3].tolist(),
            'betas': body_beta.flatten().tolist(),
            'transl': body_tran[0].flatten().tolist(),
            'intrinsics': intrinsics.flatten().tolist(),
        }

    with open(os.path.join(output_root, video, 'cameras.json'), 'w') as f:
        json.dump(results, f, indent=4)