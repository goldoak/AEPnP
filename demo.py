import os
import json
import argparse
from copy import deepcopy
import numpy as np
import open3d as o3d
import seaborn as sns
palette = sns.color_palette("bright", 25)  # create color palette
from utils import AEPnP, compute_errors, save_viewpoint, load_viewpoint, draw_detections


id2cls = {'02691156': 'airplane', '02808440': 'bathtub', '02818832': 'bed', '02876657': 'bottle', '02954340': 'cap',
          '02958343': 'car', '03001627': 'chair', '03467517': 'guitar', '03513137': 'helmet', '03624134': 'knife',
          '03642806': 'laptop', '03790512': 'motorcycle', '03797390': 'mug', '04225987': 'skateboard',
          '04379243': 'table', '04530566': 'vessel'}
cls2id = dict((value, key) for key, value in id2cls.items())


def run_demo(viewpoint_json, save_path, cls, idx=0):
    cls_id = cls2id[cls]
    label_path = os.path.join(args.dataset_path, 'annotations', f'{cls}.json')
    labels = json.load(open(label_path))

    # decode the ground truth pose
    cam_params = o3d.io.read_pinhole_camera_parameters(viewpoint_json)
    intrinsic = cam_params.intrinsic.intrinsic_matrix
    extrinsic = cam_params.extrinsic
    width = cam_params.intrinsic.width
    height = cam_params.intrinsic.height

    # check the number of keypoints
    model_id = labels[idx]['model_id']
    textured_mesh = o3d.io.read_triangle_mesh(
        os.path.join(args.dataset_path, 'ShapeNetCore.v2.ply', f'{cls_id}/{model_id}.ply'))

    face_ids = [kp['mesh_info']['face_index'] for kp in labels[idx]['keypoints']]
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(textured_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=intrinsic,
        extrinsic_matrix=extrinsic,
        width_px=width,
        height_px=height
    )
    ans = scene.cast_rays(rays)

    n_vis_kp = 0
    kp_3d = []
    for j, face_id in enumerate(face_ids):
        if face_id in ans['primitive_ids'].numpy():
            kp_3d.append(labels[idx]['keypoints'][j]['xyz'])
            n_vis_kp += 1

    if n_vis_kp < 5:
        print("Too less keypoints! Please change the viewpoint or the model.")
        return

    # re-scale the canonical model & display
    s1 = np.random.random(1) * 1.5 + 0.5  # [0.5, 2.0]
    s2 = np.random.random(1) * 1.5 + 0.5  # [0.5, 2.0]

    textured_mesh.compute_vertex_normals()
    vertices = np.array(textured_mesh.vertices)

    vertices[:, 1] *= s1
    vertices[:, 2] *= s2

    textured_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    faces = np.array(textured_mesh.triangles)

    size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    shift = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2

    mesh_spheres = []
    for kp in labels[idx]['keypoints']:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        face_coords = vertices[faces[kp['mesh_info']['face_index']]]
        mesh_sphere.translate(face_coords.T @ kp['mesh_info']['face_uv'])
        mesh_sphere.paint_uniform_color(palette[kp['semantic_id']])
        mesh_spheres.append(mesh_sphere)

    print('visualizing re-scaled ply mesh with keypoints highlighted')
    model_img = os.path.join(save_path, f'{cls}.png')
    load_viewpoint([textured_mesh, *mesh_spheres], viewpoint_json, model_img)

    # use AEPnP solver to solve for pose and scale factors
    gt_T = deepcopy(cam_params.extrinsic)
    kp_3d = np.vstack(kp_3d)
    p3d_homo = np.concatenate([kp_3d, np.ones((kp_3d.shape[0], 1))], axis=1)
    cam_pts = np.dot(gt_T, p3d_homo.T)
    p2d = np.dot(intrinsic, cam_pts[:3, :])
    p2d /= p2d[2, :]
    kp_2d = p2d[:2, :].T

    kp_2d += np.random.normal(0, 1, (kp_2d.shape[0], 2))  # add some noise to 2D coordinates

    normalized_p2d = kp_2d  # get normalized image coordinates
    normalized_p2d[:, 0] = (normalized_p2d[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0]
    normalized_p2d[:, 1] = (normalized_p2d[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1]

    kp_3d[:, 1] *= s1  # re-scale 3D coordinates
    kp_3d[:, 2] *= s2

    gt_T[:3, 1] /= s1
    gt_T[:3, 2] /= s2

    pred_T = AEPnP(normalized_p2d, kp_3d)

    err_rot, err_trans, err_s1, err_s2 = compute_errors(pred_T, gt_T)

    print(f"Class: {cls}, err_rot: {err_rot:.3f}, err_trans: {err_trans:.3f}, err_s1: {err_s1:.3f}, err_s2: {err_s2:.3f}")

    # draw results
    pred_img = os.path.join(save_path, f'{cls}_pred.png')
    draw_detections(pred_img, intrinsic, size, pred_T, gt_T, shift=shift)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='./json', help='path to save viewpoint json file')
    parser.add_argument('--dataset_path', type=str, default='./keypoint', help='path to dataset')
    parser.add_argument('--save_path', type=str, default='vis', help='path to save images')
    parser.add_argument('--test_cls', type=str, default='car', choices=['airplane', 'bathtub', 'bed', 'bottle', 
                                                                        'cap', 'car', 'chair', 'guitar', 
                                                                        'helmet', 'knife', 'laptop', 'motorcycle', 
                                                                        'mug', 'skateboard', 'table', 'vessel'], help='class name for testing')
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        os.makedirs(args.json_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # generate viewpoint json file
    viewpoint_path = os.path.join(args.json_path, f'{args.test_cls}.json')
    cls_id = cls2id[args.test_cls]
    label_path = os.path.join(args.dataset_path, 'annotations', f'{args.test_cls}.json')
    labels = json.load(open(label_path))
    rand_idx = np.random.randint(len(labels))
    model_id = labels[rand_idx]['model_id']
    mesh = o3d.io.read_triangle_mesh(os.path.join(args.dataset_path, 'ShapeNetCore.v2.ply', f'{cls_id}/{model_id}.ply'))

    img_path = os.path.join(args.save_path, f'{args.test_cls}_pred.png')
    save_viewpoint(mesh, viewpoint_path, img_path)  # press 'q' to save the viewpoint

    run_demo(viewpoint_path, args.save_path, args.test_cls, idx=rand_idx)
