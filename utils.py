from copy import deepcopy
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import open3d as o3d

from apnp import APnPSolver


def decompose_solution(T):
    scaled_rotation = T[:3, :3]
    tvec = T[:3, 3]

    scale = np.eye(3)
    scale[1, 1] = np.linalg.norm(scaled_rotation[:, 1])
    scale[2, 2] = np.linalg.norm(scaled_rotation[:, 2])

    rotation = deepcopy(scaled_rotation)
    rotation[:, 1] /= scale[1, 1]
    rotation[:, 2] /= scale[2, 2]

    return rotation, tvec, scale


def x2T(x):
    T = np.eye(4)
    r = Rotation.from_quat(x[:4])
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = x[4:7]
    T[:3, 1] *= x[7]
    T[:3, 2] *= x[8]
    return T


def T2x(T):
    rotation, tvec, scale = decompose_solution(T)
    r = Rotation.from_matrix(rotation)
    x = np.concatenate([r.as_quat(), tvec, np.array([scale[1, 1], scale[2, 2]])])
    return x


def cost_func(x, pts_2d, pts_3d, K):
    T = x2T(x)
    cam_pts = np.dot(T, np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1).T)
    projected_pts = np.dot(K, cam_pts[:3, :])
    projected_pts /= projected_pts[2, :]
    reproj_error = np.mean(np.linalg.norm(pts_2d - projected_pts[:2, :].T, axis=1))
    return reproj_error


def non_linear_refinement(x0, pts_2d, pts_3d, K):
    res = least_squares(cost_func, x0, args=(pts_2d, pts_3d, K))
    refined_T = x2T(res.x)
    return refined_T


def APnP(pts_2d, pts_3d, K):
    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
    K = K.astype(np.float64)

    bearing_vectors = np.dot(np.linalg.inv(K), np.concatenate([pts_2d, np.ones((pts_2d.shape[0], 1))], axis=1).T)
    bearing_vectors = bearing_vectors.T
    bearing_vectors /= np.linalg.norm(bearing_vectors, axis=1, keepdims=True)

    solutions = APnPSolver(bearing_vectors, pts_3d)
    if len(solutions) == 0:
        return None

    min_error = np.inf
    min_idx = None
    for i in range(len(solutions)):
        T = solutions[i]
        rotation, tvec, scale = decompose_solution(T)
        det = np.linalg.det(rotation)
        if det < 0 or np.abs(det - 1) > 0.1:
            continue
        cam_pts = np.dot(T, np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1).T)
        projected_pts = np.dot(K, cam_pts[:3, :])
        projected_pts /= projected_pts[2, :]
        reproj_error = np.sum(np.linalg.norm(pts_2d - projected_pts[:2, :].T, axis=1))
        if reproj_error < min_error:
            min_error = reproj_error
            min_idx = i

    if min_idx is None:
        res = None
    else:
        res = solutions[min_idx]

    return res


def ransac_APnP(pts_2d, pts_3d, K, n_sample_pts=4, thresh=0.1, refine=True):
    max_iter = 1000
    reproj_error_thresh = thresh
    confidence = 0.999
    sample_size = n_sample_pts
    n_pts = pts_2d.shape[0]

    best_inlier_ratio = 0
    best_inlier_idx = np.arange(n_pts)
    best_T = None
    for i in range(max_iter):
        rand_idx = np.random.randint(n_pts, size=sample_size)
        T = APnP(pts_2d[rand_idx, :], pts_3d[rand_idx, :], K)
        if T is None:
            continue

        cam_pts = np.dot(T, np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1).T)
        projected_pts = np.dot(K, cam_pts[:3, :])
        if np.any(projected_pts[2, :] < 0):
            continue
        projected_pts /= projected_pts[2, :]
        reproj_error = np.linalg.norm(pts_2d - projected_pts[:2, :].T, axis=1)

        inlier_idx = np.where(reproj_error < reproj_error_thresh)[0]
        n_inliers = inlier_idx.shape[0]
        inlier_ratio = n_inliers / n_pts
        if inlier_ratio > best_inlier_ratio:
            best_inlier_idx = inlier_idx
            best_inlier_ratio = inlier_ratio
            best_T = T

        # early break
        if (1 - (1 - best_inlier_ratio ** 5) ** i) > confidence:
            break

    if refine is True and best_T is not None:
        try:
            x0 = T2x(best_T)
            best_T = non_linear_refinement(x0, pts_2d[best_inlier_idx, :], pts_3d[best_inlier_idx, :], K)
        except:
            print('Invalid x0!')

    return best_T


def compute_errors(T1, T2):
    try:
        assert np.array_equal(T1[3, :], T2[3, :])
        assert np.array_equal(T1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(T1[3, :], T2[3, :])

    R1 = deepcopy(T1[:3, :3])
    s11 = np.linalg.norm(R1[:, 1])
    s12 = np.linalg.norm(R1[:, 2])
    R1[:, 1] /= s11
    R1[:, 2] /= s12
    t1 = T1[:3, 3]

    R2 = deepcopy(T2[:3, :3])
    s21 = np.linalg.norm(R2[:, 1])
    s22 = np.linalg.norm(R2[:, 2])
    R2[:, 1] /= s21
    R2[:, 2] /= s22
    t2 = T2[:3, 3]

    R12 = R1 @ R2.transpose()
    R_error = np.arccos(np.clip((np.trace(R12) - 1) / 2, -1.0, 1.0)) * 180 / np.pi
    t_error = np.linalg.norm(t1 - t2)

    s1_error = np.abs(s11 - s21) / s21
    s2_error = np.abs(s12 - s22) / s22

    if np.isnan(R_error):
        R_error = 180
    if np.isnan(t_error):
        t_error = np.inf
    if np.isnan(s1_error):
        s1_error = np.inf
    if np.isnan(s2_error):
        s2_error = np.inf

    return R_error, t_error, s1_error, s2_error


def save_viewpoint(mesh, json_file, img_file):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(json_file, param)
    vis.capture_screen_image(img_file, do_render=True)
    vis.destroy_window()


def load_viewpoint(mesh, json_file, img_file):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(json_file)
    for obj in mesh:
       vis.add_geometry(obj)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.capture_screen_image(img_file, do_render=True)
    vis.destroy_window()


def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def draw_bboxes(img, img_pts, axes, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    # draw axes
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3)  ## y last

    return img


def draw_detections(img_file, intrinsics, size, pred_sRT, gt_sRT, draw_gt=True, shift=0):
    """ Visualize pose predictions.
    """
    img = cv2.imread(img_file)
    xyz_axis = 0.3 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).transpose()
    bbox_3d = get_3d_bbox(size, shift)

    # darw ground truth - GREEN color
    if draw_gt:
        transformed_axes = transform_coordinates_3d(xyz_axis, gt_sRT)
        projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

        transformed_bbox_3d = transform_coordinates_3d(bbox_3d, gt_sRT)
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        img = draw_bboxes(img, projected_bbox, projected_axes, (0, 255, 0))

    # darw prediction - RED color
    transformed_axes = transform_coordinates_3d(xyz_axis, pred_sRT)
    projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, pred_sRT)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    img = draw_bboxes(img, projected_bbox, projected_axes, (0, 0, 255))

    cv2.imwrite(img_file, img)
    cv2.imshow('visualization', img)
    cv2.waitKey(0)
