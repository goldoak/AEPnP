from copy import deepcopy
import numpy as np
import cv2
import open3d as o3d

from aepnp import AEPnPSolver


def AEPnP(pts_2d, pts_3d, refine=False):
    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))

    T = AEPnPSolver(pts_2d, pts_3d)

    return T


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
