import csv
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml


workspace_dir = Path(__file__).resolve().parents[1]
video_dir = workspace_dir / "huojian" / "d415_aurco" / "d455_0417"
color_dir = video_dir / "color"
depth_dir = workspace_dir / "depth_outputs" / "d455_aruco" / "depth"
rgb_calib_path = workspace_dir / "jiegouguang" / "color" / "rgb_calib_zyb_20250121.yaml"

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

marker_length = 148.0
half_marker_length = marker_length / 2.0
marker_object_points = np.array([
    [-half_marker_length, half_marker_length, 0.0],
    [half_marker_length, half_marker_length, 0.0],
    [half_marker_length, -half_marker_length, 0.0],
    [-half_marker_length, -half_marker_length, 0.0],
], dtype=np.float32)

pixel_stride = 4
max_depth_mm = 2000.0
voxel_size_mm = 5.0

output_dir = depth_dir.parent / "aruco_fusion"
pose_csv_path = output_dir / "camera_poses.csv"
cloud_path = output_dir / "merged_cloud_marker_frame.ply"
pose_vis_dir = output_dir / "pose_vis"


def load_rgb_intrinsics(calib_path):
    with calib_path.open("r", encoding="utf-8") as file:
        calib = yaml.safe_load(file)

    intrinsics = calib["rgb_intrinsics"]
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    image_size = (int(intrinsics["width"]), int(intrinsics["height"]))
    return camera_matrix, dist_coeffs, image_size


def load_depth_paths(depth_root):
    depth_paths = sorted(depth_root.glob("depth_*.png"))
    if not depth_paths:
        raise FileNotFoundError(f"no depth images found under {depth_root}")
    return depth_paths


def load_color_paths(color_root):
    color_paths = sorted(color_root.glob("frame_*.jpg"))
    if not color_paths:
        color_paths = sorted(color_root.glob("frame_*.png"))
    if not color_paths:
        raise FileNotFoundError(f"no RGB images found under {color_root}")
    return color_paths


def choose_marker(corners, ids):
    if ids is None or len(ids) == 0:
        return None, None

    areas = []
    for corner in corners:
        areas.append(abs(cv2.contourArea(np.asarray(corner[0], dtype=np.float32))))
    best_idx = int(np.argmax(areas))
    return corners[best_idx][0], int(ids[best_idx][0])


def estimate_camera_pose(frame, camera_matrix, dist_coeffs):
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    marker_corners, marker_id = choose_marker(corners, ids)
    if marker_corners is None:
        return None

    success, rvec_obj_to_cam, tvec_obj_to_cam = cv2.solvePnP(
        objectPoints=marker_object_points,
        imagePoints=np.asarray(marker_corners, dtype=np.float32),
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None

    rotation_obj_to_cam, _ = cv2.Rodrigues(rvec_obj_to_cam)

    rotation_cam_to_obj = rotation_obj_to_cam.T
    translation_cam_to_obj = -rotation_cam_to_obj @ tvec_obj_to_cam
    rvec_cam_to_obj, _ = cv2.Rodrigues(rotation_cam_to_obj)

    return {
        "corners": corners,
        "ids": ids,
        "marker_corners": marker_corners,
        "marker_id": marker_id,
        "rvec_obj_to_cam": rvec_obj_to_cam,
        "tvec_obj_to_cam": tvec_obj_to_cam,
        "rotation_cam_to_obj": rotation_cam_to_obj,
        "translation_cam_to_obj": translation_cam_to_obj,
        "rvec_cam_to_obj": rvec_cam_to_obj,
    }


def draw_pose_visualization(frame, pose, camera_matrix, dist_coeffs, frame_idx):
    vis_frame = frame.copy()
    if pose is None:
        cv2.putText(
            vis_frame,
            f"Frame {frame_idx}: no marker",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return vis_frame

    cv2.aruco.drawDetectedMarkers(vis_frame, pose["corners"], pose["ids"])
    cv2.drawFrameAxes(
        vis_frame,
        camera_matrix,
        dist_coeffs,
        pose["rvec_obj_to_cam"],
        pose["tvec_obj_to_cam"],
        marker_length * 0.5,
    )

    cam_t = pose["translation_cam_to_obj"].reshape(-1)
    obj_t = pose["tvec_obj_to_cam"].reshape(-1)
    text_lines = [
        f"Frame {frame_idx} Marker {pose['marker_id']}",
        f"Cam@Marker(mm): ({cam_t[0]:.1f}, {cam_t[1]:.1f}, {cam_t[2]:.1f})",
        f"Marker@Cam(mm): ({obj_t[0]:.1f}, {obj_t[1]:.1f}, {obj_t[2]:.1f})",
    ]
    for line_idx, text in enumerate(text_lines):
        cv2.putText(
            vis_frame,
            text,
            (10, 30 + 28 * line_idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return vis_frame


def depth_to_colored_points(depth, color_frame, camera_matrix):
    sampled_depth = depth[::pixel_stride, ::pixel_stride].astype(np.float64)
    sampled_color = color_frame[::pixel_stride, ::pixel_stride]

    valid_mask = np.isfinite(sampled_depth) & (sampled_depth > 0)
    valid_mask &= sampled_depth <= max_depth_mm
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    rows, cols = np.indices(sampled_depth.shape)
    u = (cols * pixel_stride)[valid_mask].astype(np.float64)
    v = (rows * pixel_stride)[valid_mask].astype(np.float64)
    z = sampled_depth[valid_mask]

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.stack((x, y, z), axis=-1)
    colors = sampled_color[valid_mask][:, ::-1].astype(np.float64) / 255.0
    return points_cam, colors


def transform_points(points_cam, rotation_cam_to_obj, translation_cam_to_obj):
    if len(points_cam) == 0:
        return points_cam
    return (rotation_cam_to_obj @ points_cam.T).T + translation_cam_to_obj.reshape(1, 3)


def build_point_cloud(points, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size_mm > 0:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size_mm)
    return point_cloud


def main():
    camera_matrix, dist_coeffs, image_size = load_rgb_intrinsics(rgb_calib_path)
    color_paths = load_color_paths(color_dir)
    depth_paths = load_depth_paths(depth_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_vis_dir.mkdir(parents=True, exist_ok=True)

    if len(color_paths) != len(depth_paths):
        raise RuntimeError(f"RGB frame count {len(color_paths)} does not match depth frame count {len(depth_paths)}")

    all_points = []
    all_colors = []

    with pose_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_idx", "depth_name", "marker_id",
                "cam_x_mm", "cam_y_mm", "cam_z_mm",
                "cam_rvec_x", "cam_rvec_y", "cam_rvec_z",
                "obj_x_mm", "obj_y_mm", "obj_z_mm",
                "obj_rvec_x", "obj_rvec_y", "obj_rvec_z",
            ],
        )
        writer.writeheader()

        for frame_idx, (color_path, depth_path) in enumerate(zip(color_paths, depth_paths)):
            if frame_idx < 60 or frame_idx > 300:
                continue
            frame = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read RGB image: {color_path}")

            if (frame.shape[1], frame.shape[0]) != image_size:
                raise RuntimeError(
                    f"RGB frame size {(frame.shape[1], frame.shape[0])} does not match calibration size {image_size}"
                )

            pose = estimate_camera_pose(frame, camera_matrix, dist_coeffs)
            vis_frame = draw_pose_visualization(frame, pose, camera_matrix, dist_coeffs, frame_idx)
            cv2.imwrite(str(pose_vis_dir / f"frame_{frame_idx:04d}.jpg"), vis_frame)
            if pose is None:
                continue

            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            if depth.shape != frame.shape[:2]:
                raise RuntimeError(f"depth shape {depth.shape} does not match RGB frame shape {frame.shape[:2]}")

            points_cam, colors = depth_to_colored_points(depth, frame, camera_matrix)
            points_obj = transform_points(
                points_cam,
                pose["rotation_cam_to_obj"],
                pose["translation_cam_to_obj"],
            )
            if len(points_obj) > 0:
                all_points.append(points_obj)
                all_colors.append(colors)

            cam_t = pose["translation_cam_to_obj"].reshape(-1)
            cam_r = pose["rvec_cam_to_obj"].reshape(-1)
            obj_t = pose["tvec_obj_to_cam"].reshape(-1)
            obj_r = pose["rvec_obj_to_cam"].reshape(-1)
            writer.writerow({
                "frame_idx": frame_idx,
                "depth_name": depth_path.name,
                "marker_id": pose["marker_id"],
                "cam_x_mm": float(cam_t[0]),
                "cam_y_mm": float(cam_t[1]),
                "cam_z_mm": float(cam_t[2]),
                "cam_rvec_x": float(cam_r[0]),
                "cam_rvec_y": float(cam_r[1]),
                "cam_rvec_z": float(cam_r[2]),
                "obj_x_mm": float(obj_t[0]),
                "obj_y_mm": float(obj_t[1]),
                "obj_z_mm": float(obj_t[2]),
                "obj_rvec_x": float(obj_r[0]),
                "obj_rvec_y": float(obj_r[1]),
                "obj_rvec_z": float(obj_r[2]),
            })

            if frame_idx % 50 == 0:
                print(f"processed frame {frame_idx}/{len(color_paths) - 1}")

    if not all_points:
        raise RuntimeError("no valid ArUco pose was found, merged point cloud was not created")

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    merged_cloud = build_point_cloud(merged_points, merged_colors)
    o3d.io.write_point_cloud(str(cloud_path), merged_cloud)

    print(f"saved camera poses to: {pose_csv_path}")
    print(f"saved merged cloud to: {cloud_path}")
    print(f"saved pose visualizations to: {pose_vis_dir}")


if __name__ == "__main__":
    main()
