import argparse
import copy
import itertools
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FOUNDATION_STEREO_DIR = WORKSPACE_DIR / "jiegouguang" / "FoundationStereo"
ULTRALYTICS_DIR = WORKSPACE_DIR / "submodule" / "ultralytics"
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

DEFAULT_INPUT_ROOT = "huojian/d455_penguan_20260525/per1"
DEFAULT_OUTPUT_PATH = (
    "depth_outputs/d455_penguan_20260525/per1/"
    "first_frame_yolo_centered.ply"
)
DEFAULT_STEREO_MODEL_PATH = (
    "jiegouguang/weights/FoundationStereo/23-51-11/model_best_bp2.pth"
)
DEFAULT_ICP_TARGET_PATH = (
    "depth_outputs/d455_penguan_20260525/video/aruco_fusion/"
    "pose_graph_registration/"
    "merged_cloud_pose_graph_sam_cut_pca_aligned_mannual.ply"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the first stereo frame, apply YOLO segmentation, "
            "voxel-downsample it, and move its centroid to the origin."
        )
    )
    parser.add_argument("--config", default="config/pose/main.yaml")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--icp-target-path", default=DEFAULT_ICP_TARGET_PATH)
    parser.add_argument(
        "--registered-output-path",
        default=None,
        help="Registered point-cloud path. Defaults to <output>_registered.ply.",
    )
    parser.add_argument(
        "--pose-output-path",
        default=None,
        help="ICP pose JSON path. Defaults to <registered-output>_pose.json.",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--stereo-model-path", default=DEFAULT_STEREO_MODEL_PATH)
    parser.add_argument("--stereo-device", default=None)
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument(
        "--max-depth-mm",
        type=float,
        default=1000.0,
        help="Discard depth values farther than this distance in mm.",
    )
    parser.add_argument("--icp-coarse-voxel-size", type=float, default=3.0)
    parser.add_argument(
        "--icp-coarse-max-correspondence",
        type=float,
        default=30.0,
    )
    parser.add_argument("--icp-coarse-iterations", type=int, default=60)
    parser.add_argument("--icp-fine-voxel-size", type=float, default=1.0)
    parser.add_argument(
        "--icp-fine-max-correspondence",
        type=float,
        default=10.0,
    )
    parser.add_argument("--icp-fine-iterations", type=int, default=100)
    parser.add_argument("--yolo-model-path", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument(
        "--disable-brightness-mask",
        action="store_true",
        help="Ignore the brightness-mask setting from the YAML configuration.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write an ASCII PLY instead of the default binary PLY.",
    )
    return parser.parse_args()


def resolve_workspace_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return WORKSPACE_DIR / path


def default_registered_output_path(output_path):
    return output_path.with_name("{}_registered.ply".format(output_path.stem))


def default_pose_output_path(registered_output_path):
    return registered_output_path.with_name(
        "{}_pose.json".format(registered_output_path.stem)
    )


def load_project_config(config_path):
    from Recon.config_loader import load_config

    return load_config(str(resolve_workspace_path(config_path)))


def list_images(directory):
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )
    if not paths:
        raise FileNotFoundError("No images found in {}".format(directory))
    return paths


def load_frame_paths(input_root, frame_index):
    input_root = resolve_workspace_path(input_root)
    sequences = {
        name: list_images(input_root / name)
        for name in ("left", "right", "color")
    }

    if frame_index < 0:
        raise ValueError("--frame-index must be non-negative")
    for name, paths in sequences.items():
        if frame_index >= len(paths):
            raise IndexError(
                "Frame index {} is out of range for {} ({} frames)".format(
                    frame_index,
                    name,
                    len(paths),
                )
            )

    selected = {name: paths[frame_index] for name, paths in sequences.items()}
    stems = {path.stem for path in selected.values()}
    if len(stems) != 1:
        raise ValueError(
            "Selected left/right/color frame names do not match: {}".format(
                selected
            )
        )
    return selected


def read_images(frame_paths):
    left = cv2.imread(str(frame_paths["left"]), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(frame_paths["right"]), cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(str(frame_paths["color"]), cv2.IMREAD_COLOR)
    if left is None or right is None or color is None:
        raise ValueError("Failed to read one or more input images")
    if left.shape != right.shape:
        raise ValueError(
            "Left and right image shapes differ: {} vs {}".format(
                left.shape,
                right.shape,
            )
        )
    return left, right, color


def read_calibration(int_yaml_path, ext_yaml_path, image_size):
    intrinsics = cv2.FileStorage(str(int_yaml_path), cv2.FILE_STORAGE_READ)
    extrinsics = cv2.FileStorage(str(ext_yaml_path), cv2.FILE_STORAGE_READ)
    try:
        m1 = intrinsics.getNode("M1").mat()
        m2 = intrinsics.getNode("M2").mat()
        d1 = intrinsics.getNode("D1").mat()
        d2 = intrinsics.getNode("D2").mat()
        rotation = extrinsics.getNode("R").mat()
        translation = extrinsics.getNode("T").mat()
    finally:
        intrinsics.release()
        extrinsics.release()

    values = (m1, m2, d1, d2, rotation, translation)
    if any(value is None for value in values):
        raise ValueError("Stereo calibration is missing required matrices")

    width, height = image_size
    new_m1, _ = cv2.getOptimalNewCameraMatrix(
        m1, d1, (width, height), 1, (width, height)
    )
    new_m2, _ = cv2.getOptimalNewCameraMatrix(
        m2, d2, (width, height), 1, (width, height)
    )
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        new_m1,
        d1,
        new_m2,
        d2,
        (width, height),
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_TANGENT_DIST,
    )
    return {
        "m1": new_m1,
        "m2": new_m2,
        "d1": d1,
        "d2": d2,
        "r1": r1,
        "r2": r2,
        "p1": p1,
        "p2": p2,
        "translation": translation,
    }


def rectify_pair(left, right, calibration):
    height, width = left.shape
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        calibration["m1"],
        calibration["d1"],
        calibration["r1"],
        calibration["p1"],
        (width, height),
        cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        calibration["m2"],
        calibration["d2"],
        calibration["r2"],
        calibration["p2"],
        (width, height),
        cv2.CV_32FC1,
    )
    left_rectified = cv2.remap(
        left, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR
    )
    right_rectified = cv2.remap(
        right, right_map_x, right_map_y, interpolation=cv2.INTER_LINEAR
    )
    return left_rectified, right_rectified


def load_stereo_inference(model_path, device):
    # FoundationStereo imports transformations in a utility module, but the
    # inference path does not use it. Keep this script runnable when that
    # optional package is absent.
    try:
        import transformations  # noqa: F401
    except ImportError:
        sys.modules["transformations"] = types.ModuleType("transformations")

    foundation_dir = str(FOUNDATION_STEREO_DIR)
    if foundation_dir not in sys.path:
        sys.path.insert(0, foundation_dir)
    from stereo_inference import StereoInference

    return StereoInference(str(model_path), device=device)


def load_yolo(model_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        ultralytics_dir = str(ULTRALYTICS_DIR)
        if ultralytics_dir not in sys.path:
            sys.path.insert(0, ultralytics_dir)
        from ultralytics import YOLO
    return YOLO(str(model_path))


def create_yolo_mask(model, color_image, confidence, image_size, device):
    predict_args = {
        "source": color_image,
        "task": "segment",
        "imgsz": image_size,
        "conf": confidence,
        "retina_masks": True,
        "save": False,
        "verbose": False,
    }
    if device is not None:
        predict_args["device"] = device

    results = model.predict(**predict_args)
    mask = np.zeros(color_image.shape[:2], dtype=bool)
    if not results or results[0].masks is None:
        return mask, 0

    result = results[0]
    for mask_index in range(len(result.masks)):
        instance_mask = result.masks.data[mask_index].cpu().numpy()
        if instance_mask.shape != mask.shape:
            instance_mask = cv2.resize(
                instance_mask.astype(np.float32),
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask |= instance_mask > 0.5
    return mask, len(result.masks)


def map_depth_to_color(depth, color_image, left_intrinsics, color_calib_path):
    from jiegouguang.color import ColorMapper

    color_mapper = ColorMapper(str(color_calib_path))
    rgbd = color_mapper.get_color(depth, color_image, left_intrinsics)
    return rgbd, color_mapper.K_RGB


def depth_to_point_cloud(depth, color_bgr, intrinsics):
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return o3d.geometry.PointCloud()

    rows, columns = np.indices(depth.shape)
    z = depth[valid].astype(np.float64)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (columns[valid] - cx) * z / fx
    y = (rows[valid] - cy) * z / fy

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        np.column_stack((x, y, z))
    )
    colors_rgb = color_bgr[valid][:, ::-1].astype(np.float64) / 255.0
    point_cloud.colors = o3d.utility.Vector3dVector(colors_rgb)
    return point_cloud


def filter_and_center_cloud(
    point_cloud,
    voxel_size,
    statistical_neighbors,
    statistical_std_ratio,
):
    if len(point_cloud.points) == 0:
        raise ValueError("YOLO-segmented point cloud is empty")

    if (
        statistical_neighbors > 0
        and len(point_cloud.points) > statistical_neighbors
    ):
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=statistical_neighbors,
            std_ratio=statistical_std_ratio,
        )
    if len(point_cloud.points) == 0:
        raise ValueError("Point cloud is empty after statistical filtering")

    point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    if len(point_cloud.points) == 0:
        raise ValueError("Point cloud is empty after voxel downsampling")

    centroid = np.asarray(point_cloud.points).mean(axis=0)
    point_cloud.translate(-centroid)
    return point_cloud, centroid


def pca_basis(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, ::-1]
    basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    basis[:, 2] /= np.linalg.norm(basis[:, 2])
    return centroid, basis


def proper_signed_permutation_matrices():
    matrices = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = permutation_matrix @ np.diag(signs)
            if np.linalg.det(candidate) > 0:
                matrices.append(candidate)
    return matrices


def prepare_icp_cloud(point_cloud, voxel_size):
    registration_cloud = copy.deepcopy(point_cloud)
    if voxel_size > 0:
        registration_cloud = registration_cloud.voxel_down_sample(
            voxel_size=voxel_size
        )
    if len(registration_cloud.points) < 3:
        raise ValueError("Too few points for ICP registration")

    normal_radius = max(voxel_size * 2.0, 2.0)
    registration_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30,
        )
    )
    return registration_cloud


def rigid_icp_registration(
    source,
    target,
    coarse_voxel_size,
    coarse_max_correspondence,
    coarse_iterations,
    fine_voxel_size,
    fine_max_correspondence,
    fine_iterations,
):
    coarse_source = prepare_icp_cloud(source, coarse_voxel_size)
    coarse_target = prepare_icp_cloud(target, coarse_voxel_size)

    source_points = np.asarray(coarse_source.points)
    target_points = np.asarray(coarse_target.points)
    source_centroid, source_basis = pca_basis(source_points)
    target_centroid, target_basis = pca_basis(target_points)

    coarse_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=coarse_iterations
    )
    coarse_estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        )
    )

    best_result = None
    for axis_mapping in proper_signed_permutation_matrices():
        rotation = target_basis @ axis_mapping @ source_basis.T
        if np.linalg.det(rotation) < 0:
            continue

        initial_transform = np.eye(4, dtype=np.float64)
        initial_transform[:3, :3] = rotation
        initial_transform[:3, 3] = (
            target_centroid - rotation @ source_centroid
        )
        result = o3d.pipelines.registration.registration_icp(
            coarse_source,
            coarse_target,
            coarse_max_correspondence,
            init=initial_transform,
            estimation_method=coarse_estimation,
            criteria=coarse_criteria,
        )
        if best_result is None:
            best_result = result
            continue
        if result.fitness > best_result.fitness:
            best_result = result
        elif (
            np.isclose(result.fitness, best_result.fitness)
            and result.inlier_rmse < best_result.inlier_rmse
        ):
            best_result = result

    if best_result is None:
        raise RuntimeError("Failed to produce an ICP initialization")

    fine_source = prepare_icp_cloud(source, fine_voxel_size)
    fine_target = prepare_icp_cloud(target, fine_voxel_size)
    fine_result = o3d.pipelines.registration.registration_icp(
        fine_source,
        fine_target,
        fine_max_correspondence,
        init=best_result.transformation,
        estimation_method=(
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        ),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=fine_iterations
        ),
    )
    return best_result, fine_result


def main():
    args = parse_args()
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be greater than zero")
    if args.valid_iters <= 0:
        raise ValueError("--valid-iters must be greater than zero")
    if args.max_depth_mm <= 0:
        raise ValueError("--max-depth-mm must be greater than zero")
    icp_positive_values = {
        "--icp-coarse-voxel-size": args.icp_coarse_voxel_size,
        "--icp-coarse-max-correspondence": (
            args.icp_coarse_max_correspondence
        ),
        "--icp-coarse-iterations": args.icp_coarse_iterations,
        "--icp-fine-voxel-size": args.icp_fine_voxel_size,
        "--icp-fine-max-correspondence": args.icp_fine_max_correspondence,
        "--icp-fine-iterations": args.icp_fine_iterations,
    }
    for parameter_name, value in icp_positive_values.items():
        if value <= 0:
            raise ValueError("{} must be greater than zero".format(parameter_name))

    config = load_project_config(args.config)
    paths_config = config["paths"]
    reconstruction_config = config["reconstruction"]
    segmentation_config = config["segmentation"]

    frame_paths = load_frame_paths(args.input_root, args.frame_index)
    left, right, color = read_images(frame_paths)
    height, width = left.shape

    calibration = read_calibration(
        resolve_workspace_path(paths_config["int_yaml_path"]),
        resolve_workspace_path(paths_config["ext_yaml_path"]),
        (width, height),
    )
    left_rectified, right_rectified = rectify_pair(
        left,
        right,
        calibration,
    )

    stereo_model_path = resolve_workspace_path(args.stereo_model_path)
    stereo = load_stereo_inference(stereo_model_path, args.stereo_device)
    stereo_result = stereo.infer(
        left_rectified,
        right_rectified,
        calibration["m1"],
        abs(float(calibration["translation"][0, 0])),
        valid_iters=args.valid_iters,
    )
    disparity = stereo_result["disparity"]

    min_depth_mm = 100.0
    max_depth_mm = float(args.max_depth_mm)
    if max_depth_mm < min_depth_mm:
        raise ValueError(
            "--max-depth-mm must be at least {:.6g}".format(min_depth_mm)
        )
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid_disparity = np.isfinite(disparity) & (disparity > 0)
    depth[valid_disparity] = (
        float(calibration["m1"][0, 0])
        * abs(float(calibration["translation"][0, 0]))
        / disparity[valid_disparity]
    )
    valid_depth = (
        valid_disparity
        & (depth >= min_depth_mm)
        & (depth <= max_depth_mm)
    )
    depth[~valid_depth] = 0

    brightness_config = reconstruction_config["brightness_mask"]
    brightness_enabled = (
        bool(brightness_config["enabled"])
        and not args.disable_brightness_mask
    )
    brightness_threshold = int(brightness_config["threshold"])
    if brightness_enabled:
        depth[left_rectified < brightness_threshold] = 0

    rgbd, color_intrinsics = map_depth_to_color(
        depth,
        color,
        calibration["m1"],
        resolve_workspace_path(paths_config["color_ext_yaml_path"]),
    )
    color_depth = rgbd[:, :, 3]

    yolo_model_path = resolve_workspace_path(
        args.yolo_model_path or segmentation_config["yolo_model_path"]
    )
    yolo_confidence = (
        args.yolo_conf
        if args.yolo_conf is not None
        else float(segmentation_config["yolo_conf"])
    )
    yolo_image_size = (
        args.yolo_imgsz
        if args.yolo_imgsz is not None
        else int(segmentation_config["yolo_imgsz"])
    )
    yolo_device = (
        args.yolo_device
        if args.yolo_device is not None
        else segmentation_config["yolo_device"]
    )

    yolo_model = load_yolo(yolo_model_path)
    yolo_mask, instance_count = create_yolo_mask(
        yolo_model,
        color,
        yolo_confidence,
        yolo_image_size,
        yolo_device,
    )
    if not np.any(yolo_mask):
        raise ValueError("YOLO did not produce any segmentation mask")

    segmented_depth = color_depth.copy()
    segmented_depth[~yolo_mask] = 0
    segmented_color = color.copy()
    segmented_color[~yolo_mask] = 0

    point_cloud = depth_to_point_cloud(
        segmented_depth,
        segmented_color,
        color_intrinsics,
    )
    raw_point_count = len(point_cloud.points)
    point_cloud, original_centroid = filter_and_center_cloud(
        point_cloud,
        voxel_size=float(args.voxel_size),
        statistical_neighbors=int(
            segmentation_config["statistical_nb_neighbors"]
        ),
        statistical_std_ratio=float(
            segmentation_config["statistical_std_ratio"]
        ),
    )

    output_path = resolve_workspace_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ok = o3d.io.write_point_cloud(
        str(output_path),
        point_cloud,
        write_ascii=args.ascii,
    )
    if not write_ok:
        raise OSError("Failed to write point cloud: {}".format(output_path))

    target_path = resolve_workspace_path(args.icp_target_path)
    if not target_path.is_file():
        raise FileNotFoundError(
            "ICP target point cloud does not exist: {}".format(target_path)
        )
    target_cloud = o3d.io.read_point_cloud(str(target_path))
    if len(target_cloud.points) == 0:
        raise ValueError("ICP target point cloud is empty: {}".format(target_path))

    coarse_icp, fine_icp = rigid_icp_registration(
        point_cloud,
        target_cloud,
        coarse_voxel_size=float(args.icp_coarse_voxel_size),
        coarse_max_correspondence=float(
            args.icp_coarse_max_correspondence
        ),
        coarse_iterations=int(args.icp_coarse_iterations),
        fine_voxel_size=float(args.icp_fine_voxel_size),
        fine_max_correspondence=float(args.icp_fine_max_correspondence),
        fine_iterations=int(args.icp_fine_iterations),
    )
    centered_to_target = np.asarray(
        fine_icp.transformation,
        dtype=np.float64,
    )
    centering_transform = np.eye(4, dtype=np.float64)
    centering_transform[:3, 3] = -original_centroid
    original_to_target = centered_to_target @ centering_transform

    registered_output_path = (
        resolve_workspace_path(args.registered_output_path)
        if args.registered_output_path is not None
        else default_registered_output_path(output_path)
    )
    pose_output_path = (
        resolve_workspace_path(args.pose_output_path)
        if args.pose_output_path is not None
        else default_pose_output_path(registered_output_path)
    )
    registered_output_path.parent.mkdir(parents=True, exist_ok=True)
    pose_output_path.parent.mkdir(parents=True, exist_ok=True)

    registered_cloud = copy.deepcopy(point_cloud)
    registered_cloud.transform(centered_to_target)
    registered_write_ok = o3d.io.write_point_cloud(
        str(registered_output_path),
        registered_cloud,
        write_ascii=args.ascii,
    )
    if not registered_write_ok:
        raise OSError(
            "Failed to write registered point cloud: {}".format(
                registered_output_path
            )
        )

    pose_npy_path = pose_output_path.with_suffix(".npy")
    np.save(pose_npy_path, centered_to_target)
    rotation = centered_to_target[:3, :3]
    rotation_singular_values = np.linalg.svd(
        rotation,
        compute_uv=False,
    )

    metadata_path = output_path.with_name(
        "{}_metadata.json".format(output_path.stem)
    )
    centered_centroid = np.asarray(point_cloud.points).mean(axis=0)
    metadata = {
        "frame_paths": {
            name: str(path) for name, path in frame_paths.items()
        },
        "output_path": str(output_path),
        "yolo_instance_count": int(instance_count),
        "yolo_mask_pixels": int(yolo_mask.sum()),
        "raw_point_count": int(raw_point_count),
        "output_point_count": int(len(point_cloud.points)),
        "voxel_size_mm": float(args.voxel_size),
        "minimum_depth_mm": min_depth_mm,
        "maximum_depth_mm": max_depth_mm,
        "brightness_mask_enabled": brightness_enabled,
        "brightness_threshold": brightness_threshold,
        "centroid_before_translation_mm": original_centroid.tolist(),
        "centroid_after_translation_mm": centered_centroid.tolist(),
        "translation_to_origin_mm": (-original_centroid).tolist(),
        "icp": {
            "target_path": str(target_path),
            "registered_output_path": str(registered_output_path),
            "pose_output_path": str(pose_output_path),
            "pose_npy_path": str(pose_npy_path),
            "coarse_voxel_size_mm": float(args.icp_coarse_voxel_size),
            "coarse_max_correspondence_mm": float(
                args.icp_coarse_max_correspondence
            ),
            "coarse_iterations": int(args.icp_coarse_iterations),
            "coarse_fitness": float(coarse_icp.fitness),
            "coarse_inlier_rmse_mm": float(coarse_icp.inlier_rmse),
            "fine_voxel_size_mm": float(args.icp_fine_voxel_size),
            "fine_max_correspondence_mm": float(
                args.icp_fine_max_correspondence
            ),
            "fine_iterations": int(args.icp_fine_iterations),
            "fine_fitness": float(fine_icp.fitness),
            "fine_inlier_rmse_mm": float(fine_icp.inlier_rmse),
            "scale_estimation_enabled": False,
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_singular_values": rotation_singular_values.tolist(),
            "transform_centered_source_to_target": (
                centered_to_target.tolist()
            ),
            "transform_original_source_to_target": (
                original_to_target.tolist()
            ),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    pose_metadata = metadata["icp"]
    pose_output_path.write_text(
        json.dumps(pose_metadata, indent=2),
        encoding="utf-8",
    )

    print("Frame: {}".format(frame_paths["left"].stem))
    print(
        "Depth range kept: {:.6g} to {:.6g} mm".format(
            min_depth_mm,
            max_depth_mm,
        )
    )
    print("YOLO instances: {}".format(instance_count))
    print("YOLO mask pixels: {}".format(int(yolo_mask.sum())))
    print("Raw segmented points: {}".format(raw_point_count))
    print(
        "Output points after filtering and {:.6g} mm voxel downsampling: {}".format(
            args.voxel_size,
            len(point_cloud.points),
        )
    )
    print(
        "Centroid before translation: {}".format(
            np.array2string(original_centroid, precision=6)
        )
    )
    print(
        "Centroid after translation: {}".format(
            np.array2string(centered_centroid, precision=6)
        )
    )
    print("Saved point cloud: {}".format(output_path))
    print(
        "ICP coarse: fitness={:.6f}, RMSE={:.6f} mm".format(
            coarse_icp.fitness,
            coarse_icp.inlier_rmse,
        )
    )
    print(
        "ICP fine: fitness={:.6f}, RMSE={:.6f} mm".format(
            fine_icp.fitness,
            fine_icp.inlier_rmse,
        )
    )
    print(
        "Rigid transform check: det(R)={:.9f}, singular values={}".format(
            np.linalg.det(rotation),
            np.array2string(rotation_singular_values, precision=9),
        )
    )
    print("Saved registered point cloud: {}".format(registered_output_path))
    print("Saved ICP pose JSON: {}".format(pose_output_path))
    print("Saved ICP pose matrix: {}".format(pose_npy_path))
    print("Saved metadata: {}".format(metadata_path))


if __name__ == "__main__":
    main()
