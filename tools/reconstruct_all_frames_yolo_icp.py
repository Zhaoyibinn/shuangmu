import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from tools.reconstruct_first_frame_yolo import (
    DEFAULT_ICP_TARGET_PATH,
    DEFAULT_INPUT_ROOT,
    DEFAULT_STEREO_MODEL_PATH,
    create_yolo_mask,
    depth_to_point_cloud,
    filter_and_center_cloud,
    list_images,
    load_project_config,
    load_stereo_inference,
    load_yolo,
    pca_basis,
    prepare_icp_cloud,
    proper_signed_permutation_matrices,
    read_calibration,
    resolve_workspace_path,
)


DEFAULT_OUTPUT_ROOT = (
    "depth_outputs/d455_penguan_20260525/per1/all_frames_yolo_icp"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct every stereo frame, apply YOLO segmentation and "
            "rigid no-scale ICP, and print six-DoF poses in real time."
        )
    )
    parser.add_argument("--config", default="config/pose/main.yaml")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--icp-target-path", default=DEFAULT_ICP_TARGET_PATH)
    parser.add_argument("--stereo-model-path", default=DEFAULT_STEREO_MODEL_PATH)
    parser.add_argument("--stereo-device", default=None)
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--max-depth-mm", type=float, default=1000.0)
    parser.add_argument("--yolo-model-path", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--yolo-device", default=None)
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
    parser.add_argument(
        "--icp-fallback-fitness",
        type=float,
        default=0.85,
        help="Run the 24-PCA-initialization search below this coarse fitness.",
    )
    parser.add_argument(
        "--disable-brightness-mask",
        action="store_true",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed frames and continue instead of stopping.",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Print detailed startup and per-frame stage timings.",
    )
    parser.add_argument("--ascii", action="store_true")
    return parser.parse_args()


def load_sequences(input_root):
    input_root = resolve_workspace_path(input_root)
    sequences = {
        name: list_images(input_root / name)
        for name in ("left", "right", "color")
    }
    lengths = {name: len(paths) for name, paths in sequences.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError("Sequence lengths do not match: {}".format(lengths))

    for frame_index, paths in enumerate(
        zip(sequences["left"], sequences["right"], sequences["color"])
    ):
        stems = {path.stem for path in paths}
        if len(stems) != 1:
            raise ValueError(
                "Frame names do not match at index {}: {}".format(
                    frame_index,
                    paths,
                )
            )
    return sequences


def read_frame(sequences, frame_index):
    paths = {
        name: sequence[frame_index]
        for name, sequence in sequences.items()
    }
    left = cv2.imread(str(paths["left"]), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(paths["right"]), cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(str(paths["color"]), cv2.IMREAD_COLOR)
    if left is None or right is None or color is None:
        raise ValueError("Failed to read frame {}".format(frame_index))
    if left.shape != right.shape:
        raise ValueError(
            "Left/right image shape mismatch at frame {}".format(frame_index)
        )
    return paths, left, right, color


def create_rectification_maps(calibration, image_shape):
    height, width = image_shape
    left_maps = cv2.initUndistortRectifyMap(
        calibration["m1"],
        calibration["d1"],
        calibration["r1"],
        calibration["p1"],
        (width, height),
        cv2.CV_32FC1,
    )
    right_maps = cv2.initUndistortRectifyMap(
        calibration["m2"],
        calibration["d2"],
        calibration["r2"],
        calibration["p2"],
        (width, height),
        cv2.CV_32FC1,
    )
    return left_maps, right_maps


def matrix_to_rpy_degrees(rotation):
    # Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


class TemporalRigidIcp:
    def __init__(self, target_cloud, args):
        self.args = args
        self.target_cloud = target_cloud
        self.coarse_target = prepare_icp_cloud(
            target_cloud,
            args.icp_coarse_voxel_size,
        )
        self.fine_target = prepare_icp_cloud(
            target_cloud,
            args.icp_fine_voxel_size,
        )
        target_points = np.asarray(self.coarse_target.points)
        self.target_centroid, self.target_basis = pca_basis(target_points)
        self.axis_mappings = proper_signed_permutation_matrices()
        self.previous_transform = None

    def _run_coarse(self, source, initial_transform):
        return o3d.pipelines.registration.registration_icp(
            source,
            self.coarse_target,
            self.args.icp_coarse_max_correspondence,
            init=initial_transform,
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=False
                )
            ),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.args.icp_coarse_iterations
            ),
        )

    @staticmethod
    def _is_better(candidate, best):
        if best is None:
            return True
        if candidate.fitness > best.fitness:
            return True
        return (
            np.isclose(candidate.fitness, best.fitness)
            and candidate.inlier_rmse < best.inlier_rmse
        )

    def _global_coarse_search(self, coarse_source):
        source_points = np.asarray(coarse_source.points)
        source_centroid, source_basis = pca_basis(source_points)
        best_result = None
        for axis_mapping in self.axis_mappings:
            rotation = self.target_basis @ axis_mapping @ source_basis.T
            initial_transform = np.eye(4, dtype=np.float64)
            initial_transform[:3, :3] = rotation
            initial_transform[:3, 3] = (
                self.target_centroid - rotation @ source_centroid
            )
            result = self._run_coarse(coarse_source, initial_transform)
            if self._is_better(result, best_result):
                best_result = result
        return best_result

    def register(self, source_cloud):
        timings = {}
        stage_start = time.perf_counter()
        coarse_source = prepare_icp_cloud(
            source_cloud,
            self.args.icp_coarse_voxel_size,
        )
        timings["icp_prepare_coarse_source"] = (
            time.perf_counter() - stage_start
        )

        initialization = "temporal"
        coarse_result = None
        if self.previous_transform is not None:
            stage_start = time.perf_counter()
            coarse_result = self._run_coarse(
                coarse_source,
                self.previous_transform,
            )
            timings["icp_temporal_coarse"] = (
                time.perf_counter() - stage_start
            )

        if (
            coarse_result is None
            or coarse_result.fitness < self.args.icp_fallback_fitness
        ):
            initialization = "pca24"
            stage_start = time.perf_counter()
            global_result = self._global_coarse_search(coarse_source)
            timings["icp_pca24_coarse"] = (
                time.perf_counter() - stage_start
            )
            if self._is_better(global_result, coarse_result):
                coarse_result = global_result

        stage_start = time.perf_counter()
        fine_source = prepare_icp_cloud(
            source_cloud,
            self.args.icp_fine_voxel_size,
        )
        timings["icp_prepare_fine_source"] = (
            time.perf_counter() - stage_start
        )
        stage_start = time.perf_counter()
        fine_result = o3d.pipelines.registration.registration_icp(
            fine_source,
            self.fine_target,
            self.args.icp_fine_max_correspondence,
            init=coarse_result.transformation,
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            ),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.args.icp_fine_iterations
            ),
        )
        timings["icp_fine"] = time.perf_counter() - stage_start
        self.previous_transform = np.asarray(
            fine_result.transformation,
            dtype=np.float64,
        )
        return initialization, coarse_result, fine_result, timings


def validate_args(args, frame_count):
    positive_values = {
        "--valid-iters": args.valid_iters,
        "--frame-stride": args.frame_stride,
        "--voxel-size": args.voxel_size,
        "--max-depth-mm": args.max_depth_mm,
        "--icp-coarse-voxel-size": args.icp_coarse_voxel_size,
        "--icp-coarse-max-correspondence": (
            args.icp_coarse_max_correspondence
        ),
        "--icp-coarse-iterations": args.icp_coarse_iterations,
        "--icp-fine-voxel-size": args.icp_fine_voxel_size,
        "--icp-fine-max-correspondence": args.icp_fine_max_correspondence,
        "--icp-fine-iterations": args.icp_fine_iterations,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError("{} must be greater than zero".format(name))
    if not 0 <= args.icp_fallback_fitness <= 1:
        raise ValueError("--icp-fallback-fitness must be between 0 and 1")
    if args.start_frame < 0 or args.start_frame >= frame_count:
        raise ValueError("--start-frame is out of range")
    if args.end_frame is not None:
        if args.end_frame <= args.start_frame or args.end_frame > frame_count:
            raise ValueError(
                "--end-frame must be greater than start and at most {}".format(
                    frame_count
                )
            )


def write_csv_header(csv_path):
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame_index",
            "frame_name",
            "tx_mm",
            "ty_mm",
            "tz_mm",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "fitness",
            "rmse_mm",
            "initialization",
            "point_count",
            "elapsed_seconds",
            "status",
            "error",
        ],
    )
    if new_file:
        writer.writeheader()
        csv_file.flush()
    return csv_file, writer


def main():
    args = parse_args()
    startup_start = time.perf_counter()
    startup_timings = {}

    stage_start = time.perf_counter()
    config = load_project_config(args.config)
    startup_timings["load_config"] = time.perf_counter() - stage_start
    paths_config = config["paths"]
    reconstruction_config = config["reconstruction"]
    segmentation_config = config["segmentation"]

    stage_start = time.perf_counter()
    sequences = load_sequences(args.input_root)
    startup_timings["index_sequences"] = time.perf_counter() - stage_start
    frame_count = len(sequences["left"])
    validate_args(args, frame_count)
    end_frame = args.end_frame if args.end_frame is not None else frame_count
    frame_indices = list(
        range(args.start_frame, end_frame, args.frame_stride)
    )

    first_paths, first_left, _, _ = read_frame(
        sequences,
        frame_indices[0],
    )
    height, width = first_left.shape
    stage_start = time.perf_counter()
    calibration = read_calibration(
        resolve_workspace_path(paths_config["int_yaml_path"]),
        resolve_workspace_path(paths_config["ext_yaml_path"]),
        (width, height),
    )
    left_maps, right_maps = create_rectification_maps(
        calibration,
        first_left.shape,
    )
    startup_timings["calibration_and_rectification_maps"] = (
        time.perf_counter() - stage_start
    )

    stage_start = time.perf_counter()
    stereo = load_stereo_inference(
        resolve_workspace_path(args.stereo_model_path),
        args.stereo_device,
    )
    startup_timings["load_foundation_stereo"] = (
        time.perf_counter() - stage_start
    )
    yolo_model_path = resolve_workspace_path(
        args.yolo_model_path or segmentation_config["yolo_model_path"]
    )
    stage_start = time.perf_counter()
    yolo_model = load_yolo(yolo_model_path)
    startup_timings["load_yolo"] = time.perf_counter() - stage_start
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

    from jiegouguang.color import ColorMapper

    stage_start = time.perf_counter()
    color_mapper = ColorMapper(
        str(resolve_workspace_path(paths_config["color_ext_yaml_path"]))
    )
    target_path = resolve_workspace_path(args.icp_target_path)
    target_cloud = o3d.io.read_point_cloud(str(target_path))
    if len(target_cloud.points) == 0:
        raise ValueError("ICP target cloud is empty: {}".format(target_path))
    icp = TemporalRigidIcp(target_cloud, args)
    startup_timings["load_and_prepare_icp_target"] = (
        time.perf_counter() - stage_start
    )
    startup_timings["total_startup"] = time.perf_counter() - startup_start

    output_root = resolve_workspace_path(args.output_root)
    centered_dir = output_root / "centered"
    registered_dir = output_root / "registered"
    poses_dir = output_root / "poses"
    for directory in (centered_dir, registered_dir, poses_dir):
        directory.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_root / "poses.jsonl"
    csv_path = output_root / "poses.csv"
    summary_path = output_root / "summary.json"
    jsonl_file = jsonl_path.open("a", encoding="utf-8")
    csv_file, csv_writer = write_csv_header(csv_path)

    brightness_config = reconstruction_config["brightness_mask"]
    brightness_enabled = (
        bool(brightness_config["enabled"])
        and not args.disable_brightness_mask
    )
    brightness_threshold = int(brightness_config["threshold"])
    baseline_mm = abs(float(calibration["translation"][0, 0]))
    min_depth_mm = 100.0
    max_depth_mm = float(args.max_depth_mm)

    records = []
    failed_frames = []
    sequence_start = time.perf_counter()
    print(
        "Processing {} frames from {} to {} with stride {}".format(
            len(frame_indices),
            frame_indices[0],
            frame_indices[-1],
            args.frame_stride,
        ),
        flush=True,
    )
    print(
        "Pose convention: R=Rz(yaw)*Ry(pitch)*Rx(roll), "
        "translation in mm, rotation in degrees",
        flush=True,
    )
    if args.profile_stages:
        print(
            "Startup timings: {}".format(
                " ".join(
                    "{}={:.3f}s".format(name, value)
                    for name, value in startup_timings.items()
                )
            ),
            flush=True,
        )

    try:
        for sequence_position, frame_index in enumerate(frame_indices, start=1):
            frame_start = time.perf_counter()
            frame_name = sequences["left"][frame_index].stem
            stage_timings = {}
            try:
                stage_start = time.perf_counter()
                frame_paths, left, right, color = read_frame(
                    sequences,
                    frame_index,
                )
                stage_timings["read_images"] = (
                    time.perf_counter() - stage_start
                )
                stage_start = time.perf_counter()
                left_rectified = cv2.remap(
                    left,
                    left_maps[0],
                    left_maps[1],
                    interpolation=cv2.INTER_LINEAR,
                )
                right_rectified = cv2.remap(
                    right,
                    right_maps[0],
                    right_maps[1],
                    interpolation=cv2.INTER_LINEAR,
                )
                stage_timings["rectify"] = (
                    time.perf_counter() - stage_start
                )

                stage_start = time.perf_counter()
                stereo_result = stereo.infer(
                    left_rectified,
                    right_rectified,
                    calibration["m1"],
                    baseline_mm,
                    valid_iters=args.valid_iters,
                )
                stage_timings["foundation_stereo"] = (
                    time.perf_counter() - stage_start
                )
                stage_start = time.perf_counter()
                disparity = stereo_result["disparity"]
                depth = np.zeros_like(disparity, dtype=np.float32)
                valid = np.isfinite(disparity) & (disparity > 0)
                depth[valid] = (
                    float(calibration["m1"][0, 0])
                    * baseline_mm
                    / disparity[valid]
                )
                valid &= (
                    (depth >= min_depth_mm)
                    & (depth <= max_depth_mm)
                )
                depth[~valid] = 0
                if brightness_enabled:
                    depth[left_rectified < brightness_threshold] = 0
                stage_timings["depth_filter"] = (
                    time.perf_counter() - stage_start
                )

                stage_start = time.perf_counter()
                rgbd = color_mapper.get_color(
                    depth,
                    color,
                    calibration["m1"],
                )
                color_depth = rgbd[:, :, 3]
                stage_timings["map_depth_to_color"] = (
                    time.perf_counter() - stage_start
                )
                stage_start = time.perf_counter()
                yolo_mask, instance_count = create_yolo_mask(
                    yolo_model,
                    color,
                    yolo_confidence,
                    yolo_image_size,
                    yolo_device,
                )
                stage_timings["yolo"] = (
                    time.perf_counter() - stage_start
                )
                if not np.any(yolo_mask):
                    raise ValueError("YOLO produced no segmentation mask")

                stage_start = time.perf_counter()
                segmented_depth = color_depth.copy()
                segmented_depth[~yolo_mask] = 0
                segmented_color = color.copy()
                segmented_color[~yolo_mask] = 0
                raw_cloud = depth_to_point_cloud(
                    segmented_depth,
                    segmented_color,
                    color_mapper.K_RGB,
                )
                raw_point_count = len(raw_cloud.points)
                centered_cloud, original_centroid = filter_and_center_cloud(
                    raw_cloud,
                    voxel_size=args.voxel_size,
                    statistical_neighbors=int(
                        segmentation_config["statistical_nb_neighbors"]
                    ),
                    statistical_std_ratio=float(
                        segmentation_config["statistical_std_ratio"]
                    ),
                )
                stage_timings["build_filter_center_cloud"] = (
                    time.perf_counter() - stage_start
                )

                stage_start = time.perf_counter()
                (
                    initialization,
                    coarse_result,
                    fine_result,
                    icp_timings,
                ) = icp.register(centered_cloud)
                stage_timings["icp_total"] = (
                    time.perf_counter() - stage_start
                )
                stage_timings.update(icp_timings)
                centered_to_target = np.asarray(
                    fine_result.transformation,
                    dtype=np.float64,
                )
                centering_transform = np.eye(4, dtype=np.float64)
                centering_transform[:3, 3] = -original_centroid
                original_to_target = (
                    centered_to_target @ centering_transform
                )

                stage_start = time.perf_counter()
                registered_cloud = copy.deepcopy(centered_cloud)
                registered_cloud.transform(centered_to_target)
                centered_path = centered_dir / (
                    "frame_{}_centered.ply".format(frame_name)
                )
                registered_path = registered_dir / (
                    "frame_{}_registered.ply".format(frame_name)
                )
                o3d.io.write_point_cloud(
                    str(centered_path),
                    centered_cloud,
                    write_ascii=args.ascii,
                )
                o3d.io.write_point_cloud(
                    str(registered_path),
                    registered_cloud,
                    write_ascii=args.ascii,
                )

                pose_path = poses_dir / (
                    "frame_{}_pose.json".format(frame_name)
                )
                pose_npy_path = poses_dir / (
                    "frame_{}_pose.npy".format(frame_name)
                )
                np.save(pose_npy_path, centered_to_target)

                translation = centered_to_target[:3, 3]
                roll, pitch, yaw = matrix_to_rpy_degrees(
                    centered_to_target[:3, :3]
                )
                elapsed = time.perf_counter() - frame_start
                rotation_singular_values = np.linalg.svd(
                    centered_to_target[:3, :3],
                    compute_uv=False,
                )
                record = {
                    "frame_index": frame_index,
                    "frame_name": frame_name,
                    "frame_paths": {
                        name: str(path)
                        for name, path in frame_paths.items()
                    },
                    "status": "ok",
                    "yolo_instance_count": int(instance_count),
                    "yolo_mask_pixels": int(yolo_mask.sum()),
                    "raw_point_count": int(raw_point_count),
                    "point_count": int(len(centered_cloud.points)),
                    "centroid_before_translation_mm": (
                        original_centroid.tolist()
                    ),
                    "tx_mm": float(translation[0]),
                    "ty_mm": float(translation[1]),
                    "tz_mm": float(translation[2]),
                    "roll_deg": float(roll),
                    "pitch_deg": float(pitch),
                    "yaw_deg": float(yaw),
                    "euler_convention": (
                        "R=Rz(yaw)*Ry(pitch)*Rx(roll)"
                    ),
                    "initialization": initialization,
                    "coarse_fitness": float(coarse_result.fitness),
                    "coarse_inlier_rmse_mm": float(
                        coarse_result.inlier_rmse
                    ),
                    "fine_fitness": float(fine_result.fitness),
                    "fine_inlier_rmse_mm": float(
                        fine_result.inlier_rmse
                    ),
                    "scale_estimation_enabled": False,
                    "rotation_determinant": float(
                        np.linalg.det(centered_to_target[:3, :3])
                    ),
                    "rotation_singular_values": (
                        rotation_singular_values.tolist()
                    ),
                    "transform_centered_source_to_target": (
                        centered_to_target.tolist()
                    ),
                    "transform_original_source_to_target": (
                        original_to_target.tolist()
                    ),
                    "centered_cloud_path": str(centered_path),
                    "registered_cloud_path": str(registered_path),
                    "pose_npy_path": str(pose_npy_path),
                    "elapsed_seconds": elapsed,
                }
                pose_path.write_text(
                    json.dumps(record, indent=2),
                    encoding="utf-8",
                )
                stage_timings["save_frame_files"] = (
                    time.perf_counter() - stage_start
                )
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()
                csv_writer.writerow(
                    {
                        "frame_index": frame_index,
                        "frame_name": frame_name,
                        "tx_mm": translation[0],
                        "ty_mm": translation[1],
                        "tz_mm": translation[2],
                        "roll_deg": roll,
                        "pitch_deg": pitch,
                        "yaw_deg": yaw,
                        "fitness": fine_result.fitness,
                        "rmse_mm": fine_result.inlier_rmse,
                        "initialization": initialization,
                        "point_count": len(centered_cloud.points),
                        "elapsed_seconds": elapsed,
                        "status": "ok",
                        "error": "",
                    }
                )
                csv_file.flush()
                records.append(record)

                print(
                    "[{}/{}] frame={} "
                    "T(mm)=({:+.3f},{:+.3f},{:+.3f}) "
                    "RPY(deg)=({:+.3f},{:+.3f},{:+.3f}) "
                    "fitness={:.6f} rmse={:.3f} init={} "
                    "points={} time={:.2f}s".format(
                        sequence_position,
                        len(frame_indices),
                        frame_name,
                        translation[0],
                        translation[1],
                        translation[2],
                        roll,
                        pitch,
                        yaw,
                        fine_result.fitness,
                        fine_result.inlier_rmse,
                        initialization,
                        len(centered_cloud.points),
                        elapsed,
                    ),
                    flush=True,
                )
                if args.profile_stages:
                    print(
                        "  stages: {}".format(
                            " ".join(
                                "{}={:.3f}s".format(name, value)
                                for name, value in stage_timings.items()
                            )
                        ),
                        flush=True,
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - frame_start
                error_record = {
                    "frame_index": frame_index,
                    "frame_name": frame_name,
                    "status": "error",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "elapsed_seconds": elapsed,
                }
                failed_frames.append(error_record)
                jsonl_file.write(json.dumps(error_record) + "\n")
                jsonl_file.flush()
                csv_writer.writerow(
                    {
                        "frame_index": frame_index,
                        "frame_name": frame_name,
                        "status": "error",
                        "error": error_record["error"],
                        "elapsed_seconds": elapsed,
                    }
                )
                csv_file.flush()
                print(
                    "[{}/{}] frame={} ERROR {}".format(
                        sequence_position,
                        len(frame_indices),
                        frame_name,
                        error_record["error"],
                    ),
                    flush=True,
                )
                if not args.continue_on_error:
                    raise
    finally:
        jsonl_file.close()
        csv_file.close()

    summary = {
        "input_root": str(resolve_workspace_path(args.input_root)),
        "target_path": str(target_path),
        "output_root": str(output_root),
        "frame_count_requested": len(frame_indices),
        "frame_count_succeeded": len(records),
        "frame_count_failed": len(failed_frames),
        "failed_frames": failed_frames,
        "pose_convention": "R=Rz(yaw)*Ry(pitch)*Rx(roll)",
        "translation_unit": "mm",
        "rotation_unit": "degree",
        "scale_estimation_enabled": False,
        "elapsed_seconds": time.perf_counter() - sequence_start,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        "Completed: {} succeeded, {} failed, {:.2f}s total".format(
            len(records),
            len(failed_frames),
            summary["elapsed_seconds"],
        ),
        flush=True,
    )
    print("Pose CSV: {}".format(csv_path), flush=True)
    print("Pose JSONL: {}".format(jsonl_path), flush=True)


if __name__ == "__main__":
    main()
