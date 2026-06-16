import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from Pose.Pose_est import (
    PoseEstimator,
    matrix_to_rpy_degrees,
    resolve_workspace_path,
)
from Recon.config_loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate six-DoF object poses from stereo/color sequences using "
            "YOLO segmentation and rigid ICP."
        )
    )
    parser.add_argument("--config", default="config/pose/main.yaml")
    return parser.parse_args()


def validate_config(config, frame_count=None):
    required_paths = (
        "input_root",
        "output_root",
        "icp_target_path",
        "stereo_model_path",
        "ext_yaml_path",
        "int_yaml_path",
        "color_ext_yaml_path",
    )
    for name in required_paths:
        if not config["paths"][name]:
            raise ValueError("paths.{} must be configured".format(name))

    positive_values = {
        "stereo.valid_iters": config["stereo"]["valid_iters"],
        "point_cloud.voxel_size_mm": (
            config["point_cloud"]["voxel_size_mm"]
        ),
        "point_cloud.min_depth_mm": config["point_cloud"]["min_depth_mm"],
        "point_cloud.max_depth_mm": config["point_cloud"]["max_depth_mm"],
        "segmentation.yolo_conf": config["segmentation"]["yolo_conf"],
        "segmentation.yolo_imgsz": config["segmentation"]["yolo_imgsz"],
        "segmentation.statistical_std_ratio": (
            config["segmentation"]["statistical_std_ratio"]
        ),
        "icp.coarse_voxel_size_mm": config["icp"][
            "coarse_voxel_size_mm"
        ],
        "icp.coarse_max_correspondence_mm": config["icp"][
            "coarse_max_correspondence_mm"
        ],
        "icp.coarse_iterations": config["icp"]["coarse_iterations"],
        "icp.fine_voxel_size_mm": config["icp"]["fine_voxel_size_mm"],
        "icp.fine_max_correspondence_mm": config["icp"][
            "fine_max_correspondence_mm"
        ],
        "icp.fine_iterations": config["icp"]["fine_iterations"],
    }
    for name, value in positive_values.items():
        if float(value) <= 0:
            raise ValueError("{} must be greater than zero".format(name))
    if int(config["segmentation"]["statistical_nb_neighbors"]) < 0:
        raise ValueError(
            "segmentation.statistical_nb_neighbors cannot be negative"
        )
    if not 0 < float(config["segmentation"]["yolo_conf"]) <= 1:
        raise ValueError("segmentation.yolo_conf must be in (0, 1]")
    if not 0 <= float(config["icp"]["fallback_fitness"]) <= 1:
        raise ValueError("icp.fallback_fitness must be between 0 and 1")
    if (
        float(config["point_cloud"]["max_depth_mm"])
        < float(config["point_cloud"]["min_depth_mm"])
    ):
        raise ValueError(
            "point_cloud.max_depth_mm must be greater than or equal to "
            "point_cloud.min_depth_mm"
        )

    runtime = config["runtime"]
    start_frame = int(runtime["start_frame"])
    end_frame = runtime["end_frame"]
    keyframe_interval = get_keyframe_interval(runtime)
    if keyframe_interval <= 0:
        raise ValueError(
            "runtime.keyframe_interval must be greater than zero"
        )
    if start_frame < 0:
        raise ValueError("runtime.start_frame cannot be negative")
    if end_frame is not None and int(end_frame) <= start_frame:
        raise ValueError(
            "runtime.end_frame must be greater than runtime.start_frame"
        )
    if frame_count is not None:
        if start_frame >= frame_count:
            raise ValueError("runtime.start_frame is out of range")
        if end_frame is not None and int(end_frame) > frame_count:
            raise ValueError(
                "runtime.end_frame cannot exceed {}".format(frame_count)
            )

    brightness_threshold = int(
        config["reconstruction"]["brightness_mask"]["threshold"]
    )
    if not 0 <= brightness_threshold <= 255:
        raise ValueError(
            "reconstruction.brightness_mask.threshold must be between 0 and 255"
        )


def get_keyframe_interval(runtime_config):
    return int(
        runtime_config.get(
            "keyframe_interval",
            runtime_config.get("frame_stride", 1),
        )
    )


def rotation_axis_angle(rotation):
    cos_angle = (np.trace(rotation) - 1.0) * 0.5
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64), 0.0

    sin_angle = np.sin(angle)
    if abs(sin_angle) >= 1e-6:
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
        axis /= 2.0 * sin_angle
    else:
        _, _, vh = np.linalg.svd(rotation - np.eye(3))
        axis = vh[-1].astype(np.float64)

    norm = np.linalg.norm(axis)
    if norm > 0:
        axis /= norm
    return axis, angle


class IncrementalRotationPointEstimator:
    def __init__(self, min_angle_degrees=1e-3):
        self.min_angle_radians = np.deg2rad(min_angle_degrees)
        self.lhs = np.zeros((3, 3), dtype=np.float64)
        self.rhs = np.zeros(3, dtype=np.float64)
        self.t_norm_squared = 0.0
        self.observation_count = 0

    def add(self, transform):
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        axis, angle = rotation_axis_angle(rotation)
        angle_degrees = float(np.degrees(angle))

        used_for_center = bool(angle >= self.min_angle_radians)
        current_a = np.eye(3, dtype=np.float64) - rotation
        if used_for_center:
            self.lhs += current_a.T @ current_a
            self.rhs += current_a.T @ translation
            self.t_norm_squared += float(translation @ translation)
            self.observation_count += 1

        center = None
        rank = 0
        singular_values = np.linalg.svd(
            self.lhs, compute_uv=False
        )
        if self.observation_count > 0:
            rank = int(np.linalg.matrix_rank(self.lhs))
            center = np.linalg.pinv(self.lhs) @ self.rhs

        current_residual = None
        center_rmse = None
        if center is not None:
            residual = current_a @ center - translation
            current_residual = float(np.linalg.norm(residual))
            residual_ss = (
                float(center @ self.lhs @ center)
                - 2.0 * float(center @ self.rhs)
                + self.t_norm_squared
            )
            residual_ss = max(residual_ss, 0.0)
            center_rmse = float(
                np.sqrt(residual_ss / max(3 * self.observation_count, 1))
            )

        return {
            "rotation_center_estimated": bool(
                center is not None and rank == 3
            ),
            "rotation_center_x_mm": (
                None if center is None else float(center[0])
            ),
            "rotation_center_y_mm": (
                None if center is None else float(center[1])
            ),
            "rotation_center_z_mm": (
                None if center is None else float(center[2])
            ),
            "rotation_center_observation_count": self.observation_count,
            "rotation_center_rank": rank,
            "rotation_center_singular_values": singular_values.tolist(),
            "rotation_center_rmse_mm": center_rmse,
            "rotation_center_current_residual_mm": current_residual,
            "rotation_center_constraint_used": bool(used_for_center),
            "rotation_angle_about_center_deg": angle_degrees,
            "rotation_axis_x": float(axis[0]),
            "rotation_axis_y": float(axis[1]),
            "rotation_axis_z": float(axis[2]),
        }


def open_pose_csv(csv_path: Path):
    if csv_path.exists():
        csv_path.unlink()
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "frame_index",
        "frame_name",
        "tx_mm",
        "ty_mm",
        "tz_mm",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "original_tx_mm",
        "original_ty_mm",
        "original_tz_mm",
        "original_roll_deg",
        "original_pitch_deg",
        "original_yaw_deg",
        "relative_to_first_tx_mm",
        "relative_to_first_ty_mm",
        "relative_to_first_tz_mm",
        "relative_to_first_roll_deg",
        "relative_to_first_pitch_deg",
        "relative_to_first_yaw_deg",
        "rotation_center_estimated",
        "rotation_center_x_mm",
        "rotation_center_y_mm",
        "rotation_center_z_mm",
        "rotation_center_observation_count",
        "rotation_center_rank",
        "rotation_center_rmse_mm",
        "rotation_center_current_residual_mm",
        "rotation_center_constraint_used",
        "rotation_angle_about_center_deg",
        "rotation_axis_x",
        "rotation_axis_y",
        "rotation_axis_z",
        "relative_pose_reference_frame_index",
        "relative_pose_reference_frame_name",
        "fitness",
        "rmse_mm",
        "initialization",
        "point_count",
        "elapsed_seconds",
        "status",
        "error",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    return csv_file, writer


def pose_components(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return translation, matrix_to_rpy_degrees(rotation)


def format_optional_float(value):
    return "None" if value is None else "{:+.3f}".format(value)


def save_angle_plot(records, plot_path):
    if not records:
        return None
    try:
        import os

        mpl_config_dir = plot_path.parent / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    frame_indices = [record["frame_index"] for record in records]
    angles = [
        record["rotation_angle_about_center_deg"]
        for record in records
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(frame_indices, angles, marker="o", linewidth=1.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("angle (deg)")
    ax.set_title("Rotation angle")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def save_rotation_angle_csv(records, csv_path):
    fieldnames = [
        "frame_index",
        "frame_name",
        "angle_deg",
        "rotation_center_x_mm",
        "rotation_center_y_mm",
        "rotation_center_z_mm",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "frame_index": record["frame_index"],
                    "frame_name": record["frame_name"],
                    "angle_deg": (
                        record["rotation_angle_about_center_deg"]
                    ),
                    "rotation_center_x_mm": (
                        record["rotation_center_x_mm"]
                    ),
                    "rotation_center_y_mm": (
                        record["rotation_center_y_mm"]
                    ),
                    "rotation_center_z_mm": (
                        record["rotation_center_z_mm"]
                    ),
                }
            )


def save_estimate(
    estimate,
    directories,
    write_ascii,
    relative_to_first,
    relative_reference,
    rotation_analysis,
):
    centered_path = (
        directories["centered"]
        / "frame_{}_centered.ply".format(estimate.frame_name)
    )
    registered_path = (
        directories["registered"]
        / "frame_{}_registered.ply".format(estimate.frame_name)
    )
    pose_path = (
        directories["poses"]
        / "frame_{}_pose.json".format(estimate.frame_name)
    )
    pose_npy_path = (
        directories["poses"]
        / "frame_{}_pose.npy".format(estimate.frame_name)
    )
    original_pose_npy_path = (
        directories["poses"]
        / "frame_{}_original_source_to_target.npy".format(
            estimate.frame_name
        )
    )
    relative_pose_npy_path = (
        directories["poses"]
        / "frame_{}_original_source_to_first.npy".format(
            estimate.frame_name
        )
    )

    if not o3d.io.write_point_cloud(
        str(centered_path),
        estimate.centered_cloud,
        write_ascii=write_ascii,
    ):
        raise OSError("Failed to write {}".format(centered_path))
    if not o3d.io.write_point_cloud(
        str(registered_path),
        estimate.registered_cloud,
        write_ascii=write_ascii,
    ):
        raise OSError("Failed to write {}".format(registered_path))
    np.save(pose_npy_path, estimate.centered_to_target)
    np.save(original_pose_npy_path, estimate.original_to_target)
    np.save(relative_pose_npy_path, relative_to_first)

    original_translation, original_rpy = pose_components(
        estimate.original_to_target
    )
    relative_translation, relative_rpy = pose_components(relative_to_first)

    record = estimate.to_record()
    record.update(
        {
            "centered_cloud_path": str(centered_path),
            "registered_cloud_path": str(registered_path),
            "pose_npy_path": str(pose_npy_path),
            "original_source_to_target_npy_path": str(
                original_pose_npy_path
            ),
            "original_source_to_first_npy_path": str(
                relative_pose_npy_path
            ),
            "original_tx_mm": float(original_translation[0]),
            "original_ty_mm": float(original_translation[1]),
            "original_tz_mm": float(original_translation[2]),
            "original_roll_deg": float(original_rpy[0]),
            "original_pitch_deg": float(original_rpy[1]),
            "original_yaw_deg": float(original_rpy[2]),
            "relative_pose_reference_frame_index": (
                relative_reference["frame_index"]
            ),
            "relative_pose_reference_frame_name": (
                relative_reference["frame_name"]
            ),
            "relative_pose_convention": (
                "transform_original_source_to_first maps current frame "
                "original source coordinates into the first successful "
                "frame original source coordinates"
            ),
            "relative_to_first_tx_mm": float(relative_translation[0]),
            "relative_to_first_ty_mm": float(relative_translation[1]),
            "relative_to_first_tz_mm": float(relative_translation[2]),
            "relative_to_first_roll_deg": float(relative_rpy[0]),
            "relative_to_first_pitch_deg": float(relative_rpy[1]),
            "relative_to_first_yaw_deg": float(relative_rpy[2]),
            "transform_original_source_to_first": (
                relative_to_first.tolist()
            ),
        }
    )
    record.update(rotation_analysis)
    pose_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )
    return record


def main():
    args = parse_args()
    config = load_config(resolve_workspace_path(args.config))
    validate_config(config)
    estimator = PoseEstimator(config)
    validate_config(config, len(estimator))

    runtime_config = config["runtime"]
    output_config = config["output"]
    start_frame = int(runtime_config["start_frame"])
    end_frame = runtime_config["end_frame"]
    end_frame = int(end_frame) if end_frame is not None else len(estimator)
    keyframe_interval = get_keyframe_interval(runtime_config)
    continue_on_error = bool(runtime_config["continue_on_error"])
    profile_stages = bool(runtime_config["profile_stages"])
    write_ascii = bool(output_config["write_ascii"])
    frame_indices = range(start_frame, end_frame, keyframe_interval)
    frame_count = len(frame_indices)

    output_root = resolve_workspace_path(config["paths"]["output_root"])
    directories = {
        name: output_root / name
        for name in ("centered", "registered", "poses")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_root / "poses.jsonl"
    csv_path = output_root / "poses.csv"
    angle_csv_path = output_root / "rotation_angles.csv"
    angle_plot_path = output_root / "rotation_angle.png"
    summary_path = output_root / "summary.json"
    jsonl_file = jsonl_path.open("w", encoding="utf-8")
    csv_file, csv_writer = open_pose_csv(csv_path)

    records = []
    skipped_frames = []
    failed_frames = []
    first_pose_inv = None
    relative_reference = None
    rotation_point_estimator = IncrementalRotationPointEstimator()
    sequence_start = time.perf_counter()
    print("frame, angle(deg), rotation_center(mm)", flush=True)
    try:
        for position, frame_index in enumerate(frame_indices, start=1):
            frame_start = time.perf_counter()
            frame_name = estimator.sequences["left"][frame_index].stem
            try:
                estimate = estimator.estimate_frame(frame_index)
                if estimate is None:
                    skipped_record = {
                        "frame_index": frame_index,
                        "frame_name": frame_name,
                        "status": "skipped",
                        "error": "YOLO produced no segmentation mask",
                        "elapsed_seconds": (
                            time.perf_counter() - frame_start
                        ),
                    }
                    skipped_frames.append(skipped_record)
                    jsonl_file.write(json.dumps(skipped_record) + "\n")
                    jsonl_file.flush()
                    csv_writer.writerow(skipped_record)
                    csv_file.flush()
                    print(
                        "[{}/{}] frame={} WARNING: "
                        "YOLO produced no segmentation mask; skipped".format(
                            position,
                            frame_count,
                            frame_name,
                        ),
                        flush=True,
                    )
                    continue

                if first_pose_inv is None:
                    first_pose_inv = np.linalg.inv(
                        estimate.original_to_target
                    )
                    relative_reference = {
                        "frame_index": estimate.frame_index,
                        "frame_name": estimate.frame_name,
                    }
                relative_to_first = (
                    first_pose_inv @ estimate.original_to_target
                )
                rotation_analysis = rotation_point_estimator.add(
                    relative_to_first
                )

                record = save_estimate(
                    estimate,
                    directories,
                    write_ascii=write_ascii,
                    relative_to_first=relative_to_first,
                    relative_reference=relative_reference,
                    rotation_analysis=rotation_analysis,
                )
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()
                original_translation, original_rpy = pose_components(
                    estimate.original_to_target
                )
                relative_translation, relative_rpy = pose_components(
                    relative_to_first
                )
                csv_writer.writerow(
                    {
                        "frame_index": frame_index,
                        "frame_name": estimate.frame_name,
                        "tx_mm": estimate.translation_mm[0],
                        "ty_mm": estimate.translation_mm[1],
                        "tz_mm": estimate.translation_mm[2],
                        "roll_deg": estimate.rpy_degrees[0],
                        "pitch_deg": estimate.rpy_degrees[1],
                        "yaw_deg": estimate.rpy_degrees[2],
                        "original_tx_mm": original_translation[0],
                        "original_ty_mm": original_translation[1],
                        "original_tz_mm": original_translation[2],
                        "original_roll_deg": original_rpy[0],
                        "original_pitch_deg": original_rpy[1],
                        "original_yaw_deg": original_rpy[2],
                        "relative_to_first_tx_mm": relative_translation[0],
                        "relative_to_first_ty_mm": relative_translation[1],
                        "relative_to_first_tz_mm": relative_translation[2],
                        "relative_to_first_roll_deg": relative_rpy[0],
                        "relative_to_first_pitch_deg": relative_rpy[1],
                        "relative_to_first_yaw_deg": relative_rpy[2],
                        "rotation_center_estimated": (
                            rotation_analysis[
                                "rotation_center_estimated"
                            ]
                        ),
                        "rotation_center_x_mm": (
                            rotation_analysis["rotation_center_x_mm"]
                        ),
                        "rotation_center_y_mm": (
                            rotation_analysis["rotation_center_y_mm"]
                        ),
                        "rotation_center_z_mm": (
                            rotation_analysis["rotation_center_z_mm"]
                        ),
                        "rotation_center_observation_count": (
                            rotation_analysis[
                                "rotation_center_observation_count"
                            ]
                        ),
                        "rotation_center_rank": (
                            rotation_analysis["rotation_center_rank"]
                        ),
                        "rotation_center_rmse_mm": (
                            rotation_analysis["rotation_center_rmse_mm"]
                        ),
                        "rotation_center_current_residual_mm": (
                            rotation_analysis[
                                "rotation_center_current_residual_mm"
                            ]
                        ),
                        "rotation_center_constraint_used": (
                            rotation_analysis[
                                "rotation_center_constraint_used"
                            ]
                        ),
                        "rotation_angle_about_center_deg": (
                            rotation_analysis[
                                "rotation_angle_about_center_deg"
                            ]
                        ),
                        "rotation_axis_x": (
                            rotation_analysis["rotation_axis_x"]
                        ),
                        "rotation_axis_y": (
                            rotation_analysis["rotation_axis_y"]
                        ),
                        "rotation_axis_z": (
                            rotation_analysis["rotation_axis_z"]
                        ),
                        "relative_pose_reference_frame_index": (
                            relative_reference["frame_index"]
                        ),
                        "relative_pose_reference_frame_name": (
                            relative_reference["frame_name"]
                        ),
                        "fitness": estimate.fine_fitness,
                        "rmse_mm": estimate.fine_rmse_mm,
                        "initialization": estimate.initialization,
                        "point_count": estimate.point_count,
                        "elapsed_seconds": estimate.elapsed_seconds,
                        "status": "ok",
                        "error": "",
                    }
                )
                csv_file.flush()
                records.append(record)
                print(
                    "{}, {:+.3f}, ({},{},{})".format(
                        estimate.frame_name,
                        rotation_analysis[
                            "rotation_angle_about_center_deg"
                        ],
                        format_optional_float(
                            rotation_analysis["rotation_center_x_mm"]
                        ),
                        format_optional_float(
                            rotation_analysis["rotation_center_y_mm"]
                        ),
                        format_optional_float(
                            rotation_analysis["rotation_center_z_mm"]
                        ),
                    ),
                    flush=True,
                )
                if profile_stages:
                    print(
                        "  stages: {}".format(
                            " ".join(
                                "{}={:.3f}s".format(name, value)
                                for name, value in (
                                    estimate.stage_timings.items()
                                )
                            )
                        ),
                        flush=True,
                    )
            except Exception as exc:
                error_record = {
                    "frame_index": frame_index,
                    "frame_name": frame_name,
                    "status": "error",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "elapsed_seconds": time.perf_counter() - frame_start,
                }
                failed_frames.append(error_record)
                jsonl_file.write(json.dumps(error_record) + "\n")
                jsonl_file.flush()
                csv_writer.writerow(error_record)
                csv_file.flush()
                print(
                    "[{}/{}] frame={} ERROR {}".format(
                        position,
                        frame_count,
                        frame_name,
                        error_record["error"],
                    ),
                    flush=True,
                )
                if not continue_on_error:
                    raise
    finally:
        jsonl_file.close()
        csv_file.close()

    final_rotation_analysis = None
    if records:
        last_record = records[-1]
        final_rotation_analysis = {
            "rotation_center_estimated": (
                last_record["rotation_center_estimated"]
            ),
            "rotation_center_x_mm": (
                last_record["rotation_center_x_mm"]
            ),
            "rotation_center_y_mm": (
                last_record["rotation_center_y_mm"]
            ),
            "rotation_center_z_mm": (
                last_record["rotation_center_z_mm"]
            ),
            "rotation_center_observation_count": (
                last_record["rotation_center_observation_count"]
            ),
            "rotation_center_rank": (
                last_record["rotation_center_rank"]
            ),
            "rotation_center_singular_values": (
                last_record["rotation_center_singular_values"]
            ),
            "rotation_center_rmse_mm": (
                last_record["rotation_center_rmse_mm"]
            ),
        }

    save_rotation_angle_csv(records, angle_csv_path)
    angle_plot_result = save_angle_plot(records, angle_plot_path)

    summary = {
        "input_root": str(estimator.input_root),
        "target_path": str(estimator.target_path),
        "output_root": str(output_root),
        "frame_count_requested": frame_count,
        "frame_count_succeeded": len(records),
        "frame_count_skipped": len(skipped_frames),
        "frame_count_failed": len(failed_frames),
        "keyframe_interval": keyframe_interval,
        "skipped_frames": skipped_frames,
        "failed_frames": failed_frames,
        "pose_convention": "R=Rz(yaw)*Ry(pitch)*Rx(roll)",
        "relative_pose_reference": relative_reference,
        "relative_pose_convention": (
            "transform_original_source_to_first = "
            "inverse(first transform_original_source_to_target) @ "
            "current transform_original_source_to_target"
        ),
        "translation_unit": "mm",
        "rotation_unit": "degree",
        "rotation_angle_csv": str(angle_csv_path),
        "rotation_angle_plot": (
            None if angle_plot_result is None else str(angle_plot_result)
        ),
        "rotation_center_estimation": final_rotation_analysis,
        "rotation_center_convention": (
            "Incremental least-squares fixed point c from "
            "(I - R) c = t using each transform_original_source_to_first. "
            "rank=3 means the current sequence constrains a unique 3D point; "
            "rank<3 means the motion is under-constrained, for example a "
            "single fixed rotation axis."
        ),
        "scale_estimation_enabled": False,
        "elapsed_seconds": time.perf_counter() - sequence_start,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("angle_csv: {}".format(angle_csv_path), flush=True)
    if angle_plot_result is None:
        print("angle_plot: matplotlib unavailable", flush=True)
    else:
        print("angle_plot: {}".format(angle_plot_result), flush=True)


if __name__ == "__main__":
    main()
