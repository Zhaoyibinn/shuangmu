import copy
import itertools
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import open3d as o3d


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FOUNDATION_STEREO_DIR = WORKSPACE_DIR / "jiegouguang" / "FoundationStereo"
ULTRALYTICS_DIR = WORKSPACE_DIR / "submodule" / "ultralytics"


@dataclass(frozen=True)
class IcpConfig:
    coarse_voxel_size: float = 3.0
    coarse_max_correspondence: float = 30.0
    coarse_iterations: int = 60
    fine_voxel_size: float = 1.0
    fine_max_correspondence: float = 10.0
    fine_iterations: int = 100
    fallback_fitness: float = 0.85

    @classmethod
    def from_config(cls, config):
        return cls(
            coarse_voxel_size=float(config["coarse_voxel_size_mm"]),
            coarse_max_correspondence=float(
                config["coarse_max_correspondence_mm"]
            ),
            coarse_iterations=int(config["coarse_iterations"]),
            fine_voxel_size=float(config["fine_voxel_size_mm"]),
            fine_max_correspondence=float(
                config["fine_max_correspondence_mm"]
            ),
            fine_iterations=int(config["fine_iterations"]),
            fallback_fitness=float(config["fallback_fitness"]),
        )

    def validate(self) -> None:
        positive_values = {
            "coarse_voxel_size": self.coarse_voxel_size,
            "coarse_max_correspondence": self.coarse_max_correspondence,
            "coarse_iterations": self.coarse_iterations,
            "fine_voxel_size": self.fine_voxel_size,
            "fine_max_correspondence": self.fine_max_correspondence,
            "fine_iterations": self.fine_iterations,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError("{} must be greater than zero".format(name))
        if not 0.0 <= self.fallback_fitness <= 1.0:
            raise ValueError("fallback_fitness must be between 0 and 1")


@dataclass
class PoseEstimate:
    frame_index: int
    frame_name: str
    frame_paths: Dict[str, Path]
    centered_cloud: o3d.geometry.PointCloud
    registered_cloud: o3d.geometry.PointCloud
    centered_to_target: np.ndarray
    original_to_target: np.ndarray
    centroid_before_translation: np.ndarray
    translation_mm: np.ndarray
    rpy_degrees: np.ndarray
    initialization: str
    coarse_fitness: float
    coarse_rmse_mm: float
    fine_fitness: float
    fine_rmse_mm: float
    yolo_instance_count: int
    yolo_mask_pixels: int
    raw_point_count: int
    elapsed_seconds: float
    stage_timings: Dict[str, float]

    @property
    def point_count(self) -> int:
        return len(self.centered_cloud.points)

    def to_record(self) -> dict:
        rotation = self.centered_to_target[:3, :3]
        return {
            "frame_index": self.frame_index,
            "frame_name": self.frame_name,
            "frame_paths": {
                name: str(path) for name, path in self.frame_paths.items()
            },
            "status": "ok",
            "yolo_instance_count": self.yolo_instance_count,
            "yolo_mask_pixels": self.yolo_mask_pixels,
            "raw_point_count": self.raw_point_count,
            "point_count": self.point_count,
            "centroid_before_translation_mm": (
                self.centroid_before_translation.tolist()
            ),
            "tx_mm": float(self.translation_mm[0]),
            "ty_mm": float(self.translation_mm[1]),
            "tz_mm": float(self.translation_mm[2]),
            "roll_deg": float(self.rpy_degrees[0]),
            "pitch_deg": float(self.rpy_degrees[1]),
            "yaw_deg": float(self.rpy_degrees[2]),
            "euler_convention": "R=Rz(yaw)*Ry(pitch)*Rx(roll)",
            "initialization": self.initialization,
            "coarse_fitness": self.coarse_fitness,
            "coarse_inlier_rmse_mm": self.coarse_rmse_mm,
            "fine_fitness": self.fine_fitness,
            "fine_inlier_rmse_mm": self.fine_rmse_mm,
            "scale_estimation_enabled": False,
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_singular_values": np.linalg.svd(
                rotation, compute_uv=False
            ).tolist(),
            "transform_centered_source_to_target": (
                self.centered_to_target.tolist()
            ),
            "transform_original_source_to_target": (
                self.original_to_target.tolist()
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "stage_timings": self.stage_timings,
        }


def resolve_workspace_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else WORKSPACE_DIR / path


def matrix_to_rpy_degrees(rotation: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    if sy >= 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def _list_images(directory: Path):
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )
    if not paths:
        raise FileNotFoundError("No images found in {}".format(directory))
    return paths


def _load_sequences(input_root) -> Dict[str, list]:
    root = resolve_workspace_path(input_root)
    sequences = {
        name: _list_images(root / name)
        for name in ("left", "right", "color")
    }
    lengths = {name: len(paths) for name, paths in sequences.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError("Sequence lengths do not match: {}".format(lengths))
    for frame_index, paths in enumerate(
        zip(sequences["left"], sequences["right"], sequences["color"])
    ):
        if len({path.stem for path in paths}) != 1:
            raise ValueError(
                "Frame names do not match at index {}: {}".format(
                    frame_index, paths
                )
            )
    return sequences


def _read_calibration(
    intrinsics_path: Path,
    extrinsics_path: Path,
    image_size: Tuple[int, int],
) -> dict:
    intrinsics = cv2.FileStorage(
        str(intrinsics_path), cv2.FILE_STORAGE_READ
    )
    extrinsics = cv2.FileStorage(
        str(extrinsics_path), cv2.FILE_STORAGE_READ
    )
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

    if any(
        value is None
        for value in (m1, m2, d1, d2, rotation, translation)
    ):
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


def _create_rectification_maps(calibration: dict, image_shape):
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


def _load_stereo_inference(model_path: Path, device):
    try:
        import transformations  # noqa: F401
    except ImportError:
        sys.modules["transformations"] = types.ModuleType("transformations")

    foundation_dir = str(FOUNDATION_STEREO_DIR)
    if foundation_dir not in sys.path:
        sys.path.insert(0, foundation_dir)
    from stereo_inference import StereoInference

    return StereoInference(str(model_path), device=device)


def _load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        ultralytics_dir = str(ULTRALYTICS_DIR)
        if ultralytics_dir not in sys.path:
            sys.path.insert(0, ultralytics_dir)
        from ultralytics import YOLO
    return YOLO(str(model_path))


def _create_yolo_mask(model, color_image, confidence, image_size, device):
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
    for instance in results[0].masks.data:
        instance_mask = instance.cpu().numpy()
        if instance_mask.shape != mask.shape:
            instance_mask = cv2.resize(
                instance_mask.astype(np.float32),
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask |= instance_mask > 0.5
    return mask, len(results[0].masks)


def _depth_to_point_cloud(depth, color_bgr, intrinsics):
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

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))
    colors_rgb = color_bgr[valid][:, ::-1].astype(np.float64) / 255.0
    cloud.colors = o3d.utility.Vector3dVector(colors_rgb)
    return cloud


def _filter_and_center_cloud(
    cloud,
    voxel_size,
    statistical_neighbors,
    statistical_std_ratio,
):
    if len(cloud.points) == 0:
        raise ValueError("YOLO-segmented point cloud is empty")
    if statistical_neighbors > 0 and len(cloud.points) > statistical_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=statistical_neighbors,
            std_ratio=statistical_std_ratio,
        )
    if len(cloud.points) == 0:
        raise ValueError("Point cloud is empty after statistical filtering")
    cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
    if len(cloud.points) == 0:
        raise ValueError("Point cloud is empty after voxel downsampling")

    centroid = np.asarray(cloud.points).mean(axis=0)
    cloud.translate(-centroid)
    return cloud, centroid


def _pca_basis(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, ::-1]
    basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    basis[:, 2] /= np.linalg.norm(basis[:, 2])
    return centroid, basis


def _proper_signed_permutation_matrices():
    matrices = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = permutation_matrix @ np.diag(signs)
            if np.linalg.det(candidate) > 0:
                matrices.append(candidate)
    return matrices


def _prepare_icp_cloud(cloud, voxel_size):
    registration_cloud = copy.deepcopy(cloud)
    registration_cloud = registration_cloud.voxel_down_sample(
        voxel_size=voxel_size
    )
    if len(registration_cloud.points) < 3:
        raise ValueError("Too few points for ICP registration")
    registration_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(voxel_size * 2.0, 2.0),
            max_nn=30,
        )
    )
    return registration_cloud


class _TemporalRigidIcp:
    def __init__(self, target_cloud, config: IcpConfig):
        self.config = config
        self.coarse_target = _prepare_icp_cloud(
            target_cloud, config.coarse_voxel_size
        )
        self.fine_target = _prepare_icp_cloud(
            target_cloud, config.fine_voxel_size
        )
        self.target_centroid, self.target_basis = _pca_basis(
            np.asarray(self.coarse_target.points)
        )
        self.axis_mappings = _proper_signed_permutation_matrices()
        self.previous_transform = None

    def reset(self):
        self.previous_transform = None

    def _run_coarse(self, source, initial_transform):
        return o3d.pipelines.registration.registration_icp(
            source,
            self.coarse_target,
            self.config.coarse_max_correspondence,
            init=initial_transform,
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=False
                )
            ),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.config.coarse_iterations
            ),
        )

    @staticmethod
    def _is_better(candidate, best):
        if candidate is None:
            return False
        if best is None or candidate.fitness > best.fitness:
            return True
        return (
            np.isclose(candidate.fitness, best.fitness)
            and candidate.inlier_rmse < best.inlier_rmse
        )

    def _global_coarse_search(self, coarse_source):
        source_centroid, source_basis = _pca_basis(
            np.asarray(coarse_source.points)
        )
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
        if best_result is None:
            raise RuntimeError("Failed to produce an ICP initialization")
        return best_result

    def register(self, source_cloud):
        timings = {}
        stage_start = time.perf_counter()
        coarse_source = _prepare_icp_cloud(
            source_cloud, self.config.coarse_voxel_size
        )
        timings["icp_prepare_coarse_source"] = (
            time.perf_counter() - stage_start
        )

        initialization = "temporal"
        coarse_result = None
        if self.previous_transform is not None:
            stage_start = time.perf_counter()
            coarse_result = self._run_coarse(
                coarse_source, self.previous_transform
            )
            timings["icp_temporal_coarse"] = (
                time.perf_counter() - stage_start
            )

        if (
            coarse_result is None
            or coarse_result.fitness < self.config.fallback_fitness
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
        fine_source = _prepare_icp_cloud(
            source_cloud, self.config.fine_voxel_size
        )
        timings["icp_prepare_fine_source"] = (
            time.perf_counter() - stage_start
        )
        stage_start = time.perf_counter()
        fine_result = o3d.pipelines.registration.registration_icp(
            fine_source,
            self.fine_target,
            self.config.fine_max_correspondence,
            init=coarse_result.transformation,
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            ),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.config.fine_iterations
            ),
        )
        timings["icp_fine"] = time.perf_counter() - stage_start
        self.previous_transform = np.asarray(
            fine_result.transformation, dtype=np.float64
        )
        return initialization, coarse_result, fine_result, timings


class PoseEstimator:
    """Estimate object pose from synchronized left/right/color frames."""

    def __init__(self, config):
        self.config = config
        paths_config = self.config["paths"]
        stereo_config = self.config["stereo"]
        reconstruction_config = self.config["reconstruction"]
        point_cloud_config = self.config["point_cloud"]
        segmentation_config = self.config["segmentation"]
        self.icp_config = IcpConfig.from_config(self.config["icp"])

        self.valid_iters = int(stereo_config["valid_iters"])
        self.voxel_size = float(point_cloud_config["voxel_size_mm"])
        self.min_depth_mm = float(point_cloud_config["min_depth_mm"])
        self.max_depth_mm = float(point_cloud_config["max_depth_mm"])
        self._validate_settings()

        self.input_root = resolve_workspace_path(paths_config["input_root"])
        self.sequences = _load_sequences(self.input_root)
        first_paths, first_left, _, _ = self._read_frame(0)
        del first_paths
        height, width = first_left.shape
        self.calibration = _read_calibration(
            resolve_workspace_path(paths_config["int_yaml_path"]),
            resolve_workspace_path(paths_config["ext_yaml_path"]),
            (width, height),
        )
        self.left_maps, self.right_maps = _create_rectification_maps(
            self.calibration, first_left.shape
        )
        self.baseline_mm = abs(
            float(self.calibration["translation"][0, 0])
        )

        self.stereo = _load_stereo_inference(
            resolve_workspace_path(paths_config["stereo_model_path"]),
            stereo_config["device"],
        )
        if not segmentation_config["yolo_model_path"]:
            raise ValueError("A YOLO model path is required")
        self.yolo = _load_yolo(
            resolve_workspace_path(segmentation_config["yolo_model_path"])
        )
        self.yolo_conf = float(segmentation_config["yolo_conf"])
        self.yolo_imgsz = int(segmentation_config["yolo_imgsz"])
        self.yolo_device = segmentation_config["yolo_device"]
        self.statistical_neighbors = int(
            segmentation_config["statistical_nb_neighbors"]
        )
        self.statistical_std_ratio = float(
            segmentation_config["statistical_std_ratio"]
        )

        brightness_config = reconstruction_config["brightness_mask"]
        self.brightness_enabled = bool(brightness_config["enabled"])
        self.brightness_threshold = int(brightness_config["threshold"])

        from jiegouguang.color import ColorMapper

        self.color_mapper = ColorMapper(
            str(resolve_workspace_path(paths_config["color_ext_yaml_path"]))
        )
        self.target_path = resolve_workspace_path(
            paths_config["icp_target_path"]
        )
        if not self.target_path.is_file():
            raise FileNotFoundError(
                "ICP target cloud does not exist: {}".format(self.target_path)
            )
        target_cloud = o3d.io.read_point_cloud(str(self.target_path))
        if len(target_cloud.points) == 0:
            raise ValueError(
                "ICP target cloud is empty: {}".format(self.target_path)
            )
        self.icp = _TemporalRigidIcp(target_cloud, self.icp_config)

    def _validate_settings(self):
        positive_values = {
            "valid_iters": self.valid_iters,
            "voxel_size": self.voxel_size,
            "min_depth_mm": self.min_depth_mm,
            "max_depth_mm": self.max_depth_mm,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError("{} must be greater than zero".format(name))
        if self.max_depth_mm < self.min_depth_mm:
            raise ValueError(
                "max_depth_mm must be greater than or equal to min_depth_mm"
            )
        self.icp_config.validate()

    def __len__(self):
        return len(self.sequences["left"])

    def reset_tracking(self):
        """Discard the previous-frame ICP initialization."""
        self.icp.reset()

    def _read_frame(self, frame_index):
        if frame_index < 0 or frame_index >= len(self):
            raise IndexError(
                "Frame index {} is out of range [0, {})".format(
                    frame_index, len(self)
                )
            )
        paths = {
            name: sequence[frame_index]
            for name, sequence in self.sequences.items()
        }
        left = cv2.imread(str(paths["left"]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(paths["right"]), cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(str(paths["color"]), cv2.IMREAD_COLOR)
        if left is None or right is None or color is None:
            raise ValueError("Failed to read frame {}".format(frame_index))
        if left.shape != right.shape:
            raise ValueError(
                "Left/right image shape mismatch at frame {}".format(
                    frame_index
                )
            )
        return paths, left, right, color

    def estimate_frame(self, frame_index: int) -> PoseEstimate:
        frame_start = time.perf_counter()
        timings = {}

        stage_start = time.perf_counter()
        frame_paths, left, right, color = self._read_frame(frame_index)
        timings["read_images"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        left_rectified = cv2.remap(
            left,
            self.left_maps[0],
            self.left_maps[1],
            interpolation=cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            right,
            self.right_maps[0],
            self.right_maps[1],
            interpolation=cv2.INTER_LINEAR,
        )
        timings["rectify"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        stereo_result = self.stereo.infer(
            left_rectified,
            right_rectified,
            self.calibration["m1"],
            self.baseline_mm,
            valid_iters=self.valid_iters,
        )
        timings["foundation_stereo"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        disparity = stereo_result["disparity"]
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = np.isfinite(disparity) & (disparity > 0)
        depth[valid] = (
            float(self.calibration["m1"][0, 0])
            * self.baseline_mm
            / disparity[valid]
        )
        valid &= (
            (depth >= self.min_depth_mm)
            & (depth <= self.max_depth_mm)
        )
        depth[~valid] = 0
        if self.brightness_enabled:
            depth[left_rectified < self.brightness_threshold] = 0
        timings["depth_filter"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        rgbd = self.color_mapper.get_color(
            depth, color, self.calibration["m1"]
        )
        color_depth = rgbd[:, :, 3]
        timings["map_depth_to_color"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        yolo_mask, instance_count = _create_yolo_mask(
            self.yolo,
            color,
            self.yolo_conf,
            self.yolo_imgsz,
            self.yolo_device,
        )
        timings["yolo"] = time.perf_counter() - stage_start
        if not np.any(yolo_mask):
            return None

        stage_start = time.perf_counter()
        segmented_depth = color_depth.copy()
        segmented_depth[~yolo_mask] = 0
        segmented_color = color.copy()
        segmented_color[~yolo_mask] = 0
        raw_cloud = _depth_to_point_cloud(
            segmented_depth, segmented_color, self.color_mapper.K_RGB
        )
        raw_point_count = len(raw_cloud.points)
        centered_cloud, original_centroid = _filter_and_center_cloud(
            raw_cloud,
            self.voxel_size,
            self.statistical_neighbors,
            self.statistical_std_ratio,
        )
        timings["build_filter_center_cloud"] = (
            time.perf_counter() - stage_start
        )

        stage_start = time.perf_counter()
        initialization, coarse_result, fine_result, icp_timings = (
            self.icp.register(centered_cloud)
        )
        timings["icp_total"] = time.perf_counter() - stage_start
        timings.update(icp_timings)

        centered_to_target = np.asarray(
            fine_result.transformation, dtype=np.float64
        )
        centering_transform = np.eye(4, dtype=np.float64)
        centering_transform[:3, 3] = -original_centroid
        original_to_target = centered_to_target @ centering_transform
        registered_cloud = copy.deepcopy(centered_cloud)
        registered_cloud.transform(centered_to_target)

        return PoseEstimate(
            frame_index=frame_index,
            frame_name=frame_paths["left"].stem,
            frame_paths=frame_paths,
            centered_cloud=centered_cloud,
            registered_cloud=registered_cloud,
            centered_to_target=centered_to_target,
            original_to_target=original_to_target,
            centroid_before_translation=original_centroid,
            translation_mm=centered_to_target[:3, 3].copy(),
            rpy_degrees=matrix_to_rpy_degrees(
                centered_to_target[:3, :3]
            ),
            initialization=initialization,
            coarse_fitness=float(coarse_result.fitness),
            coarse_rmse_mm=float(coarse_result.inlier_rmse),
            fine_fitness=float(fine_result.fitness),
            fine_rmse_mm=float(fine_result.inlier_rmse),
            yolo_instance_count=int(instance_count),
            yolo_mask_pixels=int(yolo_mask.sum()),
            raw_point_count=int(raw_point_count),
            elapsed_seconds=time.perf_counter() - frame_start,
            stage_timings=timings,
        )
