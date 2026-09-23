
import cv2
import numpy as np
import open3d as o3d
import yaml

from Recon.config_loader import load_config


class Aruco_pose_Estimater(object):
    BOARD_MARKER_PITCH_RATIO = 32.0 / 24.0
    MIN_JOINT_CALIBRATION_FRAMES = 3
    MAX_JOINT_CALIBRATION_RMSE_PX = 2.0

    def __init__(self, camera_matrix, dist_coeffs, aruco_length, image_size=None, aruco_dict=cv2.aruco.DICT_6X6_1000, board_marker_layouts=None):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        self.aruco_length = float(aruco_length)
        self.image_size = image_size
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        params = cv2.aruco.DetectorParameters()

        # params.adaptiveThreshWinSizeMin = 3
        # params.adaptiveThreshWinSizeMax = 53
        # params.adaptiveThreshWinSizeStep = 4
        # params.minMarkerPerimeterRate = 0.01
        # params.maxMarkerPerimeterRate = 4.0
        # params.polygonalApproxAccuracyRate = 0.06
        # params.minCornerDistanceRate = 0.02
        # params.perspectiveRemovePixelPerCell = 8
        # params.errorCorrectionRate = 0.8
        # 用于aruco码识别的调参

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self.anchor_marker_id = None
        self.marker_to_anchor_transforms = {}
        self.board_calibration_frames = {}
        self.board_calibration_results = {}
        self.marker_object_points = self.create_marker_object_points(self.aruco_length)
        if board_marker_layouts is None:
            board_marker_layouts = self.default_board_marker_layouts()
        self.board_marker_layouts = self.normalize_board_marker_layouts(board_marker_layouts)

    @staticmethod
    def create_marker_object_points(marker_length):
        half_marker_length = float(marker_length) / 2.0
        return np.array([
            [-half_marker_length, half_marker_length, 0.0],
            [half_marker_length, half_marker_length, 0.0],
            [half_marker_length, -half_marker_length, 0.0],
            [-half_marker_length, -half_marker_length, 0.0],
        ], dtype=np.float32)

    def default_board_marker_layouts(self):
        marker_pitch = self.aruco_length * self.BOARD_MARKER_PITCH_RATIO
        return {
            0: {
                0: (0.0, marker_pitch),
                1: (-marker_pitch, 0.0),
                2: (marker_pitch, 0.0),
                3: (0.0, -marker_pitch),
            },
            1: {
                4: (0.0, marker_pitch),
                5: (-marker_pitch, 0.0),
                6: (marker_pitch, 0.0),
                7: (0.0, -marker_pitch),
            },
        }

    @staticmethod
    def normalize_board_marker_layouts(board_marker_layouts):
        normalized_layouts = {}
        for board_id, marker_layout in board_marker_layouts.items():
            normalized_markers = {}
            for marker_id, marker_center in marker_layout.items():
                marker_center = np.asarray(marker_center, dtype=np.float64).reshape(-1)
                if marker_center.size == 2:
                    marker_center = np.array([marker_center[0], marker_center[1], 0.0], dtype=np.float64)
                elif marker_center.size != 3:
                    raise ValueError('board marker center must have 2 or 3 values')
                normalized_markers[int(marker_id)] = marker_center
            normalized_layouts[int(board_id)] = normalized_markers
        return normalized_layouts

    @classmethod
    def load_board_config(cls, board_yaml_path):
        with open(board_yaml_path, 'r', encoding='utf-8') as board_file:
            board_config = yaml.safe_load(board_file) or {}
        if 'defaults' in board_config:
            board_config = load_config(board_yaml_path)
        if not isinstance(board_config, dict):
            raise ValueError(
                'ArUco board configuration root must be a mapping: {}'.format(
                    board_yaml_path
                )
            )

        aruco_length = None
        if 'aruco_length' in board_config:
            aruco_length = float(board_config['aruco_length'])

        board_marker_layouts = {}
        for board_id, marker_layout_info in board_config['boards'].items():
            marker_layout = {}
            for marker_id, marker_center in marker_layout_info.items():
                marker_layout[int(marker_id)] = marker_center
            board_marker_layouts[int(board_id)] = marker_layout
        return aruco_length, board_marker_layouts

    @classmethod
    def from_yaml(cls, yaml_path, aruco_length, aruco_dict=cv2.aruco.DICT_6X6_1000, board_marker_layouts=None, board_yaml_path=None):
        with open(yaml_path, 'r', encoding='utf-8') as file:
            is_opencv_yaml = file.readline().startswith('%YAML:')

        if is_opencv_yaml:
            calibration = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
            try:
                camera_matrix = calibration.getNode('M1').mat()
                dist_coeffs = calibration.getNode('D1').mat()
                width = int(calibration.getNode('image_width').real())
                height = int(calibration.getNode('image_height').real())
            finally:
                calibration.release()
            if camera_matrix is None:
                raise ValueError('M1 is missing from {}'.format(yaml_path))
            if dist_coeffs is None:
                dist_coeffs = np.zeros((5, 1), dtype=np.float64)
            image_size = (width, height)
        else:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                calib = yaml.safe_load(file)
            intrinsics = calib['rgb_intrinsics']
            camera_matrix = np.array([
                [intrinsics['fx'], 0.0, intrinsics['cx']],
                [0.0, intrinsics['fy'], intrinsics['cy']],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            distortion_coefficients = calib.get(
                'distortion_coefficients',
                intrinsics.get('distortion_coefficients', [0.0] * 5),
            )
            dist_coeffs = np.asarray(
                distortion_coefficients,
                dtype=np.float64,
            ).reshape(-1, 1)
            image_size = (int(intrinsics['width']), int(intrinsics['height']))

        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        if dist_coeffs.size not in {4, 5, 8, 12, 14}:
            raise ValueError(
                'distortion_coefficients must contain 4, 5, 8, 12, or 14 values'
            )
        if board_yaml_path is not None:
            board_aruco_length, board_marker_layouts = cls.load_board_config(board_yaml_path)
            if board_aruco_length is not None:
                aruco_length = board_aruco_length
        return cls(camera_matrix, dist_coeffs, aruco_length, image_size=image_size, aruco_dict=aruco_dict, board_marker_layouts=board_marker_layouts)

    def choose_marker(self, corners, ids):
        if ids is None or len(ids) == 0:
            return None, None

        areas = []
        for corner in corners:
            areas.append(abs(cv2.contourArea(np.asarray(corner[0], dtype=np.float32))))
        best_idx = int(np.argmax(areas))
        return corners[best_idx][0], int(ids[best_idx][0])

    @staticmethod
    def build_transform(rotation, translation):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return transform

    @staticmethod
    def invert_transform(transform):
        inverse = np.eye(4, dtype=np.float64)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ translation
        return inverse

    @staticmethod
    def choose_best_marker_pose(marker_poses, candidate_ids=None):
        if not marker_poses:
            return None

        if candidate_ids is None:
            candidate_ids = marker_poses.keys()

        best_marker_id = None
        best_area = -1.0
        for marker_id in candidate_ids:
            pose = marker_poses.get(marker_id)
            if pose is None:
                continue
            if float(pose['area']) > best_area:
                best_area = float(pose['area'])
                best_marker_id = marker_id
        return best_marker_id

    def pose_to_marker_from_camera(self, pose):
        return self.build_transform(pose['rotation_cam_to_obj'], pose['translation_cam_to_obj'])

    @staticmethod
    def transform_to_vector(transform):
        rvec, _ = cv2.Rodrigues(
            np.asarray(transform[:3, :3], dtype=np.float64)
        )
        return np.concatenate([
            rvec.reshape(3),
            np.asarray(transform[:3, 3], dtype=np.float64).reshape(3),
        ])

    @classmethod
    def vector_to_transform(cls, vector):
        vector = np.asarray(vector, dtype=np.float64).reshape(6)
        rotation, _ = cv2.Rodrigues(vector[:3])
        return cls.build_transform(rotation, vector[3:])

    def marker_points_in_board(self, marker_center):
        marker_center = np.asarray(marker_center, dtype=np.float64).reshape(3)
        return self.marker_object_points.astype(np.float64) + marker_center.reshape(1, 3)

    @staticmethod
    def collect_detected_markers(corners, ids):
        detected_markers = {}
        if ids is None or len(ids) == 0:
            return detected_markers

        for marker_idx, marker_id_value in enumerate(ids.reshape(-1)):
            marker_id = int(marker_id_value)
            marker_corners = np.asarray(corners[marker_idx][0], dtype=np.float32)
            marker_area = float(abs(cv2.contourArea(marker_corners)))
            previous_marker = detected_markers.get(marker_id)
            if previous_marker is None or marker_area > previous_marker['area']:
                detected_markers[marker_id] = {
                    'corners': marker_corners,
                    'area': marker_area,
                }
        return detected_markers

    def estimate_pose_from_correspondences(self, object_points, image_points, axis_length=None):
        object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            return None

        best_pose = None
        best_reprojection_error = None
        solve_pnp_flags = [cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE]
        for solve_pnp_flag in solve_pnp_flags:
            success, rvec_obj_to_cam, tvec_obj_to_cam = cv2.solvePnP(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=solve_pnp_flag,
            )
            if not success:
                continue

            rotation_obj_to_cam, _ = cv2.Rodrigues(rvec_obj_to_cam)
            rotation_cam_to_obj = rotation_obj_to_cam.T
            translation_cam_to_obj = -np.dot(rotation_cam_to_obj, tvec_obj_to_cam)
            rvec_cam_to_obj, _ = cv2.Rodrigues(rotation_cam_to_obj)
            projected_points, _ = cv2.projectPoints(
                object_points,
                rvec_obj_to_cam,
                tvec_obj_to_cam,
                self.camera_matrix,
                self.dist_coeffs,
            )
            reprojection_errors = np.linalg.norm(projected_points.reshape(-1, 2) - image_points, axis=1)
            reprojection_error_mean = float(np.mean(reprojection_errors))

            pose = {
                'object_points': object_points,
                'image_points': image_points,
                'rvec_obj_to_cam': rvec_obj_to_cam,
                'tvec_obj_to_cam': tvec_obj_to_cam,
                'rotation_cam_to_obj': rotation_cam_to_obj,
                'translation_cam_to_obj': translation_cam_to_obj,
                'rvec_cam_to_obj': rvec_cam_to_obj,
                'axis_length': self.aruco_length if axis_length is None else float(axis_length),
                'solve_pnp_flag': int(solve_pnp_flag),
                'reprojection_error_mean': reprojection_error_mean,
                'reprojection_error_max': float(np.max(reprojection_errors)),
            }
            if best_reprojection_error is None or reprojection_error_mean < best_reprojection_error:
                best_pose = pose
                best_reprojection_error = reprojection_error_mean
        return best_pose

    def estimate_board_poses(self, corners, ids):
        board_poses = {}
        detected_markers = self.collect_detected_markers(corners, ids)
        if not detected_markers:
            return board_poses

        for board_id, marker_layout in self.board_marker_layouts.items():
            object_points = []
            image_points = []
            visible_marker_ids = []
            visible_marker_area = 0.0

            for marker_id, marker_center in marker_layout.items():
                detected_marker = detected_markers.get(marker_id)
                if detected_marker is None:
                    continue
                object_points.append(self.marker_points_in_board(marker_center))
                image_points.append(detected_marker['corners'])
                visible_marker_ids.append(marker_id)
                visible_marker_area += detected_marker['area']

            if not object_points:
                continue

            pose = self.estimate_pose_from_correspondences(
                np.concatenate(object_points, axis=0),
                np.concatenate(image_points, axis=0),
                axis_length=self.aruco_length,
            )
            if pose is None:
                continue

            pose.update({
                'corners': corners,
                'ids': ids,
                'board_id': board_id,
                'marker_id': board_id,
                'visible_board_marker_ids': sorted(int(marker_id) for marker_id in visible_marker_ids),
                'marker_corners': image_points[0],
                'area': visible_marker_area,
            })
            board_poses[board_id] = pose
        return board_poses

    def estimate_marker_poses(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        board_poses = self.estimate_board_poses(corners, ids)
        return corners, ids, board_poses

    def minimum_markers_for_joint_calibration(self, board_id):
        marker_count = len(self.board_marker_layouts[int(board_id)])
        return max(marker_count - 1, 1)

    def pose_has_enough_markers_for_joint_calibration(self, board_id, pose):
        visible_marker_count = len(pose.get('visible_board_marker_ids', []))
        return (
            visible_marker_count
            >= self.minimum_markers_for_joint_calibration(board_id)
        )

    def collect_joint_calibration_frames(self, marker_poses):
        registered_board_ids = [
            board_id
            for board_id in marker_poses
            if board_id in self.marker_to_anchor_transforms
        ]
        unregistered_board_ids = [
            board_id
            for board_id in marker_poses
            if board_id not in self.marker_to_anchor_transforms
        ]

        for reference_board_id in registered_board_ids:
            reference_pose = marker_poses[reference_board_id]
            if not self.pose_has_enough_markers_for_joint_calibration(
                reference_board_id,
                reference_pose,
            ):
                continue

            for candidate_board_id in unregistered_board_ids:
                candidate_pose = marker_poses[candidate_board_id]
                if not self.pose_has_enough_markers_for_joint_calibration(
                    candidate_board_id,
                    candidate_pose,
                ):
                    continue

                key = (int(reference_board_id), int(candidate_board_id))
                frames = self.board_calibration_frames.setdefault(key, [])
                frames.append({
                    'reference_object_points': np.asarray(
                        reference_pose['object_points'],
                        dtype=np.float64,
                    ).copy(),
                    'reference_image_points': np.asarray(
                        reference_pose['image_points'],
                        dtype=np.float64,
                    ).copy(),
                    'candidate_object_points': np.asarray(
                        candidate_pose['object_points'],
                        dtype=np.float64,
                    ).copy(),
                    'candidate_image_points': np.asarray(
                        candidate_pose['image_points'],
                        dtype=np.float64,
                    ).copy(),
                    'reference_from_camera': self.pose_to_marker_from_camera(
                        reference_pose
                    ),
                    'candidate_from_camera': self.pose_to_marker_from_camera(
                        candidate_pose
                    ),
                    'reference_marker_ids': list(
                        reference_pose['visible_board_marker_ids']
                    ),
                    'candidate_marker_ids': list(
                        candidate_pose['visible_board_marker_ids']
                    ),
                })

    @staticmethod
    def choose_medoid_transform(transforms):
        if not transforms:
            return None
        if len(transforms) == 1:
            return np.asarray(transforms[0], dtype=np.float64).copy()

        pairwise_costs = np.zeros(len(transforms), dtype=np.float64)
        for source_idx, source_transform in enumerate(transforms):
            for target_idx, target_transform in enumerate(transforms):
                if target_idx <= source_idx:
                    continue
                delta = (
                    Aruco_pose_Estimater.invert_transform(source_transform)
                    @ target_transform
                )
                rotation_delta, _ = cv2.Rodrigues(delta[:3, :3])
                cost = (
                    np.linalg.norm(delta[:3, 3])
                    + 50.0 * np.linalg.norm(rotation_delta)
                )
                pairwise_costs[source_idx] += cost
                pairwise_costs[target_idx] += cost
        return np.asarray(
            transforms[int(np.argmin(pairwise_costs))],
            dtype=np.float64,
        ).copy()

    def joint_calibration_residuals(self, parameters, frames):
        reference_from_candidate = self.vector_to_transform(parameters[:6])
        residuals = []

        for frame_idx, frame in enumerate(frames):
            camera_from_reference = self.vector_to_transform(
                parameters[6 + frame_idx * 6:12 + frame_idx * 6]
            )
            rvec_camera_from_reference, _ = cv2.Rodrigues(
                camera_from_reference[:3, :3]
            )
            tvec_camera_from_reference = camera_from_reference[:3, 3].reshape(
                3,
                1,
            )

            projected_reference, _ = cv2.projectPoints(
                frame['reference_object_points'],
                rvec_camera_from_reference,
                tvec_camera_from_reference,
                self.camera_matrix,
                self.dist_coeffs,
            )

            candidate_points = frame['candidate_object_points']
            candidate_points_in_reference = (
                reference_from_candidate[:3, :3]
                @ candidate_points.T
            ).T + reference_from_candidate[:3, 3]
            projected_candidate, _ = cv2.projectPoints(
                candidate_points_in_reference,
                rvec_camera_from_reference,
                tvec_camera_from_reference,
                self.camera_matrix,
                self.dist_coeffs,
            )

            residuals.append(
                (
                    projected_reference.reshape(-1, 2)
                    - frame['reference_image_points']
                ).reshape(-1)
            )
            residuals.append(
                (
                    projected_candidate.reshape(-1, 2)
                    - frame['candidate_image_points']
                ).reshape(-1)
            )

        return np.concatenate(residuals)

    def jointly_calibrate_board(self, candidate_board_id):
        candidate_board_id = int(candidate_board_id)
        candidate_keys = [
            key
            for key, frames in self.board_calibration_frames.items()
            if (
                key[1] == candidate_board_id
                and key[0] in self.marker_to_anchor_transforms
                and len(frames) >= self.MIN_JOINT_CALIBRATION_FRAMES
            )
        ]
        if not candidate_keys:
            return False

        calibration_key = max(
            candidate_keys,
            key=lambda key: len(self.board_calibration_frames[key]),
        )
        reference_board_id, _ = calibration_key
        frames = self.board_calibration_frames[calibration_key]

        relative_transforms = []
        camera_from_reference_initials = []
        for frame in frames:
            reference_from_camera = frame['reference_from_camera']
            candidate_from_camera = frame['candidate_from_camera']
            reference_from_candidate = (
                reference_from_camera
                @ self.invert_transform(candidate_from_camera)
            )
            relative_transforms.append(reference_from_candidate)
            camera_from_reference_initials.append(
                self.invert_transform(reference_from_camera)
            )

        initial_reference_from_candidate = self.choose_medoid_transform(
            relative_transforms
        )
        initial_parameters = [self.transform_to_vector(
            initial_reference_from_candidate
        )]
        initial_parameters.extend(
            self.transform_to_vector(transform)
            for transform in camera_from_reference_initials
        )
        initial_parameters = np.concatenate(initial_parameters)

        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise ImportError(
                'Multi-frame ArUco board calibration requires scipy'
            ) from exc

        optimization = least_squares(
            self.joint_calibration_residuals,
            initial_parameters,
            args=(frames,),
            method='trf',
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=300,
        )
        residuals = self.joint_calibration_residuals(
            optimization.x,
            frames,
        )
        reprojection_rmse = float(np.sqrt(np.mean(np.square(residuals))))
        if (
            not np.all(np.isfinite(optimization.x))
            or not np.isfinite(reprojection_rmse)
            or reprojection_rmse > self.MAX_JOINT_CALIBRATION_RMSE_PX
        ):
            return False

        reference_from_candidate = self.vector_to_transform(
            optimization.x[:6]
        )
        anchor_from_reference = self.marker_to_anchor_transforms[
            reference_board_id
        ]
        self.marker_to_anchor_transforms[candidate_board_id] = (
            anchor_from_reference @ reference_from_candidate
        )
        self.board_calibration_results[candidate_board_id] = {
            'reference_board_id': int(reference_board_id),
            'frame_count': len(frames),
            'reprojection_rmse': reprojection_rmse,
            'optimizer_success': bool(optimization.success),
            'optimizer_message': str(optimization.message),
        }
        print(
            'joint ArUco calibration board {} -> anchor {} using {} frames: '
            'rmse={:.4f}px'.format(
                candidate_board_id,
                self.anchor_marker_id,
                len(frames),
                reprojection_rmse,
            )
        )
        return True

    def resolve_world_pose(self, marker_poses):
        if not marker_poses:
            return None

        if self.anchor_marker_id is None:
            self.anchor_marker_id = self.choose_best_marker_pose(marker_poses)
            if self.anchor_marker_id is None:
                return None
            self.marker_to_anchor_transforms[self.anchor_marker_id] = np.eye(4, dtype=np.float64)

        self.collect_joint_calibration_frames(marker_poses)

        preferred_board_id = self.choose_best_marker_pose(marker_poses)
        if (
            preferred_board_id is not None
            and preferred_board_id not in self.marker_to_anchor_transforms
        ):
            self.jointly_calibrate_board(preferred_board_id)

        visible_known_marker_ids = [marker_id for marker_id in marker_poses if marker_id in self.marker_to_anchor_transforms]
        if not visible_known_marker_ids:
            for candidate_board_id in marker_poses:
                if candidate_board_id in self.marker_to_anchor_transforms:
                    continue
                self.jointly_calibrate_board(candidate_board_id)
            visible_known_marker_ids = [
                marker_id
                for marker_id in marker_poses
                if marker_id in self.marker_to_anchor_transforms
            ]
        if not visible_known_marker_ids:
            return None

        reference_marker_id = self.choose_best_marker_pose(marker_poses, visible_known_marker_ids)
        reference_pose = dict(marker_poses[reference_marker_id])

        anchor_from_reference = self.marker_to_anchor_transforms[reference_marker_id]
        reference_from_camera = self.pose_to_marker_from_camera(reference_pose)
        anchor_from_camera = anchor_from_reference @ reference_from_camera

        camera_from_anchor = self.invert_transform(anchor_from_camera)
        rotation_cam_to_world = anchor_from_camera[:3, :3]
        translation_cam_to_world = anchor_from_camera[:3, 3].reshape(3, 1)
        rotation_world_to_cam = camera_from_anchor[:3, :3]
        translation_world_to_cam = camera_from_anchor[:3, 3].reshape(3, 1)
        rvec_world_to_cam, _ = cv2.Rodrigues(rotation_world_to_cam)
        visible_aruco_ids = sorted({
            int(aruco_id)
            for marker_pose in marker_poses.values()
            for aruco_id in marker_pose.get('visible_board_marker_ids', [])
        })

        reference_pose.update({
            'anchor_marker_id': self.anchor_marker_id,
            'reference_marker_id': reference_marker_id,
            'anchor_board_id': self.anchor_marker_id,
            'reference_board_id': reference_marker_id,
            'visible_marker_ids': visible_aruco_ids,
            'registered_marker_ids': sorted(int(marker_id) for marker_id in self.marker_to_anchor_transforms.keys()),
            'visible_board_ids': sorted(int(marker_id) for marker_id in marker_poses.keys()),
            'registered_board_ids': sorted(int(marker_id) for marker_id in self.marker_to_anchor_transforms.keys()),
            'board_calibration_results': {
                int(board_id): dict(result)
                for board_id, result in self.board_calibration_results.items()
            },
            'rotation_cam_to_world': rotation_cam_to_world,
            'translation_cam_to_world': translation_cam_to_world,
            'rotation_world_to_cam': rotation_world_to_cam,
            'translation_world_to_cam': translation_world_to_cam,
            'rvec_world_to_cam': rvec_world_to_cam,
        })
        return reference_pose

    def detect_markers(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        marker_corners, marker_id = self.choose_marker(corners, ids)
        return corners, ids, marker_corners, marker_id

    def estimate_pose(self, corners, marker_length=None, marker_id=None, all_corners=None, ids=None):
        if corners is None:
            return None

        marker_object_points = self.marker_object_points
        axis_length = self.aruco_length * 0.5
        if marker_length is not None and float(marker_length) != self.aruco_length:
            half_marker_length = float(marker_length) / 2.0
            marker_object_points = np.array([
                [-half_marker_length, half_marker_length, 0.0],
                [half_marker_length, half_marker_length, 0.0],
                [half_marker_length, -half_marker_length, 0.0],
                [-half_marker_length, -half_marker_length, 0.0],
            ], dtype=np.float32)
            axis_length = float(marker_length) * 0.5

        pose = self.estimate_pose_from_correspondences(
            marker_object_points,
            np.asarray(corners, dtype=np.float32),
            axis_length=axis_length,
        )
        if pose is None:
            return None

        pose.update({
            'corners': all_corners,
            'ids': ids,
            'marker_corners': corners,
            'marker_id': marker_id,
        })
        return pose

    def estimate_pose_from_frame(self, frame):
        corners, ids, marker_poses = self.estimate_marker_poses(frame)
        if not marker_poses:
            return None

        world_pose = self.resolve_world_pose(marker_poses)
        if world_pose is None:
            return None

        world_pose['corners'] = corners
        world_pose['ids'] = ids
        return world_pose

    def draw_pose_visualization(self, frame, pose, frame_idx):
        if frame.ndim == 2:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            vis_frame = frame.copy()
        elif frame.ndim == 3 and frame.shape[2] == 4:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(
                'pose visualization expects grayscale, BGR, or BGRA input; '
                'got shape {}'.format(frame.shape)
            )
        if pose is None:
            cv2.putText(
                vis_frame,
                'Frame {}: no marker'.format(frame_idx),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return vis_frame

        cv2.aruco.drawDetectedMarkers(vis_frame, pose['corners'], pose['ids'])
        cv2.drawFrameAxes(
            vis_frame,
            self.camera_matrix,
            self.dist_coeffs,
            pose['rvec_obj_to_cam'],
            pose['tvec_obj_to_cam'],
            pose['axis_length'],
        )

        cam_t = pose.get('translation_cam_to_world', pose['translation_cam_to_obj']).reshape(-1)
        text_lines = [
            'Frame {} AnchorBoard {} RefBoard {}'.format(frame_idx, pose['anchor_board_id'], pose['reference_board_id']),
            'Cam@World(mm): ({:.1f}, {:.1f}, {:.1f})'.format(cam_t[0], cam_t[1], cam_t[2]),
            'Board {} ids {} err {:.2f}px'.format(
                pose['board_id'],
                pose.get('visible_board_marker_ids', []),
                pose.get('reprojection_error_mean', 0.0),
            ),
        ]
        calibration_result = pose.get('board_calibration_results', {}).get(
            pose['reference_board_id']
        )
        if calibration_result is not None:
            text_lines.append(
                'JointCalib {} frames rmse {:.2f}px'.format(
                    calibration_result['frame_count'],
                    calibration_result['reprojection_rmse'],
                )
            )
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

    def transform_point_cloud(self, point_cloud, pose):
        if point_cloud is None or pose is None or len(point_cloud.points) == 0:
            return o3d.geometry.PointCloud()

        transformed_cloud = o3d.geometry.PointCloud(point_cloud)
        transform = np.eye(4, dtype=np.float64)
        if 'rotation_cam_to_world' in pose and 'translation_cam_to_world' in pose:
            transform[:3, :3] = pose['rotation_cam_to_world']
            transform[:3, 3] = pose['translation_cam_to_world'].reshape(3)
        else:
            transform[:3, :3] = pose['rotation_cam_to_obj']
            transform[:3, 3] = pose['translation_cam_to_obj'].reshape(3)
        transformed_cloud.transform(transform)
        return transformed_cloud
