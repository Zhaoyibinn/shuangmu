
import cv2
import numpy as np
import open3d as o3d
import yaml


class Aruco_pose_Estimater(object):
    def __init__(self, camera_matrix, dist_coeffs, aruco_length, image_size=None, aruco_dict=cv2.aruco.DICT_6X6_250):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        self.aruco_length = float(aruco_length)
        self.image_size = image_size
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
        half_marker_length = self.aruco_length / 2.0
        self.marker_object_points = np.array([
            [-half_marker_length, half_marker_length, 0.0],
            [half_marker_length, half_marker_length, 0.0],
            [half_marker_length, -half_marker_length, 0.0],
            [-half_marker_length, -half_marker_length, 0.0],
        ], dtype=np.float32)

    @classmethod
    def from_yaml(cls, yaml_path, aruco_length, aruco_dict=cv2.aruco.DICT_6X6_250):
        with open(yaml_path, 'r', encoding='utf-8') as file:
            calib = yaml.safe_load(file)

        intrinsics = calib['rgb_intrinsics']
        camera_matrix = np.array([
            [intrinsics['fx'], 0.0, intrinsics['cx']],
            [0.0, intrinsics['fy'], intrinsics['cy']],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        image_size = (int(intrinsics['width']), int(intrinsics['height']))
        return cls(camera_matrix, dist_coeffs, aruco_length, image_size=image_size, aruco_dict=aruco_dict)

    def choose_marker(self, corners, ids):
        if ids is None or len(ids) == 0:
            return None, None

        areas = []
        for corner in corners:
            areas.append(abs(cv2.contourArea(np.asarray(corner[0], dtype=np.float32))))
        best_idx = int(np.argmax(areas))
        return corners[best_idx][0], int(ids[best_idx][0])

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

        success, rvec_obj_to_cam, tvec_obj_to_cam = cv2.solvePnP(
            objectPoints=marker_object_points,
            imagePoints=np.asarray(corners, dtype=np.float32),
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None

        rotation_obj_to_cam, _ = cv2.Rodrigues(rvec_obj_to_cam)
        rotation_cam_to_obj = rotation_obj_to_cam.T
        translation_cam_to_obj = -np.dot(rotation_cam_to_obj, tvec_obj_to_cam)
        rvec_cam_to_obj, _ = cv2.Rodrigues(rotation_cam_to_obj)

        return {
            'corners': all_corners,
            'ids': ids,
            'marker_corners': corners,
            'marker_id': marker_id,
            'rvec_obj_to_cam': rvec_obj_to_cam,
            'tvec_obj_to_cam': tvec_obj_to_cam,
            'rotation_cam_to_obj': rotation_cam_to_obj,
            'translation_cam_to_obj': translation_cam_to_obj,
            'rvec_cam_to_obj': rvec_cam_to_obj,
            'axis_length': axis_length,
        }

    def estimate_pose_from_frame(self, frame):
        corners, ids, marker_corners, marker_id = self.detect_markers(frame)
        if marker_corners is None:
            return None
        return self.estimate_pose(
            marker_corners,
            marker_length=self.aruco_length,
            marker_id=marker_id,
            all_corners=corners,
            ids=ids,
        )

    def draw_pose_visualization(self, frame, pose, frame_idx):
        vis_frame = frame.copy()
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

        cam_t = pose['translation_cam_to_obj'].reshape(-1)
        obj_t = pose['tvec_obj_to_cam'].reshape(-1)
        text_lines = [
            'Frame {} Marker {}'.format(frame_idx, pose['marker_id']),
            'Cam@Marker(mm): ({:.1f}, {:.1f}, {:.1f})'.format(cam_t[0], cam_t[1], cam_t[2]),
            'Marker@Cam(mm): ({:.1f}, {:.1f}, {:.1f})'.format(obj_t[0], obj_t[1], obj_t[2]),
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

    def transform_point_cloud(self, point_cloud, pose):
        if point_cloud is None or pose is None or len(point_cloud.points) == 0:
            return o3d.geometry.PointCloud()

        transformed_cloud = o3d.geometry.PointCloud(point_cloud)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = pose['rotation_cam_to_obj']
        transform[:3, 3] = pose['translation_cam_to_obj'].reshape(3)
        transformed_cloud.transform(transform)
        return transformed_cloud