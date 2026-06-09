import os
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from Recon.Aruco_pose import Aruco_pose_Estimater
from jiegouguang.jiegouguang_class import JieGouGuang


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
ULTRALYTICS_DIR = WORKSPACE_DIR / 'submodule' / 'ultralytics'
EJRGF_PYTHON_DIR = WORKSPACE_DIR / 'EJRGF' / 'src' / 'python'
try:
    from ultralytics.models.sam import SAM3DynamicInteractivePredictor
except ImportError:
    if ULTRALYTICS_DIR.exists():
        sys.path.insert(0, str(ULTRALYTICS_DIR))
        from ultralytics.models.sam import SAM3DynamicInteractivePredictor
    else:
        raise


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        if ULTRALYTICS_DIR.exists():
            sys.path.insert(0, str(ULTRALYTICS_DIR))
            from ultralytics import YOLO
        else:
            raise
    return YOLO


def import_ejrgf():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'EJRGF registration requires PyTorch with CUDA support'
        ) from exc

    try:
        from EJRGF import EJRGF_register
    except ImportError:
        if EJRGF_PYTHON_DIR.exists() and str(EJRGF_PYTHON_DIR) not in sys.path:
            sys.path.insert(0, str(EJRGF_PYTHON_DIR))
        try:
            from EJRGF import EJRGF_register
        except ImportError as exc:
            raise ImportError(
                'EJRGF was not found. Install it or place its Python module at {}'.format(
                    EJRGF_PYTHON_DIR
                )
            ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            'EJRGF requires CUDA, but torch.cuda.is_available() is False'
        )
    return torch, EJRGF_register


class Reconstruction(object):
    def __init__(
        self,
        color_ext_yaml_path=None,
        aruco_length=148.0,
        world_voxel_size_mm=5.0,
        method='fast_foundation_stereo',
        use_sem=False,
        use_aruco=False,
        brightness_mask_enabled=False,
        brightness_threshold=20,
        segmentation_method='sam',
        sam_model_path='sam3.pt',
        yolo_model_path=None,
        yolo_conf=0.25,
        yolo_imgsz=1280,
        yolo_device=None,
        sem_statistical_nb_neighbors=20,
        sem_statistical_std_ratio=2.0,
        data_root_path=None,
        aruco_board_yaml_path=None,
        use_sam=None,
    ):
        self.color_ext_yaml_path = color_ext_yaml_path
        self.aruco_length = float(aruco_length)
        self.aruco_board_yaml_path = aruco_board_yaml_path
        self.world_voxel_size_mm = float(world_voxel_size_mm)
        self.method = method
        if use_sam is not None:
            use_sem = use_sam
            segmentation_method = 'sam'
        self.use_sem = bool(use_sem)
        self.segmentation_method = str(segmentation_method).lower()
        if self.segmentation_method not in {'sam', 'yolo'}:
            raise ValueError("segmentation_method must be 'sam' or 'yolo'")
        self.use_sam = self.use_sem and self.segmentation_method == 'sam'
        self.use_yolo = self.use_sem and self.segmentation_method == 'yolo'
        self.use_aruco = use_aruco
        self.brightness_mask_enabled = bool(brightness_mask_enabled)
        self.brightness_threshold = int(brightness_threshold)
        if not 0 <= self.brightness_threshold <= 255:
            raise ValueError('brightness_threshold must be between 0 and 255')
        self.sam_model_path = sam_model_path
        self.yolo_model_path = yolo_model_path
        self.yolo_conf = float(yolo_conf)
        self.yolo_imgsz = int(yolo_imgsz)
        self.yolo_device = yolo_device
        self.sem_statistical_nb_neighbors = int(sem_statistical_nb_neighbors)
        self.sem_statistical_std_ratio = float(sem_statistical_std_ratio)
        self.data_root_path = data_root_path
        logging.getLogger('ultralytics').setLevel(logging.ERROR)

        if self.data_root_path is None:
            raise ValueError('Reconstruction requires data_root_path')

        rgb_path = os.path.join(self.data_root_path, 'color')
        self.jiegouguang = JieGouGuang(
            os.path.join(self.data_root_path, 'left'),
            os.path.join(self.data_root_path, 'right'),
            img_rgb_path=rgb_path if os.path.exists(rgb_path) else None,
        )
        self.jiegouguang.method = self.method
        self.jiegouguang.init_model()
        if self.color_ext_yaml_path is not None:
            self.jiegouguang.color_mapper.load_calibration(self.color_ext_yaml_path)

        self.predictor = None
        self.yolo_model = None
        self.aruco_estimater = None
        self.pose_records = []
        self.pose_graph_entries = []
        self.depth_list = []
        self.global_cloud = o3d.geometry.PointCloud()
        self.global_cloud_sam = o3d.geometry.PointCloud()
        self.aruco_clouds = {}
        self.aruco_clouds_sam = {}
        self.video_writer = None
        self.video_writer_rgb = None

        self.output_dirs = {}
        self.save_dir = None
        self.pose_yaml_path = None
        self.global_cloud_path = None
        self.global_cloud_sam_path = None
        self.pose_graph_dir = None
        self.pose_graph_cloud_path = None
        self.pose_graph_cloud_sam_path = None
        self.pose_graph_pose_yaml_path = None
        self.pose_graph_pose_sam_yaml_path = None
        self.ejrgf_dir = None
        self.ejrgf_cloud_path = None
        self.ejrgf_cloud_sam_path = None
        self.ejrgf_pose_yaml_path = None
        self.ejrgf_pose_sam_yaml_path = None
        self.pose_vis_dir = None
        self.cloud_world_dir = None
        self.cloud_world_sam_dir = None
        self.cloud_aruco_dir = None
        self.cloud_aruco_sam_dir = None
        self.video_path = None
        self.rgb_video_path = None

        if self.use_sam:
            self.init_sam()
        if self.use_yolo:
            self.init_yolo()
        if use_aruco:
            self.init_aruco()

    @staticmethod
    def rotation_matrix_to_quaternion(rotation_matrix):
        matrix = np.asarray(rotation_matrix, dtype=np.float64)
        trace = np.trace(matrix)

        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

        quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        return quaternion.tolist()

    @staticmethod
    def frame_pose_to_transform(frame_pose):
        transform = np.eye(4, dtype=np.float64)
        if 'rotation_cam_to_world' in frame_pose and 'translation_cam_to_world' in frame_pose:
            transform[:3, :3] = np.asarray(frame_pose['rotation_cam_to_world'], dtype=np.float64)
            transform[:3, 3] = np.asarray(frame_pose['translation_cam_to_world'], dtype=np.float64).reshape(3)
        else:
            transform[:3, :3] = np.asarray(frame_pose['rotation_cam_to_obj'], dtype=np.float64)
            transform[:3, 3] = np.asarray(frame_pose['translation_cam_to_obj'], dtype=np.float64).reshape(3)
        return transform

    @staticmethod
    def find_matching_image(image_dir, stem):
        for extension in ['.png', '.jpg', '.jpeg', '.bmp']:
            image_path = os.path.join(image_dir, stem + extension)
            if os.path.exists(image_path):
                return image_path
        return None

    def prepare_outputs(self, save_dir):
        self.save_dir = save_dir
        self.output_dirs = {
            'depth': os.path.join(self.save_dir, 'depth'),
            'depth_vis': os.path.join(self.save_dir, 'depth_vis'),
            'cloud': os.path.join(self.save_dir, 'cloud'),
        }
        if self.use_sem:
            self.output_dirs.update({
                'depth_sam': os.path.join(self.save_dir, 'depth_sam'),
                'cloud_sam': os.path.join(self.save_dir, 'cloud_sam'),
                'rgb_sam': os.path.join(self.save_dir, 'rgb_sam'),
            })
        if self.use_aruco:
            aruco_output_dir = os.path.join(self.save_dir, 'aruco_fusion')
            self.pose_graph_dir = os.path.join(aruco_output_dir, 'pose_graph_registration')
            self.ejrgf_dir = os.path.join(aruco_output_dir, 'ejrgf_registration')
            self.pose_vis_dir = os.path.join(aruco_output_dir, 'pose_vis')
            self.cloud_world_dir = os.path.join(aruco_output_dir, 'cloud_world')
            self.cloud_aruco_dir = os.path.join(aruco_output_dir, 'cloud_aruco')
            if self.use_sem:
                self.cloud_world_sam_dir = os.path.join(aruco_output_dir, 'cloud_world_sam')
                self.cloud_aruco_sam_dir = os.path.join(aruco_output_dir, 'cloud_aruco_sam')
            self.pose_yaml_path = os.path.join(aruco_output_dir, 'camera_poses.yaml')
            self.global_cloud_path = os.path.join(aruco_output_dir, 'merged_cloud_marker_frame.ply')
            self.global_cloud_sam_path = os.path.join(aruco_output_dir, 'merged_cloud_marker_frame_sam.ply')
            self.pose_graph_cloud_path = os.path.join(self.pose_graph_dir, 'merged_cloud_pose_graph.ply')
            self.pose_graph_cloud_sam_path = os.path.join(self.pose_graph_dir, 'merged_cloud_pose_graph_sam.ply')
            self.pose_graph_pose_yaml_path = os.path.join(self.pose_graph_dir, 'optimized_poses.yaml')
            self.pose_graph_pose_sam_yaml_path = os.path.join(self.pose_graph_dir, 'optimized_poses_sam.yaml')
            self.ejrgf_cloud_path = os.path.join(self.ejrgf_dir, 'merged_cloud_ejrgf.ply')
            self.ejrgf_cloud_sam_path = os.path.join(self.ejrgf_dir, 'merged_cloud_ejrgf_sam.ply')
            self.ejrgf_pose_yaml_path = os.path.join(self.ejrgf_dir, 'ejrgf_transforms.yaml')
            self.ejrgf_pose_sam_yaml_path = os.path.join(self.ejrgf_dir, 'ejrgf_transforms_sam.yaml')
            self.output_dirs['aruco_fusion'] = aruco_output_dir
            self.output_dirs['pose_graph'] = self.pose_graph_dir
            self.output_dirs['ejrgf'] = self.ejrgf_dir
            self.output_dirs['pose_vis'] = self.pose_vis_dir
            self.output_dirs['cloud_world'] = self.cloud_world_dir
            self.output_dirs['cloud_aruco'] = self.cloud_aruco_dir
            if self.use_sem:
                self.output_dirs['cloud_world_sam'] = self.cloud_world_sam_dir
                self.output_dirs['cloud_aruco_sam'] = self.cloud_aruco_sam_dir

        os.makedirs(self.save_dir, exist_ok=True)
        for directory in self.output_dirs.values():
            os.makedirs(directory, exist_ok=True)

        self.video_path = os.path.join(self.save_dir, 'adepth_sequence.mp4')
        self.rgb_video_path = os.path.join(self.save_dir, 'argb_sequence.mp4')

    def init_sam(self):
        if not self.use_sam:
            return
        if self.data_root_path is None:
            raise ValueError('SAM initialization requires data_root_path')

        sam_loc_dir = os.path.join(self.data_root_path, 'sam_loc')
        sam_loc_files = sorted(os.listdir(sam_loc_dir))
        refer_images = []
        points_list = []
        labels_list = []

        for sam_txt in sam_loc_files:
            if not sam_txt.endswith('.txt'):
                continue
            image_stem = os.path.splitext(sam_txt)[0]
            image_path = self.find_matching_image(os.path.join(self.data_root_path, 'color'), image_stem)
            if image_path is None:
                continue
            refer_images.append(image_path)
            loc = np.loadtxt(os.path.join(sam_loc_dir, sam_txt), delimiter=',', dtype=float)
            points_list.append(loc[:, :2])
            labels_list.append(loc[:, 2])

        if not refer_images or not points_list:
            raise ValueError('SAM is enabled but no valid prompt image/points were found')

        points = np.vstack(points_list).tolist()
        labels = np.int_(np.concatenate(labels_list)).tolist()
        overrides = dict(conf=0.01, task='segment', mode='predict', imgsz=1008, model=self.sam_model_path, save=False, verbose=False)
        self.predictor = SAM3DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)
        self.predictor(source=refer_images[0], points=[points], labels=[labels], obj_ids=[0], update_memory=True)

    def init_yolo(self):
        if not self.use_yolo:
            return
        if self.yolo_model_path is None:
            raise ValueError('YOLO segmentation is enabled but yolo_model_path is not set')

        yolo_model_path = Path(self.yolo_model_path)
        if not yolo_model_path.is_absolute():
            yolo_model_path = WORKSPACE_DIR / yolo_model_path
        if not yolo_model_path.exists():
            raise FileNotFoundError('YOLO segmentation weights not found: {}'.format(yolo_model_path))

        YOLO = import_yolo()
        self.yolo_model = YOLO(str(yolo_model_path))

    @staticmethod
    def polygon_mask_from_yolo(result, mask_id, shape):
        height, width = shape[:2]
        if result.masks.xy is None or mask_id >= len(result.masks.xy):
            return None
        polygon = result.masks.xy[mask_id]
        if polygon is None or len(polygon) < 3:
            return None
        polygon = np.round(polygon).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        return mask.astype(bool)

    @staticmethod
    def align_letterbox_mask(mask, shape):
        height, width = shape[:2]
        mask_height, mask_width = mask.shape[:2]
        if (mask_height, mask_width) == (height, width):
            return mask.astype(bool)

        gain = min(mask_width / float(width), mask_height / float(height))
        if gain <= 0:
            raise ValueError('invalid YOLO mask scale gain: {}'.format(gain))

        unpad_width = int(round(width * gain))
        unpad_height = int(round(height * gain))
        pad_x = int(round((mask_width - unpad_width) / 2.0))
        pad_y = int(round((mask_height - unpad_height) / 2.0))

        x0 = max(pad_x, 0)
        y0 = max(pad_y, 0)
        x1 = min(x0 + unpad_width, mask_width)
        y1 = min(y0 + unpad_height, mask_height)
        cropped = mask[y0:y1, x0:x1]
        if cropped.size == 0:
            raise ValueError(
                'empty YOLO mask crop from mask shape {} to image shape {}'.format(
                    mask.shape,
                    (height, width),
                )
            )

        aligned = cv2.resize(
            cropped.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return aligned > 0

    def segment_frame(self, rgb, depth_shape):
        if rgb is None:
            raise ValueError('semantic segmentation is enabled but no RGB frame is available')

        if self.use_sam:
            results = self.predictor(source=rgb)
            masks = results[0].masks
            if masks is None or len(masks) == 0:
                return np.zeros(depth_shape, dtype=bool)
            sem_mask = masks[0].data.cpu().numpy().astype(np.uint8)[0]
            if sem_mask.shape != depth_shape:
                sem_mask = cv2.resize(sem_mask, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_NEAREST)
            return sem_mask > 0

        if self.use_yolo:
            predict_kwargs = {
                'source': rgb,
                'task': 'segment',
                'imgsz': self.yolo_imgsz,
                'conf': self.yolo_conf,
                'retina_masks': True,
                'save': False,
                'verbose': False,
            }
            if self.yolo_device is not None:
                predict_kwargs['device'] = self.yolo_device
            results = self.yolo_model.predict(**predict_kwargs)
            if not results or results[0].masks is None or len(results[0].masks) == 0:
                return np.zeros(depth_shape, dtype=bool)

            result = results[0]
            sem_mask = np.zeros(depth_shape, dtype=bool)
            for mask_id in range(len(result.masks)):
                mask = result.masks.data[mask_id].cpu().numpy() > 0.5
                try:
                    sem_mask |= self.align_letterbox_mask(mask, depth_shape)
                except ValueError:
                    polygon_mask = self.polygon_mask_from_yolo(result, mask_id, depth_shape)
                    if polygon_mask is not None:
                        sem_mask |= polygon_mask
            return sem_mask

        return np.zeros(depth_shape, dtype=bool)

    def filter_semantic_point_cloud(self, point_cloud):
        if point_cloud is None or len(point_cloud.points) == 0:
            return point_cloud
        if self.sem_statistical_nb_neighbors <= 0:
            return point_cloud
        if len(point_cloud.points) <= self.sem_statistical_nb_neighbors:
            return point_cloud

        filtered_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=self.sem_statistical_nb_neighbors,
            std_ratio=self.sem_statistical_std_ratio,
        )
        return filtered_cloud

    def init_aruco(self):
        if not self.use_aruco:
            return
        if self.color_ext_yaml_path is None:
            raise ValueError('Aruco pose estimation requires color_ext_yaml_path')
        self.aruco_estimater = Aruco_pose_Estimater.from_yaml(
            self.color_ext_yaml_path,
            aruco_length=self.aruco_length,
            board_yaml_path=self.aruco_board_yaml_path,
        )
        self.aruco_length = self.aruco_estimater.aruco_length

    def configure_stereo_context(self, sample):
        camera_params = sample['camera_params']
        self.jiegouguang.img1_rectify = sample['left_rectified']
        self.jiegouguang.img2_rectify = sample['right_rectified']
        self.jiegouguang.K1 = camera_params['K1']
        self.jiegouguang.K2 = camera_params['K2']
        self.jiegouguang.cam_R = camera_params['cam_R']
        self.jiegouguang.cam_t = camera_params['cam_t']
        self.jiegouguang.P1 = camera_params['P1']
        self.jiegouguang.P2 = camera_params['P2']
        self.jiegouguang.Q = camera_params['Q']
        self.jiegouguang.min_dis = camera_params['min_dis']
        self.jiegouguang.max_dis = camera_params['max_dis']
        self.jiegouguang.min_disp = camera_params['min_disp']
        self.jiegouguang.max_disp = camera_params['max_disp']

    def apply_brightness_mask(self, depth, left_image):
        if not self.brightness_mask_enabled:
            return depth
        if left_image is None:
            raise ValueError('brightness mask is enabled but left image is missing')

        if left_image.ndim == 3:
            brightness = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        elif left_image.ndim == 2:
            brightness = left_image
        else:
            raise ValueError(
                'left image must be grayscale or BGR, got shape {}'.format(
                    left_image.shape
                )
            )

        if brightness.shape != depth.shape:
            raise ValueError(
                'brightness mask shape {} does not match depth shape {}'.format(
                    brightness.shape,
                    depth.shape,
                )
            )

        masked_depth = np.copy(depth)
        masked_depth[brightness < self.brightness_threshold] = 0
        return masked_depth

    def process_frame(self, sample):
        idx = sample['idx']
        camera_params = sample['camera_params']
        self.configure_stereo_context(sample)
        # 继承相机参数

        disparity_raw = self.jiegouguang.forward_disparity()
        depth = (float(camera_params['K1'][0, 0]) * abs(float(camera_params['cam_t'][0]))) / disparity_raw
        depth = np.clip(depth, camera_params['min_dis'], camera_params['max_dis'])
        depth = self.apply_brightness_mask(depth, sample['left_rectified'])
        self.depth_list.append(depth)
        # 推理深度图

        rgb = None
        aruco_est_frame = sample['gray_left']
        if sample['rgb'] is not None:
            rgbd = self.jiegouguang.get_rgbd(depth_L=depth, rgb_img=sample['rgb'])
            rgb = rgbd[:, :, :3].astype(np.uint8)
            depth = rgbd[:, :, 3].astype(np.uint16)
            aruco_est_frame = rgb
        # 如果有RGB就执行反投影

        frame_pose = None
        pose_vis = None
        if self.use_aruco:
            frame_pose = self.aruco_estimater.estimate_pose_from_frame(aruco_est_frame)
            if frame_pose == None:
                print("frame_pose None")
            pose_vis = self.aruco_estimater.draw_pose_visualization(aruco_est_frame, frame_pose, idx)
        # 如果有aruco就用aruco估计位姿

        sam_results = None
        if self.use_sem:
            sam_mask = self.segment_frame(rgb, depth.shape)

            depth_sam = np.copy(depth)
            depth_sam[~sam_mask] = 0
            rgb_sam = np.copy(rgb)
            rgb_sam[~sam_mask] = [0, 0, 0]
            sam_results = {
                'mask': sam_mask,
                'depth_sam': depth_sam,
                'rgb_sam': rgb_sam,
            }
        # 如果有语义分割就识别目标并且分割

        pcd = self.jiegouguang.depth2pointcloud(depth, color_image=rgb)
        if len(pcd.colors) == 0:
            pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
        # 生成点云

        pcd_world = None
        if self.use_aruco and frame_pose is not None:
            pcd_world = self.aruco_estimater.transform_point_cloud(pcd, frame_pose)
        # 如果上面ARUCO估计了位姿就计算全局点云

        pcd_sam = None
        pcd_sam_world = None
        if sam_results is not None:
            pcd_sam = self.jiegouguang.depth2pointcloud(sam_results['depth_sam'], color_image=sam_results['rgb_sam'])
            if len(pcd_sam.colors) == 0:
                pcd_sam.colors = o3d.utility.Vector3dVector(np.repeat([[0.0, 1.0, 0.0]], len(pcd_sam.points), axis=0))
            pcd_sam = self.filter_semantic_point_cloud(pcd_sam)
            if self.use_aruco and frame_pose is not None:
                pcd_sam_world = self.aruco_estimater.transform_point_cloud(pcd_sam, frame_pose)
        # 如果有语义分割就得到全局和局部的分割后的点云


        pose_record = None
        if frame_pose is not None:
            if 'rotation_world_to_cam' in frame_pose and 'translation_world_to_cam' in frame_pose:
                rotation_w2c = frame_pose['rotation_world_to_cam']
                translation_w2c = frame_pose['translation_world_to_cam'].reshape(-1)
            else:
                rotation_w2c, _ = cv2.Rodrigues(frame_pose['rvec_obj_to_cam'])
                translation_w2c = frame_pose['tvec_obj_to_cam'].reshape(-1)
            pose_record = {
                'rotation': self.rotation_matrix_to_quaternion(rotation_w2c),
                'translation': translation_w2c.astype(np.float64).tolist(),
            }
        # 记录位姿

        return {
            'idx': idx,
            'sample': sample,
            'disparity': disparity_raw,
            'depth': depth,
            'rgb': rgb,
            'aruco_est_frame': aruco_est_frame,
            'frame_pose': frame_pose,
            'pose_vis': pose_vis,
            'pose_record': pose_record,
            'sam_results': sam_results,
            'pcd': pcd,
            'pcd_world': pcd_world,
            'pcd_sam': pcd_sam,
            'pcd_sam_world': pcd_sam_world,
        }

    def save_frame_result(self, result):
        idx = result['idx']
        depth = result['depth']
        rgb = result['rgb']
        depth_vis = np.clip(depth * 0.2, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(self.output_dirs['depth'], 'depth_{:04d}.png'.format(idx)), depth.astype(np.uint16))
        cv2.imwrite(os.path.join(self.output_dirs['depth_vis'], 'depth_vis_{:04d}.png'.format(idx)), (depth_vis / 2).astype(np.uint8))

        if self.video_writer is None:
            height, width = depth_vis.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 30, (width, height), False)
            if not self.video_writer.isOpened():
                raise RuntimeError('failed to open video writer for {}'.format(self.video_path))
        self.video_writer.write(depth_vis)

        if rgb is not None:
            if self.video_writer_rgb is None:
                height, width, _ = rgb.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer_rgb = cv2.VideoWriter(self.rgb_video_path, fourcc, 30, (width, height), True)
                if not self.video_writer_rgb.isOpened():
                    raise RuntimeError('failed to open video writer for {}'.format(self.rgb_video_path))
            self.video_writer_rgb.write(rgb)

        if self.use_aruco and result['pose_vis'] is not None:
            cv2.imwrite(os.path.join(self.pose_vis_dir, 'frame_{:04d}.png'.format(idx)), result['pose_vis'])
        if result['pose_record'] is not None:
            self.pose_records.append(result['pose_record'])

        cloud_path = os.path.join(self.output_dirs['cloud'], 'cloud_{:04d}.ply'.format(idx))
        cloud_sam_path = None
        cloud_world_path = None
        cloud_world_sam_path = None
        o3d.io.write_point_cloud(cloud_path, result['pcd'])
        if result['pcd_world'] is not None:
            cloud_world_path = os.path.join(
                self.cloud_world_dir,
                'cloud_world_{:04d}.ply'.format(idx),
            )
            o3d.io.write_point_cloud(
                cloud_world_path,
                result['pcd_world'],
            )
            self.global_cloud += result['pcd_world']
            frame_pose = result['frame_pose']
            aruco_board_id = frame_pose.get('reference_board_id', frame_pose.get('board_id'))
            if aruco_board_id is not None:
                aruco_board_id = int(aruco_board_id)
                if aruco_board_id not in self.aruco_clouds:
                    self.aruco_clouds[aruco_board_id] = o3d.geometry.PointCloud()
                self.aruco_clouds[aruco_board_id] += result['pcd_world']

        if result['sam_results'] is not None:
            cv2.imwrite(
                os.path.join(self.output_dirs['depth_sam'], 'depth_sam_{:04d}.png'.format(idx)),
                result['sam_results']['depth_sam'].astype(np.uint16),
            )
            cv2.imwrite(
                os.path.join(self.output_dirs['rgb_sam'], 'rgb_sam_{:04d}.png'.format(idx)),
                cv2.cvtColor(result['sam_results']['rgb_sam'], cv2.COLOR_BGR2RGB),
            )
            saved_cloud_sam_path = os.path.join(self.output_dirs['cloud_sam'], 'cloud_sam_{:04d}.ply'.format(idx))
            o3d.io.write_point_cloud(saved_cloud_sam_path, result['pcd_sam'])
            if len(result['pcd_sam'].points) > 0:
                cloud_sam_path = saved_cloud_sam_path
            if result['pcd_sam_world'] is not None:
                saved_cloud_world_sam_path = os.path.join(
                    self.cloud_world_sam_dir,
                    'cloud_world_sam_{:04d}.ply'.format(idx),
                )
                o3d.io.write_point_cloud(
                    saved_cloud_world_sam_path,
                    result['pcd_sam_world'],
                )
                if len(result['pcd_sam_world'].points) > 0:
                    cloud_world_sam_path = saved_cloud_world_sam_path
                self.global_cloud_sam += result['pcd_sam_world']
                frame_pose = result['frame_pose']
                aruco_board_id = frame_pose.get('reference_board_id', frame_pose.get('board_id'))
                if aruco_board_id is not None:
                    aruco_board_id = int(aruco_board_id)
                    if aruco_board_id not in self.aruco_clouds_sam:
                        self.aruco_clouds_sam[aruco_board_id] = o3d.geometry.PointCloud()
                    self.aruco_clouds_sam[aruco_board_id] += result['pcd_sam_world']

        if self.use_aruco and result['frame_pose'] is not None:
            self.pose_graph_entries.append({
                'idx': int(idx),
                'cloud_path': cloud_path,
                'cloud_sam_path': cloud_sam_path,
                'cloud_world_path': cloud_world_path,
                'cloud_world_sam_path': cloud_world_sam_path,
                'initial_pose': self.frame_pose_to_transform(result['frame_pose']),
            })

    @staticmethod
    def should_add_loop_edge(source_id, target_id, stride):
        if stride <= 0:
            return False
        if target_id <= source_id + 1:
            return False
        return (target_id - source_id) % stride == 0

    def load_registration_cloud(self, cloud_path, voxel_size):
        point_cloud = o3d.io.read_point_cloud(cloud_path)
        if len(point_cloud.points) == 0:
            raise ValueError('empty point cloud: {}'.format(cloud_path))

        if voxel_size > 0:
            point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        if len(point_cloud.points) == 0:
            raise ValueError('empty point cloud after voxel downsampling: {}'.format(cloud_path))

        if not point_cloud.has_normals():
            normal_radius = voxel_size * 2.0 if voxel_size > 0 else 2.0
            point_cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
            )
        return point_cloud

    @staticmethod
    def pairwise_registration(source, target, init_transform, max_correspondence_distance, max_iteration):
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=criteria,
        )
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source,
            target,
            max_correspondence_distance,
            result.transformation,
        )
        return result.transformation, information, result.fitness, result.inlier_rmse

    def build_pose_graph(
        self,
        point_clouds,
        initial_poses,
        frame_indices,
        max_correspondence_distance,
        max_iteration,
        loop_closure_stride,
    ):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        for initial_pose in initial_poses:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(initial_pose))

        for source_id in range(len(point_clouds)):
            for target_id in range(source_id + 1, len(point_clouds)):
                is_neighbor = target_id == source_id + 1
                is_loop = self.should_add_loop_edge(source_id, target_id, loop_closure_stride)
                if not is_neighbor and not is_loop:
                    continue

                init_transform = np.linalg.inv(initial_poses[target_id]) @ initial_poses[source_id]
                transformation, information, fitness, rmse = self.pairwise_registration(
                    point_clouds[source_id],
                    point_clouds[target_id],
                    init_transform,
                    max_correspondence_distance,
                    max_iteration,
                )
                print(
                    'pose graph frame {} -> {}: fitness={:.6f}, rmse={:.6f}, uncertain={}'.format(
                        frame_indices[source_id],
                        frame_indices[target_id],
                        fitness,
                        rmse,
                        not is_neighbor,
                    )
                )

                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation,
                        information,
                        uncertain=not is_neighbor,
                    )
                )
        return pose_graph

    @staticmethod
    def optimize_pose_graph(
        pose_graph,
        max_correspondence_distance,
        edge_prune_threshold,
        preference_loop_closure,
        reference_node,
    ):
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=edge_prune_threshold,
            preference_loop_closure=preference_loop_closure,
            reference_node=reference_node,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

    @staticmethod
    def fuse_cloud_paths(
        cloud_paths,
        poses,
        voxel_size,
        merge_downsample_every=0,
    ):
        fused_cloud = o3d.geometry.PointCloud()
        for index, (cloud_path, pose) in enumerate(zip(cloud_paths, poses)):
            point_cloud = o3d.io.read_point_cloud(cloud_path)
            if len(point_cloud.points) == 0:
                continue
            point_cloud.transform(pose)
            fused_cloud += point_cloud
            if (
                voxel_size > 0
                and merge_downsample_every > 0
                and (index + 1) % merge_downsample_every == 0
            ):
                fused_cloud = fused_cloud.voxel_down_sample(voxel_size=voxel_size)

        if voxel_size > 0 and len(fused_cloud.points) > 0:
            fused_cloud = fused_cloud.voxel_down_sample(voxel_size=voxel_size)
        return fused_cloud

    @staticmethod
    def pose_graph_yaml_records(frame_indices, poses):
        records = []
        for frame_idx, pose in zip(frame_indices, poses):
            records.append({
                'idx': int(frame_idx),
                'transform': np.asarray(pose, dtype=np.float64).tolist(),
            })
        return records

    def run_global_pose_graph_optimization(
        self,
        use_sem=False,
        voxel_size_mm=None,
        final_voxel_size_mm=0.0,
        max_correspondence_distance_mm=10.0,
        max_iteration=100,
        loop_closure_stride=30,
        preference_loop_closure=0.1,
        edge_prune_threshold=0.25,
        reference_node=0,
        use_sam=None,
    ):
        if use_sam is not None:
            use_sem = use_sam
        if not self.use_aruco:
            print('skip pose graph optimization: use_aruco is disabled')
            return None
        if len(self.pose_graph_entries) < 2:
            print('skip pose graph optimization: need at least two valid pose frames')
            return None

        voxel_size = self.world_voxel_size_mm if voxel_size_mm is None else float(voxel_size_mm)
        cloud_key = 'cloud_sam_path' if use_sem else 'cloud_path'
        entries = [
            entry for entry in self.pose_graph_entries
            if entry.get(cloud_key) is not None and os.path.exists(entry[cloud_key])
        ]
        if len(entries) < 2:
            print('skip pose graph optimization: need at least two existing point clouds')
            return None

        label = self.segmentation_method if use_sem else 'full'
        print('Building {} pose graph from {} frames...'.format(label, len(entries)))
        frame_indices = [entry['idx'] for entry in entries]
        cloud_paths = [entry[cloud_key] for entry in entries]
        initial_poses = [entry['initial_pose'] for entry in entries]
        point_clouds = [
            self.load_registration_cloud(cloud_path, voxel_size)
            for cloud_path in cloud_paths
        ]

        pose_graph = self.build_pose_graph(
            point_clouds,
            initial_poses,
            frame_indices,
            max_correspondence_distance_mm,
            max_iteration,
            loop_closure_stride,
        )
        print('Pose graph nodes: {}, edges: {}'.format(len(pose_graph.nodes), len(pose_graph.edges)))
        if len(pose_graph.edges) == 0:
            print('skip pose graph optimization: no graph edges were created')
            return None

        print('Running Open3D global pose graph optimization...')
        self.optimize_pose_graph(
            pose_graph,
            max_correspondence_distance_mm,
            edge_prune_threshold,
            preference_loop_closure,
            reference_node,
        )

        optimized_poses = [node.pose for node in pose_graph.nodes]
        fused_cloud = self.fuse_cloud_paths(
            cloud_paths,
            optimized_poses,
            voxel_size=float(final_voxel_size_mm),
        )
        output_path = self.pose_graph_cloud_sam_path if use_sem else self.pose_graph_cloud_path
        pose_yaml_path = self.pose_graph_pose_sam_yaml_path if use_sem else self.pose_graph_pose_yaml_path
        o3d.io.write_point_cloud(output_path, fused_cloud)
        with open(pose_yaml_path, 'w', encoding='utf-8') as pose_yaml_file:
            yaml.safe_dump(
                {'optimized_poses': self.pose_graph_yaml_records(frame_indices, optimized_poses)},
                pose_yaml_file,
                allow_unicode=True,
                sort_keys=False,
            )
        print('Saved pose graph fused cloud: {}'.format(output_path))
        return {
            'method': 'pose_graph',
            'pose_graph': pose_graph,
            'output_path': output_path,
            'pose_yaml_path': pose_yaml_path,
            'points': len(fused_cloud.points),
        }

    @staticmethod
    def normalize_transforms_to_first_frame(transforms):
        if not transforms:
            return transforms

        first_inverse = np.linalg.inv(transforms[0])
        normalized_transforms = [
            first_inverse @ np.asarray(transform, dtype=np.float64)
            for transform in transforms
        ]
        normalized_transforms[0] = np.eye(4, dtype=np.float64)
        return normalized_transforms

    @staticmethod
    def load_ejrgf_registration_tensor(
        cloud_path,
        voxel_size,
        max_points,
        rng,
        torch,
    ):
        point_cloud = o3d.io.read_point_cloud(cloud_path)
        if len(point_cloud.points) == 0:
            raise ValueError('empty point cloud: {}'.format(cloud_path))

        if voxel_size > 0:
            point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

        points = np.asarray(point_cloud.points, dtype=np.float32)
        if points.shape[0] == 0:
            raise ValueError(
                'empty point cloud after voxel downsampling: {}'.format(cloud_path)
            )
        if max_points > 0 and points.shape[0] > max_points:
            keep_indices = rng.choice(
                points.shape[0],
                size=max_points,
                replace=False,
            )
            points = points[keep_indices]

        return torch.from_numpy(np.ascontiguousarray(points)).float().cuda()

    def run_ejrgf_registration(
        self,
        use_sem=False,
        registration_voxel_size_mm=1.0,
        final_voxel_size_mm=1.0,
        max_registration_points=30000,
        subgroup_size=10,
        gmm_mean_local_num=500,
        epsilon=1e-6,
        local_sigma=0.0,
        local_iteration_num=200,
        global_refinement=False,
        gmm_mean_global_num=1000,
        global_sigma=0.0,
        global_iteration_num=100,
        seed=0,
        merge_downsample_every=0,
        use_sam=None,
    ):
        if use_sam is not None:
            use_sem = use_sam
        if not self.use_aruco:
            print('skip EJRGF registration: use_aruco is disabled')
            return None
        if len(self.pose_graph_entries) < 2:
            print('skip EJRGF registration: need at least two valid pose frames')
            return None

        cloud_key = 'cloud_world_sam_path' if use_sem else 'cloud_world_path'
        entries = [
            entry for entry in self.pose_graph_entries
            if entry.get(cloud_key) is not None and os.path.exists(entry[cloud_key])
        ]
        if len(entries) < 2:
            print('skip EJRGF registration: need at least two existing world point clouds')
            return None

        torch, ejrgf_register = import_ejrgf()
        frame_indices = [entry['idx'] for entry in entries]
        cloud_paths = [entry[cloud_key] for entry in entries]
        rng = np.random.default_rng(seed)
        registration_tensors = []

        label = self.segmentation_method if use_sem else 'full'
        print('Loading {} {} clouds for EJRGF registration...'.format(len(entries), label))
        for index, cloud_path in enumerate(cloud_paths):
            tensor = self.load_ejrgf_registration_tensor(
                cloud_path,
                voxel_size=float(registration_voxel_size_mm),
                max_points=int(max_registration_points),
                rng=rng,
                torch=torch,
            )
            registration_tensors.append(tensor)
            print(
                '[{:04d}/{:04d}] frame {}: {} registration points'.format(
                    index + 1,
                    len(cloud_paths),
                    frame_indices[index],
                    tensor.shape[0],
                )
            )

        print('Running EJRGF global registration...')
        transform_tensors = ejrgf_register(
            registration_tensors,
            int(subgroup_size),
            int(gmm_mean_local_num),
            float(epsilon),
            float(local_sigma),
            int(local_iteration_num),
            bool(global_refinement),
            int(gmm_mean_global_num),
            float(global_sigma),
            int(global_iteration_num),
        )
        if len(transform_tensors) != len(cloud_paths):
            raise RuntimeError(
                'EJRGF returned {} transforms for {} clouds'.format(
                    len(transform_tensors),
                    len(cloud_paths),
                )
            )

        raw_transforms = [
            transform.detach().cpu().numpy().astype(np.float64)
            for transform in transform_tensors
        ]
        transforms = self.normalize_transforms_to_first_frame(raw_transforms)

        suffix = '_sam' if use_sem else ''
        raw_transform_path = os.path.join(
            self.ejrgf_dir,
            'ejrgf_transforms_raw{}.npy'.format(suffix),
        )
        transform_path = os.path.join(
            self.ejrgf_dir,
            'ejrgf_transforms{}.npy'.format(suffix),
        )
        np.save(raw_transform_path, np.stack(raw_transforms, axis=0))
        np.save(transform_path, np.stack(transforms, axis=0))

        fused_cloud = self.fuse_cloud_paths(
            cloud_paths,
            transforms,
            voxel_size=float(final_voxel_size_mm),
            merge_downsample_every=int(merge_downsample_every),
        )
        output_path = self.ejrgf_cloud_sam_path if use_sem else self.ejrgf_cloud_path
        pose_yaml_path = self.ejrgf_pose_sam_yaml_path if use_sem else self.ejrgf_pose_yaml_path
        o3d.io.write_point_cloud(output_path, fused_cloud)
        with open(pose_yaml_path, 'w', encoding='utf-8') as pose_yaml_file:
            yaml.safe_dump(
                {
                    'ejrgf_transforms': self.pose_graph_yaml_records(
                        frame_indices,
                        transforms,
                    )
                },
                pose_yaml_file,
                allow_unicode=True,
                sort_keys=False,
            )

        print('Saved EJRGF fused cloud: {}'.format(output_path))
        return {
            'method': 'ejrgf',
            'output_path': output_path,
            'pose_yaml_path': pose_yaml_path,
            'transform_path': transform_path,
            'raw_transform_path': raw_transform_path,
            'points': len(fused_cloud.points),
        }

    def run_global_registration(self, method='pose_graph', use_sem=False, **kwargs):
        method = str(method).lower()
        if method == 'pose_graph':
            return self.run_global_pose_graph_optimization(
                use_sem=use_sem,
                **kwargs
            )
        if method == 'ejrgf':
            return self.run_ejrgf_registration(
                use_sem=use_sem,
                **kwargs
            )
        raise ValueError(
            "global registration method must be 'pose_graph' or 'ejrgf', got {!r}".format(
                method
            )
        )

    def save_final_result(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if self.video_writer_rgb is not None:
            self.video_writer_rgb.release()

        if self.use_aruco:
            with open(self.pose_yaml_path, 'w', encoding='utf-8') as pose_yaml_file:
                yaml.safe_dump({'camera_poses': self.pose_records}, pose_yaml_file, allow_unicode=True, sort_keys=False)

            if len(self.global_cloud.points) > 0:
                self.global_cloud = self.global_cloud.voxel_down_sample(voxel_size=self.world_voxel_size_mm)
                o3d.io.write_point_cloud(self.global_cloud_path, self.global_cloud)

            if self.use_sem and len(self.global_cloud_sam.points) > 0:
                self.global_cloud_sam = self.global_cloud_sam.voxel_down_sample(voxel_size=self.world_voxel_size_mm)
                o3d.io.write_point_cloud(self.global_cloud_sam_path, self.global_cloud_sam)

            for aruco_board_id, aruco_cloud in sorted(self.aruco_clouds.items()):
                if len(aruco_cloud.points) == 0:
                    continue
                aruco_cloud = aruco_cloud.voxel_down_sample(voxel_size=self.world_voxel_size_mm)
                o3d.io.write_point_cloud(
                    os.path.join(self.cloud_aruco_dir, 'cloud_aruco_board_{:04d}.ply'.format(aruco_board_id)),
                    aruco_cloud,
                )

            if self.use_sem:
                for aruco_board_id, aruco_cloud_sam in sorted(self.aruco_clouds_sam.items()):
                    if len(aruco_cloud_sam.points) == 0:
                        continue
                    aruco_cloud_sam = aruco_cloud_sam.voxel_down_sample(voxel_size=self.world_voxel_size_mm)
                    o3d.io.write_point_cloud(
                        os.path.join(self.cloud_aruco_sam_dir, 'cloud_aruco_sam_board_{:04d}.ply'.format(aruco_board_id)),
                        aruco_cloud_sam,
                    )
