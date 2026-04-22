import os
import logging

import cv2
import numpy as np
import open3d as o3d
import yaml
from ultralytics.models.sam import SAM3DynamicInteractivePredictor

from Recon.Aruco_pose import Aruco_pose_Estimater
from jiegouguang.jiegouguang_class import JieGouGuang


class Reconstruction(object):
    def __init__(
        self,
        color_ext_yaml_path=None,
        aruco_length=148.0,
        world_voxel_size_mm=5.0,
        method='fast_foundation_stereo',
        use_sam=False,
        use_aruco=False,
        sam_model_path='sam3.pt',
        data_root_path=None,
    ):
        self.color_ext_yaml_path = color_ext_yaml_path
        self.aruco_length = float(aruco_length)
        self.world_voxel_size_mm = float(world_voxel_size_mm)
        self.method = method
        self.use_sam = use_sam
        self.use_aruco = use_aruco
        self.sam_model_path = sam_model_path
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
        self.aruco_estimater = None
        self.pose_records = []
        self.depth_list = []
        self.global_cloud = o3d.geometry.PointCloud()
        self.global_cloud_sam = o3d.geometry.PointCloud()
        self.video_writer = None
        self.video_writer_rgb = None

        self.output_dirs = {}
        self.save_dir = None
        self.pose_yaml_path = None
        self.global_cloud_path = None
        self.global_cloud_sam_path = None
        self.pose_vis_dir = None
        self.video_path = None
        self.rgb_video_path = None

        if use_sam:
            self.init_sam()
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
        if self.use_sam:
            self.output_dirs.update({
                'depth_sam': os.path.join(self.save_dir, 'depth_sam'),
                'cloud_sam': os.path.join(self.save_dir, 'cloud_sam'),
                'rgb_sam': os.path.join(self.save_dir, 'rgb_sam'),
            })
        if self.use_aruco:
            aruco_output_dir = os.path.join(self.save_dir, 'aruco_fusion')
            self.pose_vis_dir = os.path.join(aruco_output_dir, 'pose_vis')
            self.pose_yaml_path = os.path.join(aruco_output_dir, 'camera_poses.yaml')
            self.global_cloud_path = os.path.join(aruco_output_dir, 'merged_cloud_marker_frame.ply')
            self.global_cloud_sam_path = os.path.join(aruco_output_dir, 'merged_cloud_marker_frame_sam.ply')
            self.output_dirs['aruco_fusion'] = aruco_output_dir
            self.output_dirs['pose_vis'] = self.pose_vis_dir

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

    def init_aruco(self):
        if not self.use_aruco:
            return
        if self.color_ext_yaml_path is None:
            raise ValueError('Aruco pose estimation requires color_ext_yaml_path')
        self.aruco_estimater = Aruco_pose_Estimater.from_yaml(self.color_ext_yaml_path, aruco_length=self.aruco_length)

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

    def process_frame(self, sample):
        idx = sample['idx']
        camera_params = sample['camera_params']
        self.configure_stereo_context(sample)
        # 继承相机参数

        disparity_raw = self.jiegouguang.forward_disparity()
        depth = (float(camera_params['K1'][0, 0]) * abs(float(camera_params['cam_t'][0]))) / disparity_raw
        depth = np.clip(depth, camera_params['min_dis'], camera_params['max_dis'])
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
            pose_vis = self.aruco_estimater.draw_pose_visualization(aruco_est_frame, frame_pose, idx)
        # 如果有aruco就用aruco估计位姿

        sam_results = None
        if self.use_sam:
            if rgb is None:
                raise ValueError('SAM is enabled but no RGB frame is available for segmentation')
            results = self.predictor(source=rgb)
            masks = results[0].masks
            if masks is not None and len(masks) > 0:
                sam_mask = masks[0].data.cpu().numpy().astype(np.uint8)[0]
                sam_mask = cv2.resize(sam_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            else:
                sam_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=bool)

            depth_sam = np.copy(depth)
            depth_sam[~sam_mask] = 0
            rgb_sam = np.copy(rgb)
            rgb_sam[~sam_mask] = [0, 0, 0]
            sam_results = {
                'mask': sam_mask,
                'depth_sam': depth_sam,
                'rgb_sam': rgb_sam,
            }
        # 如果有SAM就用SAM识别目标并且分割

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
            if self.use_aruco and frame_pose is not None:
                pcd_sam_world = self.aruco_estimater.transform_point_cloud(pcd_sam, frame_pose)
        # 如果有SAM就得到全局和局部的分割后的点云


        pose_record = None
        if frame_pose is not None:
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

        o3d.io.write_point_cloud(os.path.join(self.output_dirs['cloud'], 'cloud_{:04d}.ply'.format(idx)), result['pcd'])
        if result['pcd_world'] is not None:
            self.global_cloud += result['pcd_world']

        if result['sam_results'] is not None:
            cv2.imwrite(
                os.path.join(self.output_dirs['depth_sam'], 'depth_sam_{:04d}.png'.format(idx)),
                result['sam_results']['depth_sam'].astype(np.uint16),
            )
            cv2.imwrite(
                os.path.join(self.output_dirs['rgb_sam'], 'rgb_sam_{:04d}.png'.format(idx)),
                cv2.cvtColor(result['sam_results']['rgb_sam'], cv2.COLOR_BGR2RGB),
            )
            o3d.io.write_point_cloud(os.path.join(self.output_dirs['cloud_sam'], 'cloud_sam_{:04d}.ply'.format(idx)), result['pcd_sam'])
            if result['pcd_sam_world'] is not None:
                self.global_cloud_sam += result['pcd_sam_world']

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

            if self.use_sam and len(self.global_cloud_sam.points) > 0:
                self.global_cloud_sam = self.global_cloud_sam.voxel_down_sample(voxel_size=self.world_voxel_size_mm)
                o3d.io.write_point_cloud(self.global_cloud_sam_path, self.global_cloud_sam)