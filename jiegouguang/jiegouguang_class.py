import cv2
import numpy as np
import copy
import open3d as o3d
from pathlib import Path

# 下面都是自己实现的各种方法
from manual_feature.ManualFeature import ManualFeatureJieGouGuang
from LightGlue_feature.NeuralFeature import NeuralFeatureJieGouGuang
from sgbm.sgbm import SGBM
from FoundationStereo.stereo_inference import StereoInference
from bridgedepth.bridgedepth_stereo import BridgeDepthStereo
from color import ColorMapper  # 颜色映射模块 - DX

class JieGouGuang:
    def __init__(self,img1_path,img2_path,img_rgb_path = None):
        # self.img1_path = img1_path
        # self.img2_path = img2_path

        def _load_media_sequence(path,color = False):
            """Load an image, video, or its directory into a list."""
            path = Path(path)
            frames = []
            if path.is_dir():
                supported_exts = {".png", ".jpg", ".jpeg", ".bmp"}
                images = sorted(
                    p for p in path.iterdir() if p.is_file() and p.suffix.lower() in supported_exts
                )
                if not images:
                    raise ValueError(f'目录不包含支持的图像：{path}')
                for img_path in images:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError(f'failed to read image {img_path}')
                    if not color and img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    elif color and img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    frames.append(img)
                return frames

            cap = cv2.VideoCapture(str(path))
            try:
                if cap.isOpened():
                    ret, frame = cap.read()
                    while ret:
                        if not color:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                        else:
                            frames.append(frame)
                        ret, frame = cap.read()
                    if frames:
                        return frames
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f'failed to read media from {path}')
                if not color and img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif color and img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return [img]
            finally:
                cap.release()

        self.img1_list = _load_media_sequence(img1_path)
        self.img2_list = _load_media_sequence(img2_path)
        if img_rgb_path is not None:
            self.img_rgb_list = _load_media_sequence(img_rgb_path,color=True)
        else:
            self.img_rgb_list = None

        # self.img1 = self.img1_list[0]
        # self.img2 = self.img2_list[0]

        self.img1_rectify_list = []
        self.img2_rectify_list = []

        self.max_dis = 1500
        self.min_dis = 100
        # 最近最远深度

        self.binary_image_threshold = [19,-4]
        # 自适应二值化的两个参数
        self.contourArea_threshold = 30
        # 过滤光斑的参数
        self.lightfilter_threshold = 0
        # 按照亮度过滤的参数

        self.method = None

        # 初始化颜色映射器 - DX
        self.color_mapper = ColorMapper()

    def import_biaodin(self,extri_path,intri_path,idx = 0):
        current_img1 = self.img1_list[idx]
        current_img2 = self.img2_list[idx]
        

        extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
        intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)

        M1 = intri.getNode('M1').mat()
        M2 = intri.getNode('M2').mat()
        D1 = intri.getNode('D1').mat()
        D2 = intri.getNode('D2').mat()

        R = extri.getNode('R').mat()
        t = extri.getNode('T').mat()
        t_cross = np.array([[0, -t[2][0], t[1][0]],
                            [t[2][0], 0, -t[0][0]],
                            [-t[1][0], t[0][0], 0]])

        F = np.dot(np.dot(np.transpose(np.linalg.inv(M2)), np.dot(t_cross, R)), np.linalg.inv(M1))


        h, w = current_img1.shape[:2]
        new_M1, roi1 = cv2.getOptimalNewCameraMatrix(M1, D1, (w, h), 1, (w, h))
        new_M2, roi2 = cv2.getOptimalNewCameraMatrix(M2, D2, (w, h), 1, (w, h))

        self.K1 = new_M1
        self.K2 = new_M2

        self.cam_R = R
        self.cam_t = t

        # 计算极线变换矩阵
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(new_M1, D1, new_M2, D2, (w, h), R, t, flags=cv2.CALIB_ZERO_TANGENT_DIST)

        self.P1 = P1
        self.P2 = P2

        self.Q = Q


        # 生成映射
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(new_M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(new_M2, D2, R2, P2, (w, h), cv2.CV_32FC1)

        # 进行矫正
        left_rectified = cv2.remap(current_img1, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(current_img2, right_map_x, right_map_y, cv2.INTER_LINEAR)

        self.img1_rectify = left_rectified
        self.img2_rectify = right_rectified
        self.img1_rectify_list.append(self.img1_rectify)
        self.img2_rectify_list.append(self.img2_rectify)

        f = float(self.K1[0,0])
        B = abs(float(self.cam_t[0][0]))
        self.max_disp = f * B / self.min_dis  # for closest points
        self.min_disp = f * B / self.max_dis  # for farthest points

    def forward_disparity(self):
        if self.method == 'sgbm':
            disparity_raw = self.sgbm()
            return disparity_raw
        elif self.method == 'foundation_stereo':
            disparity_raw = self.foundation_stereo()
            return disparity_raw
        elif self.method == 'bridgedepth':
            disparity_raw = self.bridgedepth_stereo_matching()
            return disparity_raw

    def init_model(self):
        if self.method == 'sgbm':
            self.processor = SGBM(base=self)

        elif self.method == 'foundation_stereo':
            self.processor = StereoInference("jiegouguang/weights/FoundationStereo/23-51-11/model_best_bp2.pth")
        elif self.method == 'bridgedepth':
            checkpoint_path="jiegouguang/weights/BridgeDepth/bridge_rvc.pth"
            model_name='rvc'
            device='cuda'
            self.processor = BridgeDepthStereo(
                checkpoint_path=checkpoint_path,
                model_name=model_name, 
                device=device
            )


    def manual_feature_extracting(self):
        processor = ManualFeatureJieGouGuang(base=self)

        img1_with_center,img2_with_center = processor.extract_circle()
        processor.feature_extracting()
        processor.feature_matching()
        pcd = processor.pointcloud_from_disparity()

        return pcd
    

    def sgbm(self):
        # processor = SGBM(base=self)

        pcd = self.processor.sgbm(build_pointcloud = False)
        self.pcd = pcd
        return pcd
    

    def foundation_stereo(self):
        

        results = self.processor.infer(self.img1_rectify, self.img2_rectify, self.K1, abs(float(self.cam_t[0])))
        # pcd_np = results['points_cam1'].astype(np.float64).reshape(-1, 3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcd_np)
        # self.pcd = pcd
        return results['disparity']


    def lg_feature_extracting(self):
        """
        使用神经网络（SuperPoint + LightGlue）进行特征提取和匹配

        Returns:
            pcd: Open3D 点云对象
        """
        processor = NeuralFeatureJieGouGuang(base=self)

        img1_with_center, img2_with_center = processor.extract_circle()
        processor.feature_extracting()
        processor.feature_matching()
        pcd = processor.pointcloud_from_disparity()

        self.pcd = pcd
        return pcd


    def bridgedepth_stereo_matching(
        self
        # z_min=0.1,
        # z_max=10.0,
    ):
        """
        使用 BridgeDepth 深度学习进行立体匹配

        需要先调用 import_biaodin() 导入标定参数

        参数:
            checkpoint_path: 模型权重路径（可选）
            model_name: 预训练模型名称（默认: rvc_pretrain）
            device: 设备 ('cuda' 或 'cpu')
            z_min: 最小深度（米）
            z_max: 最大深度（米）
            baseline: 基线距离（米），如果为 None 则从标定参数计算

        返回:
            results: 字典，包含:
                - 'disparity': 视差图 (H, W)
                - 'depth': 深度图 (H, W) 米
                - 'xyz_map': XYZ坐标图 (H, W, 3) 米
                - 'pointcloud': Open3D点云对象
        """
        
        z_min = self.min_dis
        z_max = self.max_dis
        baseline = abs(float(self.cam_t[0]))


        # 初始化 BridgeDepthStereo（第一步）


        # 执行推理（第二步）
        results = self.processor.infer(
            left_image=self.img1_rectify,
            right_image=self.img2_rectify,
            K=self.K1,
            baseline=baseline,
            z_min=z_min,
            z_max=z_max,
            return_pointcloud=False
        )


        return results['disparity']




    def cal_error(self,pcd):

        points = np.asarray(pcd.points)
        if points.shape[0] < 3:
            raise ValueError('平面拟合至少需要三个点')
        # 使用 PCA 拟合点云平面
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)

        # 平面方程：normal · (x - centroid) = 0，可写成 normal · x + d = 0
        d = -np.dot(normal, centroid)

        distances = centered @ normal
        abs_dist = np.abs(distances)
        abs_dist[abs_dist>50] = 0
        rms_error = np.sqrt(np.mean(distances ** 2))

        print(f'Plane normal: {normal}')
        print(f'Plane offset d: {d:.6f}')
        print(f'Mean abs distance: {abs_dist.mean():.6f}')
        print(f'RMS distance: {rms_error:.6f}')
        print(f'Max abs distance: {abs_dist.max():.6f}')


    def depth2pointcloud(self,depth):
        """
        将深度图转换为 Open3D 点云。深度值与 self.K1 对应的相机坐标系一致。

        Args:
            depth (np.ndarray): 单通道深度图，单位与外部流程保持一致。

        Returns:
            o3d.geometry.PointCloud: 生成的点云。
        """

        if depth is None:
            raise ValueError('depth cannot be None')
        if not hasattr(self, 'K1'):
            raise ValueError('camera intrinsics (self.K1) are not set, call import_biaodin() first')
        if depth.ndim != 2:
            raise ValueError('depth must be a single-channel image')

        fx = float(self.K1[0, 0])
        fy = float(self.K1[1, 1])
        cx = float(self.K1[0, 2])
        cy = float(self.K1[1, 2])
        if fx == 0 or fy == 0:
            raise ValueError('invalid camera intrinsics: focal length cannot be zero')

        depth = depth.astype(np.float64)
        rows, cols = np.indices(depth.shape)
        valid_mask = np.isfinite(depth) & (depth > 0)
        if hasattr(self, 'max_dis'):
            valid_mask &= depth <= float(self.max_dis)

        if not np.any(valid_mask):
            return o3d.geometry.PointCloud()

        z = depth[valid_mask]
        u = cols[valid_mask]
        v = rows[valid_mask]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 使用对应像素灰度作为伪彩色，方便可视化
        base_img = getattr(self, 'img1_rectify', getattr(self, 'img1', None))
        if base_img is not None and base_img.shape[:2] == depth.shape:
            if base_img.ndim == 2:
                colors = base_img[valid_mask].astype(np.float64) / 255.0
                colors = np.repeat(colors[:, None], 3, axis=1)
            else:
                colors = base_img[valid_mask].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        self.pcd = pcd
        return pcd

    def get_rgbd(self, depth_L: np.ndarray, rgb_img: np.ndarray) -> np.ndarray:
        """
        获取RGBD图（四通道） - DX

        参数:
            depth_L: 深度图 HxW (float32, 单位:米)
            rgb_img: RGB图像 HxWx3 (uint8)

        返回:
            rgbd: HxWx4 数组
                  - rgbd[:,:,0:3] = RGB颜色 (uint8)
                  - rgbd[:,:,3] = 深度值 (float32, 米)
        """
        return self.color_mapper.get_color(depth_L, rgb_img, self.K1)
