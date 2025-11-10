"""
BridgeDepthStereo - 深度学习立体匹配类

作者: DX
创建时间: 2025-11-07
用途: 将 BridgeDepth 封装为两步推理接口

使用方法:
    # 第一步: 初始化并加载模型权重
    stereo = BridgeDepthStereo(
        checkpoint_path='checkpoints/bridge_rvc_pretrain.pth',
        device='cuda'
    )

    # 第二步: 输入图像和相机参数，获取结果
    results = stereo.infer(
        left_image=img1,    # (H, W, 3) NumPy 数组
        right_image=img2,   # (H, W, 3) NumPy 数组
        K=K_matrix,         # (3, 3) 内参矩阵
        baseline=0.095      # 基线 (米)
    )

    # 返回结果 (在第一个相机坐标系下):
    # - results['disparity']: 视差图 (H, W)
    # - results['depth']: 深度图 (H, W)，单位: 米
    # - results['pointcloud']: Open3D 点云对象
"""

import os
import numpy as np
import torch
import open3d as o3d
from typing import Dict, Optional, Union

from bridgedepth.bridgedepth import BridgeDepth
from bridgedepth.utils.logger import setup_logger


class BridgeDepthStereo:
    """
    BridgeDepth 深度学习立体匹配类

    设计理念:
        - 第一步 (__init__): 加载神经网络模型并读取权重
        - 第二步 (infer): 输入图像和相机参数，返回点云、深度图、视差图

    特点:
        - 基于 demo.py 的代码逻辑
        - 不缩放图像，保持原始分辨率
        - 支持灰度图（自动转换为 RGB）
        - 返回第一个相机坐标系下的结果
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = 'rvc_pretrain',
        device: str = 'cuda',
        logger_name: str = 'BridgeDepthStereo'
    ):
        """
        初始化 BridgeDepthStereo 类

        第一步: 构造神经网络主类并读取权重

        参数:
            checkpoint_path: 模型权重路径 (绝对路径或相对路径)
                           如果为 None，使用 model_name 加载预设模型
            model_name: 预训练模型名称，可选:
                       - 'rvc_pretrain' (推荐，适合红外结构光)
                       - 'rvc'
                       - 'sf', 'l_sf'
                       - 'kitti'
                       - 'eth3d_pretrain', 'eth3d'
                       - 'middlebury_pretrain', 'middlebury'
            device: 计算设备 ('cuda' 或 'cpu')
            logger_name: 日志记录器名称

        异常:
            FileNotFoundError: 如果 checkpoint_path 指定的文件不存在
            RuntimeError: 如果 CUDA 不可用但指定了 device='cuda'
        """
        # 设置日志
        self.logger = setup_logger(name=logger_name)

        # 检查设备
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA 不可用，回退到 CPU")
            device = 'cpu'
        self.device = torch.device(device)

        # 确定模型路径
        pretrained_model_name_or_path = model_name
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"模型权重文件不存在: {checkpoint_path}")
            pretrained_model_name_or_path = checkpoint_path

        # 加载模型
        self.logger.info(f"加载模型: {pretrained_model_name_or_path}")
        self.model = BridgeDepth.from_pretrained(pretrained_model_name_or_path)
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

        self.logger.info(f"模型已加载到 {self.device}")

    def infer(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        K: np.ndarray,
        baseline: float,
        z_min: float = 0.1,
        z_max: float = 10.0,
        return_pointcloud: bool = True
    ) -> Dict[str, Union[np.ndarray, o3d.geometry.PointCloud]]:
        """
        执行立体匹配推理

        第二步: 输入图像和相机参数，返回点云、深度图、视差图

        参数:
            left_image: 左图 (H, W, 3) 或 (H, W)，NumPy 数组
                       - RGB 图像: 值域 [0, 255]
                       - 灰度图像: 自动转换为 RGB
            right_image: 右图 (H, W, 3) 或 (H, W)，格式同 left_image
            K: 相机内参矩阵 (3, 3)，格式:
               [[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]]
            baseline: 双目基线 (米)
            z_min: 最小有效深度 (米)，默认 0.1m
            z_max: 最大有效深度 (米)，默认 10m，用于点云裁剪
            return_pointcloud: 是否生成点云对象，默认 True

        返回:
            results: 字典，包含:
                - 'disparity': 视差图 (H, W)，单位: 像素
                - 'depth': 深度图 (H, W)，单位: 米
                - 'pointcloud': Open3D 点云对象 (如果 return_pointcloud=True)
                - 'xyz_map': 3D 坐标图 (H, W, 3)，单位: 米

        注意:
            - 所有结果都在第一个相机（左相机）坐标系下
            - 图像不会被缩放，保持输入分辨率
            - 灰度图会被自动复制为 3 通道 RGB
        """
        # ========== 输入验证 ==========
        assert left_image.shape[:2] == right_image.shape[:2], \
            f"左右图像分辨率不一致: {left_image.shape[:2]} vs {right_image.shape[:2]}"

        H, W = left_image.shape[:2]
        self.logger.info(f"输入图像分辨率: {W}×{H}")

        # ========== 处理灰度图 ==========
        # 如果是灰度图 (H, W)，复制为 RGB (H, W, 3)
        if left_image.ndim == 2:
            self.logger.info("检测到灰度图，转换为 RGB")
            left_image = np.tile(left_image[..., None], (1, 1, 3))
        if right_image.ndim == 2:
            right_image = np.tile(right_image[..., None], (1, 1, 3))

        # ========== 准备神经网络输入 ==========
        # NumPy (H, W, C) → Tensor (1, C, H, W)
        # 关键步骤:
        # 1. torch.as_tensor(): NumPy → Tensor (不复制数据)
        # 2. .to(device): 转移到 GPU/CPU
        # 3. .float(): 转换为 float32
        # 4. [None]: 添加批次维度 (1, H, W, C)
        # 5. .permute(0, 3, 1, 2): 转换为 PyTorch 格式 (1, C, H, W)
        sample = {
            'img1': torch.as_tensor(left_image).to(self.device).float()[None].permute(0, 3, 1, 2),
            'img2': torch.as_tensor(right_image).to(self.device).float()[None].permute(0, 3, 1, 2),
        }

        # ========== 神经网络推理 ==========
        # 注意: BridgeDepth 不需要相机参数！
        # 输入: 左右图像对 (1, 3, H, W)
        # 输出: 视差图 (像素单位)
        self.logger.info("执行神经网络推理...")
        with torch.no_grad():  # 禁用梯度计算 (推理模式)
            results_dict = self.model(sample)

        # 提取视差预测结果
        # clamp_min(1e-3): 限制最小视差为 0.001 (避免除零)
        disparity = results_dict['disp_pred'].clamp_min(1e-3).cpu().numpy().reshape(H, W)
        self.logger.info(f"视差范围: {disparity.min():.2f} - {disparity.max():.2f} 像素")

        # ========== 视差 → 深度转换 ==========
        # 核心公式: Z = (fx × B) / d
        # 输入:
        #   - fx: X方向焦距 (像素)
        #   - B: 基线 (米)
        #   - d: 视差 (像素)
        # 输出:
        #   - Z: 深度 (米)
        depth = K[0, 0] * baseline / disparity
        self.logger.info(f"深度范围: {depth.min():.3f} - {depth.max():.3f} 米")

        # 准备返回结果
        results = {
            'disparity': disparity,
            'depth': depth,
        }

        # ========== 深度图 → 3D 点云 (可选) ==========
        if return_pointcloud:
            self.logger.info("生成 3D 点云...")

            # 步骤 1: 深度图 → XYZ 坐标图
            xyz_map = self._depth2xyzmap(depth, K, z_min=z_min)
            results['xyz_map'] = xyz_map

            # 步骤 2: 创建 Open3D 点云对象
            pcd = self._to_open3d_cloud(
                points=xyz_map.reshape(-1, 3),
                colors=left_image.reshape(-1, 3)
            )

            # 步骤 3: 点云过滤 (深度范围)
            # 保留条件:
            #   1. Z > z_min (有效深度)
            #   2. Z <= z_max (不超过最大深度)
            points = np.asarray(pcd.points)
            keep_mask = (points[:, 2] > z_min) & (points[:, 2] <= z_max)
            keep_ids = np.arange(len(points))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)

            self.logger.info(f"点云过滤: 保留 {len(pcd.points)} / {H*W} 个点")
            results['pointcloud'] = pcd

        return results

    @staticmethod
    def _depth2xyzmap(depth: np.ndarray, K: np.ndarray, z_min: float = 0.1) -> np.ndarray:
        """
        将深度图转换为 3D 点云坐标图

        核心原理 - 针孔相机模型的逆投影:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth

        参数:
            depth: 深度图 (H, W)，单位: 米
            K: 相机内参矩阵 (3×3)
            z_min: 最小有效深度 (米)

        返回:
            xyz_map: 3D 坐标图 (H, W, 3)，每个像素对应 [X, Y, Z] 坐标
        """
        # 标记无效深度点
        invalid_mask = (depth < z_min)
        H, W = depth.shape[:2]

        # 生成像素坐标网格
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)

        # 提取深度值
        zs = depth[vs, us]

        # ===== 核心公式: 2D像素 + 深度 → 3D空间 =====
        xs = (us - K[0, 2]) * zs / K[0, 0]  # X = (u - cx) * Z / fx
        ys = (vs - K[1, 2]) * zs / K[1, 1]  # Y = (v - cy) * Z / fy

        # 组装点云数组
        pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)

        # 初始化 XYZ 图
        xyz_map = np.zeros((H, W, 3), dtype=np.float32)
        xyz_map[vs, us] = pts

        # 将无效点置零
        if invalid_mask.any():
            xyz_map[invalid_mask] = 0

        return xyz_map

    @staticmethod
    def _to_open3d_cloud(
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
        """
        将 NumPy 数组转换为 Open3D 点云对象

        参数:
            points: (N, 3) 点云坐标数组
            colors: (N, 3) 可选的 RGB 颜色数组，值域 [0, 255] 或 [0, 1]
            normals: (N, 3) 可选的法向量数组

        返回:
            cloud: Open3D 点云对象
        """
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        if colors is not None:
            # 归一化颜色到 [0, 1]
            if colors.max() > 1:
                colors = colors / 255.0
            cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        if normals is not None:
            cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        return cloud

    def save_results(
        self,
        results: Dict,
        output_dir: str,
        save_ply: bool = True,
        save_depth: bool = True,
        save_disparity: bool = True
    ):
        """
        保存推理结果到文件

        参数:
            results: infer() 返回的结果字典
            output_dir: 输出目录路径
            save_ply: 是否保存点云 (.ply)
            save_depth: 是否保存深度图 (.npy)
            save_disparity: 是否保存视差图 (.npy)
        """
        os.makedirs(output_dir, exist_ok=True)

        if save_disparity and 'disparity' in results:
            disp_path = os.path.join(output_dir, 'disparity.npy')
            np.save(disp_path, results['disparity'])
            self.logger.info(f"视差图已保存: {disp_path}")

        if save_depth and 'depth' in results:
            depth_path = os.path.join(output_dir, 'depth.npy')
            np.save(depth_path, results['depth'])
            self.logger.info(f"深度图已保存: {depth_path}")

        if save_ply and 'pointcloud' in results:
            ply_path = os.path.join(output_dir, 'pointcloud.ply')
            o3d.io.write_point_cloud(ply_path, results['pointcloud'])
            self.logger.info(f"点云已保存: {ply_path}")
