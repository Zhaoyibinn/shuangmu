"""
颜色映射模块：深度图 + RGB → RGBD（四通道）
作者: DX
日期: 2025-11-21
"""
import numpy as np
import yaml
import os


class ColorMapper:
    """RGB颜色映射器 - 将深度图染色为RGBD"""

    def __init__(self, yaml_path: str = None):
        # self.yaml_path = None
        # self.K_RGB = None
        # self.R_L2RGB = None
        # self.t_L2RGB = None

        if yaml_path is not None:
            self.load_calibration(yaml_path)

    def load_calibration(self, yaml_path: str) -> None:
        """
        加载RGB标定参数

        参数:
            yaml_path: rgb_calib.yaml 路径
        """
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.yaml_path = yaml_path

        # RGB内参
        intr = cfg['rgb_intrinsics']
        self.K_RGB = np.array([
            [intr['fx'], 0, intr['cx']],
            [0, intr['fy'], intr['cy']],
            [0, 0, 1]
        ], dtype=np.float32)

        # 外参
        extr = cfg['rgb_extrinsics']
        self.R_L2RGB = np.array(extr['rotation'], dtype=np.float32).reshape(3, 3)
        self.t_L2RGB = np.array(extr['translation'], dtype=np.float32).reshape(3, 1)
        self.R_L2RGB = self.R_L2RGB.T
        # self.t_L2RGB = -self.R_L2RGB.T @ self.t_L2RGB

    def get_color(self,
                  depth_L: np.ndarray,
                  rgb_img: np.ndarray,
                  K_L: np.ndarray) -> np.ndarray:
        """
        深度图对齐到RGB坐标系 → RGBD（四通道）

        参数:
            depth_L: 左目深度图 HxW (float32, 单位:米)
            rgb_img: RGB图像 HxWx3 (uint8)
            K_L: 左目内参 3x3

        返回:
            rgbd: HxWx4 数组（RGB图像尺寸）
                  - rgbd[:,:,0:3] = RGB颜色 (uint8)
                  - rgbd[:,:,3] = 深度值 (float32, 米)
        """
        # if self.K_RGB is None or self.R_L2RGB is None or self.t_L2RGB is None:
        #     raise ValueError('RGB calibration is not loaded, call load_calibration() first')

        H, W = depth_L.shape
        h_c, w_c = rgb_img.shape[:2]
        # 初始化RGBD（float32存储，方便D通道）
        rgbd = np.zeros((h_c, w_c, 4), dtype=np.float32)
        rgbd[:, :, :3] = rgb_img.astype(np.float32)

        # 有效深度掩码
        valid = depth_L > 0
        if not np.any(valid):
            return rgbd

        # 像素坐标网格
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u_valid = u[valid].astype(np.float32)
        v_valid = v[valid].astype(np.float32)
        Z = depth_L[valid]

        # Step 1: 反投影 (左目像素+深度 → 左目3D)
        fx_L, fy_L = K_L[0, 0], K_L[1, 1]
        cx_L, cy_L = K_L[0, 2], K_L[1, 2]
        X_L = (u_valid - cx_L) * Z / fx_L
        Y_L = (v_valid - cy_L) * Z / fy_L
        pts_L = np.stack([X_L, Y_L, Z], axis=0)  # 3xN

        # Step 2: 坐标变换 (左目3D → RGB相机3D)
        pts_C = self.R_L2RGB @ pts_L + self.t_L2RGB
        X_C, Y_C, Z_C = pts_C

        # Step 3: 投影 (RGB相机3D → RGB像素)
        fx_C, fy_C = self.K_RGB[0, 0], self.K_RGB[1, 1]
        cx_C, cy_C = self.K_RGB[0, 2], self.K_RGB[1, 2]

        # 过滤Z_C <= 0的点
        valid_z = Z_C > 0
        u_C = np.full_like(X_C, -1)
        v_C = np.full_like(Y_C, -1)
        u_C[valid_z] = fx_C * X_C[valid_z] / Z_C[valid_z] + cx_C
        v_C[valid_z] = fy_C * Y_C[valid_z] / Z_C[valid_z] + cy_C

        # 边界检查（深度对齐到RGB像素）
        u_C_round = np.round(u_C).astype(np.int32)
        v_C_round = np.round(v_C).astype(np.int32)
        inside = valid_z & (u_C_round >= 0) & (u_C_round < w_c) & (v_C_round >= 0) & (v_C_round < h_c)

        if not np.any(inside):
            return rgbd

        # 深度写入RGB像素（使用Z_C作为RGB相机坐标系深度）
        u_in = u_C_round[inside]
        v_in = v_C_round[inside]
        z_in = Z_C[inside]

        depth_rgb = np.full((h_c, w_c), np.inf, dtype=np.float32)
        flat_idx = v_in * w_c + u_in
        depth_flat = depth_rgb.reshape(-1)
        np.minimum.at(depth_flat, flat_idx, z_in)
        depth_rgb = depth_flat.reshape(h_c, w_c)
        depth_rgb[~np.isfinite(depth_rgb)] = 0

        rgbd[:, :, 3] = depth_rgb

        return rgbd
