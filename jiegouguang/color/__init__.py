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
        """
        初始化：加载RGB标定参数

        参数:
            yaml_path: rgb_calib.yaml 路径，默认为同目录下的文件
        """
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'rgb_calib.yaml')

        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

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

    def get_color(self,
                  depth_L: np.ndarray,
                  rgb_img: np.ndarray,
                  K_L: np.ndarray) -> np.ndarray:
        """
        深度图 + RGB → RGBD（四通道）

        参数:
            depth_L: 左目深度图 HxW (float32, 单位:米)
            rgb_img: RGB图像 HxWx3 (uint8)
            K_L: 左目内参 3x3

        返回:
            rgbd: HxWx4 数组
                  - rgbd[:,:,0:3] = RGB颜色 (uint8)
                  - rgbd[:,:,3] = 深度值 (float32, 米)
        """
        H, W = depth_L.shape

        # 初始化RGBD（float32存储，方便D通道）
        rgbd = np.zeros((H, W, 4), dtype=np.float32)
        rgbd[:, :, 3] = depth_L  # D通道

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

        # 边界检查（双线性插值需要访问相邻像素）
        h_c, w_c = rgb_img.shape[:2]
        inside = valid_z & (u_C >= 0) & (u_C < w_c - 1) & (v_C >= 0) & (v_C < h_c - 1)

        if not np.any(inside):
            return rgbd

        # 双线性插值取色
        u_C_in = u_C[inside]
        v_C_in = v_C[inside]

        u0 = np.floor(u_C_in).astype(np.int32)
        v0 = np.floor(v_C_in).astype(np.int32)
        u1, v1 = u0 + 1, v0 + 1

        du = (u_C_in - u0)[:, None]
        dv = (v_C_in - v0)[:, None]

        colors = (
            (1 - du) * (1 - dv) * rgb_img[v0, u0].astype(np.float32) +
            du * (1 - dv) * rgb_img[v0, u1].astype(np.float32) +
            (1 - du) * dv * rgb_img[v1, u0].astype(np.float32) +
            du * dv * rgb_img[v1, u1].astype(np.float32)
        )

        # 写入RGBD的RGB通道
        v_L_in = v_valid[inside].astype(np.int32)
        u_L_in = u_valid[inside].astype(np.int32)
        rgbd[v_L_in, u_L_in, :3] = colors

        return rgbd
