"""
颜色映射模块：深度图 + RGB → RGBD（四通道）
作者: DX
日期: 2025-11-21
"""
import numpy as np
import yaml
import os


def project_depth_to_rgb(depth, source_k, rotation, translation, rgb_k, rgb_size):
    """Project pinhole depth to RGB, retaining the nearest target-camera Z.

    ``rotation @ source_point + translation`` maps source to RGB coordinates.
    Depth and translation must use the same length unit; output keeps that unit.
    Both images must already be undistorted. Unobserved pixels remain zero.
    """
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[v, u].astype(np.float64)
    points = np.stack(((u - source_k[0, 2]) * z / source_k[0, 0],
                       (v - source_k[1, 2]) * z / source_k[1, 1], z))
    points = rotation @ points + np.asarray(translation).reshape(3, 1)
    good = np.isfinite(points).all(axis=0) & (points[2] > 0)
    points = points[:, good]
    width, height = rgb_size
    x = np.rint(rgb_k[0, 0] * points[0] / points[2] + rgb_k[0, 2]).astype(np.int64)
    y = np.rint(rgb_k[1, 1] * points[1] / points[2] + rgb_k[1, 2]).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    buffer = np.full(height * width, np.inf)
    # Occlusion: keep the nearest RGB-camera Z when multiple samples hit a pixel.
    np.minimum.at(buffer, y[inside] * width + x[inside], points[2, inside])
    return np.where(np.isfinite(buffer), buffer, 0).reshape(height, width).astype(np.float32)


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

        h_c, w_c = rgb_img.shape[:2]
        rgbd = np.zeros((h_c, w_c, 4), dtype=np.float32)
        rgbd[:, :, :3] = rgb_img.astype(np.float32)
        rgbd[:, :, 3] = project_depth_to_rgb(
            depth_L, K_L, self.R_L2RGB, self.t_L2RGB, self.K_RGB, (w_c, h_c))
        return rgbd
