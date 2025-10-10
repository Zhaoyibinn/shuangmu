# 双目立体视觉处理模块（使用LightGlue特征匹配）
# 实现基于SuperPoint + LightGlue的高精度特征匹配和三角测量三维重建

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# 导入LightGlue相关模块
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """
    将numpy图像转换为Torch张量并正规化
    重新排列维度以符合PyTorch的输入格式

    Args:
        image: 输入的numpy图像数组

    Returns:
        torch.Tensor: 正规化后的张量，数值范围[0,1]

    Raises:
        ValueError: 如果输入不是有效的图像格式
    """
    if image.ndim == 3:
        # HxWxC 转换为 CxHxW
        image = image.transpose((2, 0, 1))
    elif image.ndim == 2:
        # 添加通道维度
        image = image[None]
    else:
        raise ValueError(f"Not an image: {image.shape}")

    # 正规化像素值到[0,1]范围并转换为float类型
    return torch.tensor(image / 255.0, dtype=torch.float)

# 设备配置：优先使用GPU，如果不可用则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化SuperPoint特征提取器，最多提取2048个特征点
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

# 初始化LightGlue匹配器，与SuperPoint特征配合使用
matcher = LightGlue(features="superpoint").eval().to(device)


if __name__ == "__main__":
    """
    主程序：演示双目特征匹配和三角测量的完整流程
    """
    # 设定输入图像路径
    left_img_path = "realsense_capture/color_2_375.png"
    right_img_path = "realsense_capture/color_1_375.png"

    # 备用图像路径（已注释）
    # left_img_path = "realsense_capture/color_1_1042.png"
    # right_img_path = "realsense_capture/depth_2_1042.png"

    # 读取左右图像并转换颜色空间
    left_img = cv2.cvtColor(cv2.imread(left_img_path), cv2.COLOR_RGB2BGR)
    right_img = cv2.cvtColor(cv2.imread(right_img_path), cv2.COLOR_RGB2BGR)

    # 将numpy图像转换为Torch张量
    image0 = numpy_image_to_torch(left_img)
    image1 = numpy_image_to_torch(right_img)

    # 使用SuperPoint提取特征
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    # 使用LightGlue进行特征匹配
    matches01 = matcher({"image0": feats0, "image1": feats1})

    # 移除batch维度
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]

    # 提取关键点和匹配结果
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # 可视化匹配结果
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.save_plot("test.png")

    # 读取双目相机标定参数
    extri = cv2.FileStorage('extrinsics_realsense_20240429.yml', cv2.FILE_STORAGE_READ)
    intri = cv2.FileStorage('intrinsics_realsense_20240429.yml', cv2.FILE_STORAGE_READ)

    # 获取内参矩阵和畸变系数
    M1 = intri.getNode('M1').mat()  # 左相机内参矩阵
    M2 = intri.getNode('M2').mat()  # 右相机内参矩阵
    D1 = intri.getNode('D1').mat()  # 左相机畸变系数
    D2 = intri.getNode('D2').mat()  # 右相机畸变系数

    # 获取外参矩阵
    R = extri.getNode('R').mat()    # 旋转矩阵
    t = extri.getNode('T').mat()    # 平移向量

    # 计算反对称矩阵（用于基础矩阵计算）
    t_cross = np.array([[0, -t[2][0], t[1][0]],
                        [t[2][0], 0, -t[0][0]],
                        [-t[1][0], t[0][0], 0]])

    # 计算基础矩阵F
    F = np.dot(np.dot(np.transpose(np.linalg.inv(M2)), np.dot(t_cross, R)), np.linalg.inv(M1))

    # 计算投影矩阵（用于三角测量）
    P1 = np.dot(M1, np.hstack((np.eye(3), np.zeros((3, 1)))))  # 左相机投影矩阵
    P2 = np.dot(M2, np.hstack((R, t)))                         # 右相机投影矩阵

    # 使用三角测量计算三维点坐标
    points_4d = cv2.triangulatePoints(P1, P2, np.array(m_kpts0.cpu()).T, np.array(m_kpts1.cpu()).T)
    # 齐次坐标转欧几里得坐标
    points_3d = points_4d[:3, :] / points_4d[3, :]

    # 创建点云对象并进行可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)

    # 使用Open3D显示三维点云
    o3d.visualization.draw_geometries([pcd])

    # 可选：打印部分三维点坐标（已注释）
    # print("部分三维点坐标：")
    # print(points_3d[:, :5].T)


def click_event(event, x, y, flags, param):
    """
    鼠标点击事件处理函数（未使用）
    在左图中点击时计算并绘制对应的极线

    注意：此函数依赖于全局变量，需要在主程序中定义相关变量
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将点击坐标转换为原始坐标
        x = x / scale
        y = y / scale
        point1 = np.array([[x, y, 1]]).T

        print(f"clicked {x},{y}")

        # 使用基础矩阵计算极线
        line = np.dot(F, point1)
        # 极线正规化
        line = line / line[2]

        # 找到极线与图像边界的交点
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [right_img.shape[1], -(line[2] + line[0] * right_img.shape[1]) / line[1]])

        # 在右图中绘制极线
        cv2.line(right_img_scaled, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)), (0, 0, 255), 2)

        # 拼接左右图像并显示
        stitched_image = np.hstack((left_img_scaled, right_img_scaled))
        cv2.imshow('Image1', stitched_image)

        return right_img
    

# 以下代码用于交互式极线验证（已注释）
# stitched_image = np.hstack((left_img_scaled, right_img_scaled))
# cv2.imshow('Image1', stitched_image)
# cv2.setMouseCallback('Image1', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 更多可视化选项（已注释）
# stitched_image = np.hstack((left_img, right_img))
# plt.imshow(stitched_image)
# plt.show()
    

def shuangmu_fun(img1, img2):
    """
    双目立体视觉处理主函数
    对输入的左右图像进行特征匹配和三角测量重建

    Args:
        img1: 左相机图像（numpy数组）
        img2: 右相机图像（numpy数组）

    Returns:
        int: 返回0表示处理完成
    """
    # 将输入图像转换为Torch张量
    image0 = numpy_image_to_torch(img1)
    image1 = numpy_image_to_torch(img2)

    # 提取特征
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    # 特征匹配
    matches01 = matcher({"image0": feats0, "image1": feats1})

    # 移除batch维度
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]

    # 提取匹配的关键点
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # 保存匹配可视化结果
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.save_plot("test.png")

    # 读取双目相机标定参数
    extri = cv2.FileStorage('extrinsics_realsense.yml', cv2.FILE_STORAGE_READ)
    intri = cv2.FileStorage('intrinsics_realsense.yml', cv2.FILE_STORAGE_READ)

    # 获取内参和外参
    M1 = intri.getNode('M1').mat()
    M2 = intri.getNode('M2').mat()
    D1 = intri.getNode('D1').mat()
    D2 = intri.getNode('D2').mat()

    R = extri.getNode('R').mat()
    t = extri.getNode('T').mat()

    # 计算基础矩阵（未使用）
    t_cross = np.array([[0, -t[2][0], t[1][0]],
                        [t[2][0], 0, -t[0][0]],
                        [-t[1][0], t[0][0], 0]])
    F = np.dot(np.dot(np.transpose(np.linalg.inv(M2)), np.dot(t_cross, R)), np.linalg.inv(M1))

    # 计算投影矩阵
    P1 = np.dot(M1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(M2, np.hstack((R, t)))

    # 三角测量重建三维点
    points_4d = cv2.triangulatePoints(P1, P2, np.array(m_kpts0.cpu()).T, np.array(m_kpts1.cpu()).T)
    points_3d = points_4d[:3, :] / points_4d[3, :]

    # 创建并显示点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    o3d.visualization.draw_geometries([pcd])

    # 可选：打印点云信息（已注释）
    # print("部分三维点坐标：")
    # print(points_3d[:, :5].T)

    return 0