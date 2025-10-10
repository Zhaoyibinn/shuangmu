# 双目立体视觉测试程序
# 使用预先标定好的双目相机参数进行图像矯正和极线验证

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# 设定输入图像路径
left_img_path = "whx_biaoding/L/left_0041.png"
right_img_path = "whx_biaoding/right_0090.png"

# 读取左右图像
left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)

# 图像缩放配置
# 设定目标宽度
target_width = 640
# 计算缩放比例
scale = target_width / float(left_img.shape[1])
# 计算目标高度
target_height = int(left_img.shape[0] * scale)

# 缩放图像到目标尺寸
left_img_scaled = cv2.resize(left_img, (target_width, target_height))
right_img_scaled = cv2.resize(right_img, (target_width, target_height))

# 读取双目相机标定参数
extri = cv2.FileStorage('biaoding/extrinsics_whx_zyb.yml', cv2.FILE_STORAGE_READ)
intri = cv2.FileStorage('biaoding/intrinsics_whx_zyb.yml', cv2.FILE_STORAGE_READ)

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

def rectify_images(left_img, right_img, M1, M2, R, t):
    """
    对双目图像进行立体校正
    使左右图像的极线平行，便于后续的立体匹配

    Args:
        left_img: 左图像
        right_img: 右图像
        M1, M2: 左右相机内参矩阵
        R, t: 外参矩阵（旋转和平移）

    Returns:
        tuple: 校正后的左右图像
    """
    # 计算新的相机矩阵和ROI区域
    h, w = left_img.shape[:2]
    new_M1, roi1 = cv2.getOptimalNewCameraMatrix(M1, D1, (w, h), 1, (w, h))
    new_M2, roi2 = cv2.getOptimalNewCameraMatrix(M2, D2, (w, h), 1, (w, h))

    # 计算立体校正的变换矩阵
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(new_M1, D1, new_M2, D2, (w, h), R, t, flags=cv2.CALIB_ZERO_TANGENT_DIST)

    # 生成重映射表
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(new_M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(new_M2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    # 应用校正
    left_rectified = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

    return left_rectified, right_rectified

# 使用内参进行立体校正
left_img_rectified, right_img_rectified = rectify_images(left_img, right_img, M1, M2, R, t)

# 缩放校正后的图像以便显示
left_img_rectified_scaled = cv2.resize(left_img_rectified, (target_width, target_height))
right_img_rectified_scaled = cv2.resize(right_img_rectified, (target_width, target_height))

# 以下代码用于棋盘格角点检测和三角测量（已注释）
# pattern_size = (11, 8)
# left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
# right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# ret1, corners1 = cv2.findChessboardCorners(left_img_gray, pattern_size, None)
# ret2, corners2 = cv2.findChessboardCorners(right_img_gray, pattern_size, None)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# corners1 = cv2.cornerSubPix(left_img_gray, corners1, (11, 11), (-1, -1), criteria)
# corners2 = cv2.cornerSubPix(right_img_gray, corners2, (11, 11), (-1, -1), criteria)

# P1 = np.dot(M1, np.hstack((np.eye(3), np.zeros((3, 1)))))
# P2 = np.dot(M2, np.hstack((R, t)))

# points_4d = cv2.triangulatePoints(P1, P2, corners1.reshape(-1, 2).T, corners2.reshape(-1, 2).T)
# points_3d = points_4d[:3, :] / points_4d[3, :]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_3d.T)

# # 可视化点云
# o3d.visualization.draw_geometries([pcd])

# print("部分三维点坐标：")
# print(points_3d[:, :5].T)


def click_event(event, x, y, flags, param):
    """
    鼠标点击事件处理函数
    在左图中点击时，在右图中绘制对应的水平极线
    用于验证立体校正的效果

    Args:
        event: 鼠标事件类型
        x, y: 点击位置坐标
        flags: 事件标志
        param: 参数
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将点击坐标转换为原始坐标（去除缩放效果）
        x = x / scale
        y = y / scale

        print(f"clicked {x},{y}")

        # 计算水平线的起点和终点
        x0, y0 = 0, int(y * scale)
        x1, y1 = left_img_rectified_scaled.shape[1] + right_img_rectified_scaled.shape[1], int(y * scale)

        # 在拼接后的图像上绘制水平线（验证极线平行性）
        stitched_image = np.hstack((left_img_rectified_scaled, right_img_rectified_scaled))
        cv2.line(stitched_image, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # 显示拼接后的图像
        cv2.imshow('Image1', stitched_image)
        return right_img
    

# 创建水平拼接的校正图像，用于交互式极线验证
stitched_image = np.hstack((left_img_rectified_scaled, right_img_rectified_scaled))
cv2.imshow('Image1', stitched_image)

# 设置鼠标回调函数
# 点击左图中的任意点，会在整个拼接图中绘制水平线
# 如果校正正确，相同高度的像素点应该在同一水平线上
cv2.setMouseCallback('Image1', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
