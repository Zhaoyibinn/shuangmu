# 结构光立体视觉处理类
# 实现双目相机标定、图像矫正、特征匹配和三维重建等功能

import cv2
import numpy as np
import copy

class JieGouGuang:
    """
    结构光立体视觉处理类
    用于处理双目相机拍摄的结构光图像，实现三维重建
    """

    def __init__(self, img1_path, img2_path):
        """
        初始化结构光处理对象

        Args:
            img1_path (str): 左相机图像路径
            img2_path (str): 右相机图像路径
        """
        self.img1_path = img1_path
        self.img2_path = img2_path
        # 读取图像并转换为灰度图
        self.img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)

    def extract_circle(self):
        """
        提取两幅图像中的圆形光斑中心

        Returns:
            tuple: (img1_with_center, img2_with_center) 标记了光斑中心的图像
        """
        # 分别提取左右图像的圆形光斑中心
        centers_img1, img1_with_center = self.extract_circle_1(self.img1)
        centers_img2, img2_with_center = self.extract_circle_1(self.img2)

        # 保存提取的中心点坐标
        self.centers_img1 = centers_img1
        self.centers_img2 = centers_img2
        return img1_with_center, img2_with_center

    def extract_circle_1(self, img):
        """
        从单幅图像中提取圆形光斑中心
        使用阈值分割和轮廓检测方法

        Args:
            img (numpy.ndarray): 输入灰度图像

        Returns:
            tuple: (centers, img_with_center) 中心点坐标数组和标记图像
        """

        H, W = img.shape
        mean_light = img.mean()
        # 使用自适应阈值进行二值化，阈值为平均亮度的2倍
        _, binary_image = cv2.threshold(img, mean_light * 2, 255, cv2.THRESH_BINARY)

        # 查找外部轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_center = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        center_extracted = []

        # 遍历每个轮廓
        for contour in contours:
            # 计算轮廓的几何矩
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            # 计算轮廓中心
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # 选择光斑周围的小区域进行亚像素精度中心定位
            bbox_size = 5
            y_start = max(0, cY-bbox_size)
            y_end = min(H, cY+bbox_size)
            x_start = max(0, cX-bbox_size)
            x_end = min(W, cX+bbox_size)
            local_patch = img[y_start:y_end, x_start:x_end]

            # 在局部区域内寻找最亮点作为精确中心
            max_pos = np.unravel_index(np.argmax(local_patch), local_patch.shape)
            subpixel_center = (cX - bbox_size + max_pos[1], cY - bbox_size + max_pos[0])

            # 在图像上标记提取的中心点
            cv2.circle(img_with_center, (int(subpixel_center[0]), int(subpixel_center[1])), 1, (0,0,255), -1)
            center_extracted.append([int(subpixel_center[0]), int(subpixel_center[1])])

        return np.array(center_extracted), img_with_center
    
    def import_biaodin(self, extri_path, intri_path):
        """
        导入双目相机标定参数并进行图像矫正

        Args:
            extri_path (str): 外参文件路径（包含旋转矩阵R和平移向量T）
            intri_path (str): 内参文件路径（包含相机矩阵M和畸变系数D）
        """
        # 读取外参和内参文件
        extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
        intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)

        # 读取左右相机的内参矩阵和畸变系数
        M1 = intri.getNode('M1').mat()  # 左相机内参矩阵
        M2 = intri.getNode('M2').mat()  # 右相机内参矩阵
        D1 = intri.getNode('D1').mat()  # 左相机畸变系数
        D2 = intri.getNode('D2').mat()  # 右相机畸变系数

        # 读取双目外参
        R = extri.getNode('R').mat()    # 右相机相对于左相机的旋转矩阵
        t = extri.getNode('T').mat()    # 右相机相对于左相机的平移向量

        # 计算反对称矩阵（用于计算基础矩阵）
        t_cross = np.array([[0, -t[2][0], t[1][0]],
                            [t[2][0], 0, -t[0][0]],
                            [-t[1][0], t[0][0], 0]])

        # 计算基础矩阵 F = inv(M2)^T * [t]_x * R * inv(M1)
        F = np.dot(np.dot(np.transpose(np.linalg.inv(M2)), np.dot(t_cross, R)), np.linalg.inv(M1))

        # 获取最优相机矩阵（去除畸变后的）
        h, w = self.img1.shape[:2]
        new_M1, roi1 = cv2.getOptimalNewCameraMatrix(M1, D1, (w, h), 1, (w, h))
        new_M2, roi2 = cv2.getOptimalNewCameraMatrix(M2, D2, (w, h), 1, (w, h))

        # 保存校正后的相机参数
        self.K1 = new_M1
        self.K2 = new_M2
        self.cam_R = R
        self.cam_t = t

        # 计算立体校正的变换矩阵
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(new_M1, D1, new_M2, D2, (w, h), R, t, flags=cv2.CALIB_ZERO_TANGENT_DIST)

        # 生成重映射表
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(new_M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(new_M2, D2, R2, P2, (w, h), cv2.CV_32FC1)

        # 应用立体校正，使左右图像的极线平行
        left_rectified = cv2.remap(self.img1, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(self.img2, right_map_x, right_map_y, cv2.INTER_LINEAR)

        # 保存校正后的图像
        self.img1_rectify = left_rectified
        self.img2_rectify = right_rectified


    def feature_matching(self):
        """
        使用SIFT算法进行特征匹配
        在立体校正后的左右图像中寻找对应特征点

        Returns:
            numpy.ndarray: 特征匹配结果的可视化图像
        """
        # 初始化匹配点列表
        self.kp1_np = []
        self.kp2_np = []

        # 创建SIFT特征检测器
        sift = cv2.SIFT_create()

        # 复制校正后的图像
        img1 = copy.copy(self.img1_rectify)
        img2 = copy.copy(self.img2_rectify)

        # 检测并计算SIFT特征点和描述子
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 使用暴力匹配器进行特征匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # 应用Lowe's比率测试筛选好的匹配
        good = []
        for m, n in matches:
            # 最近邻距离与次近邻距离的比值小于0.75
            if m.distance < 0.75 * n.distance:
                pt_1 = kp1[m.queryIdx].pt
                pt_2 = kp2[m.trainIdx].pt

                # 极线约束：匹配点的y坐标差应该很小（已校正图像中极线是水平的）
                if abs(pt_1[1] - pt_2[1]) < 3:
                    self.kp1_np.append(pt_1)
                    self.kp2_np.append(pt_2)
                    good.append([m])

        # 转换为numpy数组
        self.kp1_np = np.array(self.kp1_np)
        self.kp2_np = np.array(self.kp2_np)

        # 绘制匹配结果
        img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

        return img_matches
    

    def triangulate_points(self):
        """
        三角测量重建三维点云
        使用匹配的特征点对计算其在三维空间中的坐标

        Note:
            此方法尚未完全实现，需要投影矩阵P1和P2
        """

        # 转置匹配点坐标以符合cv2.triangulatePoints的输入格式
        pts1 = self.kp1_np.T
        pts2 = self.kp2_np.T

        # TODO: 需要从立体校正中获取投影矩阵P1, P2
        # points_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)

        pass

    def draw_chess_board(self, img1, img2):
        """
        在两幅图像中检测棋盘格角点并绘制极线对应关系
        用于验证立体校正的效果

        Args:
            img1 (numpy.ndarray): 左图像
            img2 (numpy.ndarray): 右图像

        Returns:
            numpy.ndarray or None: 绘制了角点连线的拼接图像，如果检测失败则返回None
        """
        # 角点检测的精度参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        pattern_size = (11, 8)  # 根据实际棋盘格调整尺寸

        # 检测棋盘格角点
        ret1, corners1 = cv2.findChessboardCorners(img1, pattern_size)
        ret2, corners2 = cv2.findChessboardCorners(img2, pattern_size)

        if ret1 and ret2:
            # 亚像素级角点精化
            corners1 = cv2.cornerSubPix(img1, corners1, (11,11), (-1,-1), criteria)
            corners2 = cv2.cornerSubPix(img2, corners2, (11,11), (-1,-1), criteria)

            # 创建水平拼接图像
            h_img = cv2.hconcat([img1, img2])
            h_img = cv2.cvtColor(h_img, cv2.COLOR_GRAY2BGR)

            # 绘制对应角点和连线
            w = img1.shape[1]
            for i in range(len(corners1)):
                # 左图角点坐标
                pt1 = (int(corners1[i][0][0]), int(corners1[i][0][1]))
                # 右图角点坐标（需要加上左图宽度偏移）
                pt2 = (int(corners2[i][0][0]) + w, int(corners2[i][0][1]))

                # 绘制角点（绿色圆圈）
                cv2.circle(h_img, pt1, 3, (0,255,0), -1)
                cv2.circle(h_img, pt2, 3, (0,255,0), -1)
                # 绘制对应关系连线（红色直线）
                cv2.line(h_img, pt1, pt2, (0,0,255), 1)

            return h_img

        return None

