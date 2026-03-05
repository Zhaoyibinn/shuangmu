import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

left_img_path = "whx_biaoding/L/left_0041.png"
right_img_path = "whx_biaoding/right_0090.png"

left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)

# 设定目标宽度
target_width = 640
# 计算缩放比例
scale = target_width / float(left_img.shape[1])
# 计算目标高度
target_height = int(left_img.shape[0] * scale)
left_img_scaled = cv2.resize(left_img, (target_width, target_height))
right_img_scaled = cv2.resize(right_img, (target_width, target_height))

extri = cv2.FileStorage('biaoding/extrinsics_whx_zyb.yml', cv2.FILE_STORAGE_READ)
intri = cv2.FileStorage('biaoding/intrinsics_whx_zyb.yml', cv2.FILE_STORAGE_READ)

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

# 进行极线矫正
def rectify_images(left_img, right_img, M1, M2, R, t):
    # 计算新的相机矩阵和ROI
    h, w = left_img.shape[:2]
    new_M1, roi1 = cv2.getOptimalNewCameraMatrix(M1, D1, (w, h), 1, (w, h))
    new_M2, roi2 = cv2.getOptimalNewCameraMatrix(M2, D2, (w, h), 1, (w, h))

    # 计算极线变换矩阵
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(new_M1, D1, new_M2, D2, (w, h), R, t, flags=cv2.CALIB_ZERO_TANGENT_DIST)

    # 生成映射
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(new_M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(new_M2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    # 进行矫正
    left_rectified = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

    return left_rectified, right_rectified

# 使用内参进行极线矫正
left_img_rectified, right_img_rectified = rectify_images(left_img, right_img, M1, M2, R, t)

left_img_rectified_scaled = cv2.resize(left_img_rectified, (target_width, target_height))
right_img_rectified_scaled = cv2.resize(right_img_rectified, (target_width, target_height))


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
    if event == cv2.EVENT_LBUTTONDOWN:
        x = x / scale
        y = y / scale

        print(f"clicked {x},{y}")

        # 计算横线的起点和终点
        x0, y0 = 0, int(y * scale)
        x1, y1 = left_img_rectified_scaled.shape[1] + right_img_rectified_scaled.shape[1], int(y * scale)

        # 在拼接后的图像上绘制横线
        stitched_image = np.hstack((left_img_rectified_scaled, right_img_rectified_scaled))
        cv2.line(stitched_image, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # 显示拼接后的图像
        cv2.imshow('Image1', stitched_image)
        return right_img
    

stitched_image = np.hstack((left_img_rectified_scaled, right_img_rectified_scaled))
cv2.imshow('Image1', stitched_image)
cv2.setMouseCallback('Image1', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
