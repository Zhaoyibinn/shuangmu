import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

left_img_path = "imgs\\realsense\color\left\depth_2_1887.png"
right_img_path = "imgs\\realsense\color\\right\color_1_1887.png"

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

extri = cv2.FileStorage('extrinsics_realsense.yml', cv2.FILE_STORAGE_READ)
intri = cv2.FileStorage('intrinsics_realsense.yml', cv2.FILE_STORAGE_READ)

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

pattern_size = (11, 8)
left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

ret1, corners1 = cv2.findChessboardCorners(left_img_gray, pattern_size, None)
ret2, corners2 = cv2.findChessboardCorners(right_img_gray, pattern_size, None)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners1 = cv2.cornerSubPix(left_img_gray, corners1, (11, 11), (-1, -1), criteria)
corners2 = cv2.cornerSubPix(right_img_gray, corners2, (11, 11), (-1, -1), criteria)

P1 = np.dot(M1, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(M2, np.hstack((R, t)))

points_4d = cv2.triangulatePoints(P1, P2, corners1.reshape(-1, 2).T, corners2.reshape(-1, 2).T)
points_3d = points_4d[:3, :] / points_4d[3, :]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d.T)

# 可视化点云
o3d.visualization.draw_geometries([pcd])

# print("部分三维点坐标：")
# print(points_3d[:, :5].T)


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 点击点的坐标
        
        x = x / scale
        y = y / scale
        point1 = np.array([[x, y, 1]]).T

        print(f"clicked {x},{y}")
        # 计算极线
        line = np.dot(F, point1)

        # 归一化极线
        line = line / line[2]

        # 找到极线与图像边界的交点
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [right_img.shape[1], -(line[2] + line[0] * right_img.shape[1]) / line[1]])



        # 在第二张图片上绘制极线
        cv2.line(right_img_scaled, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)), (0, 0, 255), 2)

        stitched_image = np.hstack((left_img_scaled, right_img_scaled))



        # 显示第二张图片
        cv2.imshow('Image1', stitched_image)
        # plt.imshow(right_img_scaled)
        # plt.show()
        return right_img
    

# stitched_image = np.hstack((left_img_scaled, right_img_scaled))
# cv2.imshow('Image1', stitched_image)
# cv2.setMouseCallback('Image1', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# stitched_image = np.hstack((left_img, right_img))

# plt.imshow(stitched_image)
# plt.show()