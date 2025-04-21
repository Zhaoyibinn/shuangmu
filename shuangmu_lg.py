import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)


if __name__ == "__main__":
    left_img_path = "realsense_capture/depth_2_1042.png"
    right_img_path = "realsense_capture/color_1_1042.png"

    # left_img_path = "realsense_capture/color_1_1042.png"
    # right_img_path = "realsense_capture/depth_2_1042.png"

    left_img = cv2.cvtColor(cv2.imread(left_img_path),cv2.COLOR_RGB2BGR)
    right_img = cv2.cvtColor(cv2.imread(right_img_path),cv2.COLOR_RGB2BGR)



    image0 = numpy_image_to_torch(left_img)
    image1 = numpy_image_to_torch(right_img)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.save_plot("test.png")

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


    P1 = np.dot(M1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(M2, np.hstack((R, t)))



    points_4d = cv2.triangulatePoints(P1, P2, np.array(m_kpts0.cpu()).T, np.array(m_kpts1.cpu()).T)
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
    

def shuangmu_fun(img1,img2):
    image0 = numpy_image_to_torch(img1)
    image1 = numpy_image_to_torch(img2)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.save_plot("test.png")

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


    P1 = np.dot(M1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(M2, np.hstack((R, t)))



    points_4d = cv2.triangulatePoints(P1, P2, np.array(m_kpts0.cpu()).T, np.array(m_kpts1.cpu()).T)
    points_3d = points_4d[:3, :] / points_4d[3, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

    # print("部分三维点坐标：")
    # print(points_3d[:, :5].T)


    return 0