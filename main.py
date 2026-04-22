import cv2
import os
import numpy as np
import yaml

from jiegouguang.jiegouguang_class import JieGouGuang
from Recon.Aruco_pose import Aruco_pose_Estimater
import open3d as o3d
import time

from tqdm import tqdm


from ultralytics.models.sam import SAM3DynamicInteractivePredictor
from ultralytics.utils.plotting import Annotator, colors






def rotation_matrix_to_quaternion(rotation_matrix):
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    trace = np.trace(matrix)

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


def find_matching_image(image_dir, stem):
    for extension in ['.png', '.jpg', '.jpeg', '.bmp']:
        image_path = os.path.join(image_dir, stem + extension)
        if os.path.exists(image_path):
            return image_path
    return None



WORLD_VOXEL_SIZE_MM = 5.0
aruco_length = 148.0
data_root_path = "huojian/aurco/d455_0417"
depth_dir = 'depth_outputs/d455_aruco'

ext_yaml_path = 'biaoding/extrinsics_d455_20250915.yml'
int_yaml_path = 'biaoding/intrinsics_d455_20250915.yml'
# 相机内外参标定的yaml文件路径

color_ext_yaml_path = 'jiegouguang/color/rgb_calib_zyb_20250121.yaml'

sam = True
# 是否需要开启基于SAM的目标分割
aruco = True
# 图像中是否有aruco 如果有的话就可以开启基于aruco的定位

jiegouguang_class = JieGouGuang(os.path.join(data_root_path, 'left'),os.path.join(data_root_path, 'right'),img_rgb_path=os.path.join(data_root_path, "color")) # 碎片
# jiegouguang_class = JieGouGuang(os.path.join(data_root_path, 'left'),os.path.join(data_root_path, 'right')) # 碎片

# jiegouguang_class = JieGouGuang(os.path.join(data_root_path, 'left_20260417_135836.mp4'),os.path.join(data_root_path, 'right_20260417_135836.mp4'),img_rgb_path=os.path.join(data_root_path, "color_20260417_135836.mp4")) # 碎片


if sam:
    sam_loc_dir = os.path.join(data_root_path, "sam_loc")
    sam_loc_files = sorted(os.listdir(sam_loc_dir))

    # 支持多个 refer_image 及其视觉提示
    refer_images = []
    points_list = []
    labels_list = []

    for sam_txt in sam_loc_files:
        if sam_txt.endswith('.txt'):
            image_stem = os.path.splitext(sam_txt)[0]
            image_path = find_matching_image(os.path.join(data_root_path, "color"), image_stem)
            if image_path is not None:
                refer_images.append(image_path)
                loc = np.loadtxt(os.path.join(sam_loc_dir, sam_txt), delimiter=',', dtype=float)
                points_list.append(loc[:, :2])
                labels_list.append(loc[:, 2])

    if points_list:
        points_list = np.vstack(points_list).tolist()
        labels_list = np.int_(np.concatenate(labels_list)).tolist()

    overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1008, model="sam3.pt", save=False)
    predictor = SAM3DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

    predictor(source=refer_images[0], points=[points_list], labels=[labels_list], obj_ids=[0], update_memory=True)
# results = predictor(source=refer_images[0])
# jiegouguang_class.method = 'sgbm'
# jiegouguang_class.method = 'foundation_stereo'
# jiegouguang_class.method = 'bridgedepth'
jiegouguang_class.method = 'fast_foundation_stereo'
jiegouguang_class.init_model()

depth_list = []

os.makedirs(depth_dir, exist_ok=True)
os.makedirs(os.path.join(depth_dir, "depth"), exist_ok=True)
os.makedirs(os.path.join(depth_dir, "depth_vis"), exist_ok=True)
os.makedirs(os.path.join(depth_dir, "cloud"), exist_ok=True)

if sam:
    os.makedirs(os.path.join(depth_dir, "depth_sam"), exist_ok=True)
    os.makedirs(os.path.join(depth_dir, "cloud_sam"), exist_ok=True)
    os.makedirs(os.path.join(depth_dir, "rgb_sam"), exist_ok=True)

video_writer = None
video_writer_rgb = None
rgb = None




if aruco:

    global_cloud = o3d.geometry.PointCloud()
    global_cloud_sam = o3d.geometry.PointCloud()
    aruco_output_dir = os.path.join(depth_dir, 'aruco_fusion')
    pose_vis_dir = os.path.join(aruco_output_dir, 'pose_vis')
    pose_yaml_path = os.path.join(aruco_output_dir, 'camera_poses.yaml')
    global_cloud_path = os.path.join(aruco_output_dir, 'merged_cloud_marker_frame.ply')
    global_cloud_sam_path = os.path.join(aruco_output_dir, 'merged_cloud_marker_frame_sam.ply')

    pose_records = []
    os.makedirs(aruco_output_dir, exist_ok=True)
    os.makedirs(pose_vis_dir, exist_ok=True)

    aruco_estimater = Aruco_pose_Estimater.from_yaml(color_ext_yaml_path, aruco_length=aruco_length)


video_path = os.path.join(depth_dir, 'adepth_sequence.mp4')
rgb_video_path = os.path.join(depth_dir, 'argb_sequence.mp4')

for idx in tqdm(range(len(jiegouguang_class.img1_list))):
    rgb = None
    frame_pose = None

    # if idx >= 50:
    #     break

    jiegouguang_class.import_biaodin(ext_yaml_path, int_yaml_path, color_ext_yaml_path, idx=idx)
    # 读取标定、读取图像数据


    # start = time.time()
    disparity_raw = jiegouguang_class.forward_disparity()
    # 推理视差图
    # print(f"\033[31mCosting time (s): {time.time() - start}\033[0m")
    depth = (float(jiegouguang_class.K1[0, 0]) * abs(float(jiegouguang_class.cam_t[0]))) / disparity_raw
    depth = np.clip(depth, jiegouguang_class.min_dis, jiegouguang_class.max_dis)
    depth_list.append(depth)
    # 计算深度图
    gray_left_img = jiegouguang_class.img1_list[idx]
    aruco_est_frame = gray_left_img




    if jiegouguang_class.img_rgb_list is not None:
        # 如果有彩色图，那就需要反变换了
        rgbd = jiegouguang_class.get_rgbd(depth_L=depth,rgb_img = jiegouguang_class.img_rgb_list[idx])
        rgb_path = os.path.join(depth_dir, f'rgb_{idx:04d}.png')
        rgb = rgbd[:, :, :3].astype(np.uint8)
        depth = rgbd[:, :, 3].astype(np.uint16)
        # cv2.imwrite(os.path.join(depth_dir, 'depth', f'depth_{idx:04d}.png'), rgbd[:,:,3].astype(np.uint16))
        # cv2.imwrite(os.path.join(depth_dir, 'depth_vis', f'depth_vis_{idx:04d}.png'), (rgbd[:,:,3]/4).astype(np.uint8))
        # img_fusion = cv2.addWeighted(rgb,0.6,cv2.cvtColor(gray_left_img,cv2.COLOR_GRAY2BGR),0.4,0)

        # cv2.imwrite(os.path.join(depth_dir, f'img_fusion_{idx:04d}.png'), img_fusion)
        if video_writer_rgb is None:
            height, width, _ = rgb.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_rgb = cv2.VideoWriter(rgb_video_path, fourcc, 30, (width, height), True)
            if not video_writer_rgb.isOpened():
                raise RuntimeError(f'failed to open video writer for {rgb_video_path}')
        video_writer_rgb.write(rgb)
        aruco_est_frame = rgb
        # 如果有彩色图，就用rgb对齐的depth和rgb顶掉depth和aruco_est_frame






    if aruco:
        # 用aruco码做全局定位
        frame_pose = aruco_estimater.estimate_pose_from_frame(aruco_est_frame)
        pose_vis = aruco_estimater.draw_pose_visualization(aruco_est_frame, frame_pose, idx)
        cv2.imwrite(os.path.join(pose_vis_dir, f'frame_{idx:04d}.png'), pose_vis)

        if frame_pose is not None:
            rotation_w2c, _ = cv2.Rodrigues(frame_pose['rvec_obj_to_cam'])
            translation_w2c = frame_pose['tvec_obj_to_cam'].reshape(-1)
            pose_records.append({
                'rotation': rotation_matrix_to_quaternion(rotation_w2c),
                'translation': translation_w2c.astype(np.float64).tolist(),
            })






    # 保存结果
    depth_vis = np.clip(depth * 0.2, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(depth_dir, 'depth', f'depth_{idx:04d}.png'), depth.astype(np.uint16))
    cv2.imwrite(os.path.join(depth_dir, 'depth_vis', f'depth_vis_{idx:04d}.png'), (depth_vis/2).astype(np.uint8))
    if jiegouguang_class.img_rgb_list is None:
        cv2.imwrite(os.path.join(depth_dir, 'depth', f'depth_{idx:04d}.png'), depth.astype(np.uint16))
    if video_writer is None:
        height, width = depth_vis.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height), False)
        if not video_writer.isOpened():
            raise RuntimeError(f'failed to open video writer for {video_path}')
    video_writer.write(depth_vis)
    # 保存深度图和可视化的深度图


    pcd = jiegouguang_class.depth2pointcloud(depth, color_image=rgb)
    # pcd =  pcd.voxel_down_sample(voxel_size=10)
    if len(pcd.colors) == 0:
        pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
    o3d.io.write_point_cloud(os.path.join(depth_dir, 'cloud', f'cloud_{idx:04d}.ply'), pcd)
    # 保存点云




    if aruco and frame_pose is not None:
        global_cloud += aruco_estimater.transform_point_cloud(pcd, frame_pose)
    # 如果有aruco就保存全局点云



    if sam:
        # 基于SAM的目标分割
        results = predictor(source=rgb)
        masks = results[0].masks  # 获取分割掩码
        if masks is not None and len(masks) > 0:
            sam_results_mask = masks[0].data.cpu().numpy().astype(np.uint8)[0]
            sam_results_mask = cv2.resize(sam_results_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        else:
            # 没有匹配到目标时，提供一个纯黑掩码防止报错
            sam_results_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=bool)

        # 处理 SAM 分割的部分
        depth_sam = np.copy(depth)
        depth_sam[~sam_results_mask] = 0
        cv2.imwrite(os.path.join(depth_dir, 'depth_sam', f'depth_sam_{idx:04d}.png'), depth_sam.astype(np.uint16))
        
        # 保存 SAM 分割后的 RGB 图像
        rgb_sam = np.copy(rgb)
        rgb_sam[~sam_results_mask] = [0, 0, 0]
        cv2.imwrite(os.path.join(depth_dir, 'rgb_sam', f'rgb_sam_{idx:04d}.png'), cv2.cvtColor(rgb_sam, cv2.COLOR_BGR2RGB))

        pcd_sam = jiegouguang_class.depth2pointcloud(depth_sam, color_image=rgb_sam)
        if len(pcd_sam.colors) == 0:
            pcd_sam.colors = o3d.utility.Vector3dVector(np.repeat([[0.0, 1.0, 0.0]], len(pcd_sam.points), axis=0))
        o3d.io.write_point_cloud(os.path.join(depth_dir, 'cloud_sam', f'cloud_sam_{idx:04d}.ply'), pcd_sam)
        if aruco and frame_pose is not None:
            global_cloud_sam += aruco_estimater.transform_point_cloud(pcd_sam, frame_pose)
        # pcd = jiegouguang_class.depth2pointcloud(depth)
        # # pcd =  pcd.voxel_down_sample(voxel_size=10)
        # pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
        # o3d.io.write_point_cloud("test.ply", pcd)
        # jiegouguang_class.cal_error(pcd) # 计算平面误差


if video_writer is not None:
    video_writer.release()
if video_writer_rgb is not None:
    video_writer_rgb.release()
# 保存视频


# 保存aruco的位姿和全局点云
if aruco:
    with open(pose_yaml_path, 'w', encoding='utf-8') as pose_yaml_file:
        yaml.safe_dump({'camera_poses': pose_records}, pose_yaml_file, allow_unicode=True, sort_keys=False)
if aruco and len(global_cloud.points) > 0:
    global_cloud = global_cloud.voxel_down_sample(voxel_size=WORLD_VOXEL_SIZE_MM)
    o3d.io.write_point_cloud(global_cloud_path, global_cloud)
if aruco and sam and len(global_cloud_sam.points) > 0:
    global_cloud_sam = global_cloud_sam.voxel_down_sample(voxel_size=WORLD_VOXEL_SIZE_MM)
    o3d.io.write_point_cloud(global_cloud_sam_path, global_cloud_sam)
    





