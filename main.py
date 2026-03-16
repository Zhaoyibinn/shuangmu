import cv2
import os
import numpy as np

from jiegouguang.jiegouguang_class import JieGouGuang
import open3d as o3d
import time

from tqdm import tqdm


from ultralytics.models.sam import SAM3DynamicInteractivePredictor
from ultralytics.utils.plotting import Annotator, colors






data_root_path = "huojian/d455_huojian_ce"
depth_dir = 'depth_outputs/test'

sam = False

jiegouguang_class = JieGouGuang(os.path.join(data_root_path, 'left'),os.path.join(data_root_path, 'right'),img_rgb_path=os.path.join(data_root_path, "color")) # 碎片


if sam:
    sam_loc_dir = os.path.join(data_root_path, "sam_loc")
    sam_loc_files = sorted(os.listdir(sam_loc_dir))

    # 支持多个 refer_image 及其视觉提示
    refer_images = []
    points_list = []
    labels_list = []

    for sam_txt in sam_loc_files:
        if sam_txt.endswith('.txt'):
            image_name = sam_txt.replace('.txt', '.png')  # 假设图片后缀为 .png，根据实际情况可能需调整
            image_path = os.path.join(data_root_path, "color", image_name)
            if os.path.exists(image_path):
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

video_path = os.path.join(depth_dir, 'adepth_sequence.mp4')
rgb_video_path = os.path.join(depth_dir, 'argb_sequence.mp4')

for idx in tqdm(range(len(jiegouguang_class.img1_list))):

    # if idx >= 50:
    #     break

    jiegouguang_class.import_biaodin('biaoding/extrinsics_d455_20250915.yml','biaoding/intrinsics_d455_20250915.yml',idx=idx)



    # start = time.time()
    disparity_raw = jiegouguang_class.forward_disparity()
    # print(f"\033[31mCosting time (s): {time.time() - start}\033[0m")




    depth = (float(jiegouguang_class.K1[0, 0]) * abs(float(jiegouguang_class.cam_t[0]))) / disparity_raw
    depth = np.clip(depth, jiegouguang_class.min_dis, jiegouguang_class.max_dis)
    depth_list.append(depth)

    
    




    
    if jiegouguang_class.img_rgb_list is not None:
        rgbd = jiegouguang_class.get_rgbd(depth_L=depth,rgb_img = jiegouguang_class.img_rgb_list[idx])
        rgb_path = os.path.join(depth_dir, f'rgb_{idx:04d}.png')
        rgb = rgbd[:, :, :3].astype(np.uint8)
        depth = rgbd[:, :, 3].astype(np.uint16)
        gray_left_img = jiegouguang_class.img1_list[idx]
        cv2.imwrite(os.path.join(depth_dir, 'depth', f'depth_{idx:04d}.png'), rgbd[:,:,3].astype(np.uint16))
        cv2.imwrite(os.path.join(depth_dir, 'depth_vis', f'depth_vis_{idx:04d}.png'), (rgbd[:,:,3]/4).astype(np.uint8))
        # img_fusion = cv2.addWeighted(rgb,0.6,cv2.cvtColor(gray_left_img,cv2.COLOR_GRAY2BGR),0.4,0)

        # cv2.imwrite(os.path.join(depth_dir, f'img_fusion_{idx:04d}.png'), img_fusion)
        if video_writer_rgb is None:
            height, width, _ = rgb.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_rgb = cv2.VideoWriter(rgb_video_path, fourcc, 30, (width, height), True)
            if not video_writer_rgb.isOpened():
                raise RuntimeError(f'failed to open video writer for {rgb_video_path}')
        video_writer_rgb.write(rgb)



    depth_vis = np.clip(depth * 0.2, 0, 255).astype(np.uint8)
    # cv2.imwrite(os.path.join(depth_dir, f'depth_{idx:03d}.png'), depth_vis)
    if jiegouguang_class.img_rgb_list is None:
        cv2.imwrite(os.path.join(depth_dir, 'depth', f'depth_{idx:04d}.png'), depth.astype(np.uint16))
    if video_writer is None:
        height, width = depth_vis.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height), False)
        if not video_writer.isOpened():
            raise RuntimeError(f'failed to open video writer for {video_path}')
    video_writer.write(depth_vis)


    


    pcd = jiegouguang_class.depth2pointcloud(depth)
    # pcd =  pcd.voxel_down_sample(voxel_size=10)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
    o3d.io.write_point_cloud(os.path.join(depth_dir, 'cloud', f'cloud_{idx:04d}.ply'), pcd)

    if sam:

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

        pcd_sam = jiegouguang_class.depth2pointcloud(depth_sam)
        pcd_sam.colors = o3d.utility.Vector3dVector(np.repeat([[0.0, 1.0, 0.0]], len(pcd_sam.points), axis=0))
        o3d.io.write_point_cloud(os.path.join(depth_dir, 'cloud_sam', f'cloud_sam_{idx:04d}.ply'), pcd_sam)
        

        # pcd = jiegouguang_class.depth2pointcloud(depth)
        # # pcd =  pcd.voxel_down_sample(voxel_size=10)
        # pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
        # o3d.io.write_point_cloud("test.ply", pcd)


        # jiegouguang_class.cal_error(pcd) # 计算平面误差

if video_writer is not None:
    video_writer.release()
    





