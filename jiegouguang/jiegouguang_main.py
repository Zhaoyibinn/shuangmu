import cv2
import os
import numpy as np

from jiegouguang_class import JieGouGuang
import open3d as o3d
import time

from tqdm import tqdm


data_root_path = "d455_jiegouguang_save/20260116_suipian/d455_suipian_shenzhou_old_shu"
jiegouguang_class = JieGouGuang(os.path.join(data_root_path, 'left'),os.path.join(data_root_path, 'right'),img_rgb_path=os.path.join(data_root_path, "color")) # 碎片

# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251202_rgb_videos/diff/left_20251202_065423.mp4','d455_jiegouguang_save/20251202_rgb_videos/diff/right_20251202_065423.mp4',img_rgb_path="d455_jiegouguang_save/20251202_rgb_videos/diff/rgb_20251202_065423.mp4") # 带RGB的桌子视频
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251116_videos/floor/left_20251116_130809.mp4','d455_jiegouguang_save/20251116_videos/floor/right_20251116_130809.mp4') # 地板视频
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251116_videos/zyb_desk/left_20251116_122816.mp4','d455_jiegouguang_save/20251116_videos/zyb_desk/right_20251116_122816.mp4') # zyb桌子视频
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251103/left3.png','d455_jiegouguang_save/20251103/right3.png') # 一个杯子
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/better/left2.png','d455_jiegouguang_save/better/right2.png') # 平面场景
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/left2.png','d455_jiegouguang_save/right2.png') # 复杂场景 zyb桌子

# 下面这两种是稀疏方法 暂时弃用
# pcd = jiegouguang_class.manual_feature_extracting()
# pcd = jiegouguang_class.lg_feature_extracting()


# jiegouguang_class.method = 'sgbm'
jiegouguang_class.method = 'foundation_stereo'
# jiegouguang_class.method = 'bridgedepth'
jiegouguang_class.init_model()

depth_list = []
depth_dir = 'depth_outputs_d455_suipian_shenzhou_old_shu'
os.makedirs(depth_dir, exist_ok=True)

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

    
    
    depth_vis = np.clip(depth * 0.2, 0, 255).astype(np.uint8)
    # cv2.imwrite(os.path.join(depth_dir, f'depth_{idx:03d}.png'), depth_vis)
    if jiegouguang_class.img_rgb_list is None:
        cv2.imwrite(os.path.join(depth_dir, f'depth_{idx:04d}.png'), depth.astype(np.uint16))
    if video_writer is None:
        height, width = depth_vis.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height), False)
        if not video_writer.isOpened():
            raise RuntimeError(f'failed to open video writer for {video_path}')
    video_writer.write(depth_vis)



    
    if jiegouguang_class.img_rgb_list is not None:
        rgbd = jiegouguang_class.get_rgbd(depth_L=depth,rgb_img = jiegouguang_class.img_rgb_list[idx])
        rgb_path = os.path.join(depth_dir, f'rgb_{idx:04d}.png')
        rgb = rgbd[:, :, :3].astype(np.uint8)
        gray_left_img = jiegouguang_class.img1_list[idx]
        cv2.imwrite(os.path.join(depth_dir, f'depth_{idx:04d}.png'), rgbd[:,:,3].astype(np.uint16))
        img_fusion = cv2.addWeighted(rgb,0.6,cv2.cvtColor(gray_left_img,cv2.COLOR_GRAY2BGR),0.4,0)

        cv2.imwrite(os.path.join(depth_dir, f'img_fusion_{idx:04d}.png'), img_fusion)
        if video_writer_rgb is None:
            height, width, _ = rgb.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_rgb = cv2.VideoWriter(rgb_video_path, fourcc, 30, (width, height), True)
            if not video_writer_rgb.isOpened():
                raise RuntimeError(f'failed to open video writer for {rgb_video_path}')
        video_writer_rgb.write(rgb)


    pcd = jiegouguang_class.depth2pointcloud(depth)
    # pcd =  pcd.voxel_down_sample(voxel_size=10)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
    o3d.io.write_point_cloud(os.path.join(depth_dir, f'cloud_{idx:04d}.ply'), pcd)

    

    # pcd = jiegouguang_class.depth2pointcloud(depth)
    # # pcd =  pcd.voxel_down_sample(voxel_size=10)
    # pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
    # o3d.io.write_point_cloud("test.ply", pcd)


    # jiegouguang_class.cal_error(pcd) # 计算平面误差

if video_writer is not None:
    video_writer.release()
    





