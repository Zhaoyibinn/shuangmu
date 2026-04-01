import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import copy
import datetime
import open3d as o3d

# 添加路径以便能够导入相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'jiegouguang'))
from fast_FoundationStereo.fast_stereo_inference import FastStereoInference

def setup_cameras():
    
    """初始化双目 RealSense 并启动硬件流"""
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        raise RuntimeError("未检测到至少2台RealSense相机，请检查连接！")
    
    pipeline1 = rs.pipeline()
    pipeline2 = rs.pipeline()
    config1 = rs.config()
    config2 = rs.config()
    
    # 绑定设备 (根据序列号)
    serial_1 = devices[0].get_info(rs.camera_info.serial_number)
    serial_2 = devices[1].get_info(rs.camera_info.serial_number)
    
    print(f"检测到相机1 (序列号): {serial_1}")
    print(f"检测到相机2 (序列号): {serial_2}")

    config1.enable_device(serial_1)
    config2.enable_device(serial_2)
    
    # 启用 RGB 流
    width, height = 1280, 720
    fps = 30
    config1.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # 启动管道
    print("正在启动相机管道...")
    pipeline1.start(config1)
    pipeline2.start(config2)
    return pipeline1, pipeline2

def load_calibration_and_get_maps(extri_path, intri_path, shape=(1280, 720)):
    weight_path = "jiegouguang/weights/fast_FoundationStereo/23-36-37/model_best_bp2_serialize.pth"
    """加载内外参并获取畸变与极线校正映射表"""
    if not os.path.exists(extri_path) or not os.path.exists(intri_path):
        raise FileNotFoundError(f"找不到标定文件！\n{extri_path}\n{intri_path}")
    
    extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    
    M1 = intri.getNode('M1').mat()
    D1 = intri.getNode('D1').mat()
    M2 = intri.getNode('M2').mat()
    D2 = intri.getNode('D2').mat()
    R = extri.getNode('R').mat()
    T = extri.getNode('T').mat()
    
    extri.release()
    intri.release()
    
    # 立体校正
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M1, D1, M2, D2, shape, R, T)
    
    # 获取重映射 map
    map1_x, map1_y = cv2.initUndistortRectifyMap(M1, D1, R1, P1, shape, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(M2, D2, R2, P2, shape, cv2.CV_32FC1)
    
    # K_rect 为校正后的左相机内参, baseline_m 为基线长度(平移向量的x方向距离)
    K_rect = P1[:3, :3]
    baseline_m = abs(float(T[0][0])) 
    stereo_model = FastStereoInference(ckpt_path=weight_path, valid_iters=24)
    return map1_x, map1_y, map2_x, map2_y, K_rect, baseline_m

def save_current_frame(save_root, color, depth_map, K, max_depth=1500):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存RGB
    cv2.imwrite(os.path.join(save_dir, "rgb.png"), color)
    
    # 2. 保存深度图
    cv2.imwrite(os.path.join(save_dir, "depth.png"), depth_map.astype(np.uint16))

    depth_vis = np.clip(depth_map * 0.05, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, "depth_vis.png"), depth_vis)
    
    # 3. 保存点云
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    
    depth = depth_map.astype(np.float64)
    rows, cols = np.indices(depth.shape)
    
    valid_mask = np.isfinite(depth) & (depth > 0)
    z = depth[valid_mask]
    u = cols[valid_mask]
    v = rows[valid_mask]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 给点云附加颜色
    if color is not None and color.shape[:2] == depth.shape[:2]:
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        colors = color_rgb[valid_mask].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
    o3d.io.write_point_cloud(os.path.join(save_dir, "cloud.ply"), pcd)
    print(f"[{timestamp}] 成功保存RGB、深度图和点云至: {save_dir}")

def main():
    extri_path = 'biaoding/extrinsics_2d415_20260401.yml'
    intri_path = 'biaoding/intrinsics_2d415_20260401.yml'
    weight_path = "jiegouguang/weights/fast_FoundationStereo/23-36-37/model_best_bp2_serialize.pth"
    
    # 数据保存路径
    save_root = "data_records"
    os.makedirs(save_root, exist_ok=True)
    # ==================

    print("============ 初始化标定参数 ============")
    map1_x, map1_y, map2_x, map2_y, K_rect, baseline_m = load_calibration_and_get_maps(
        extri_path, intri_path, shape=(1280, 720)
    )
    print("标定参数加载成功！基线长度: {:.4f}m".format(baseline_m))
    
    print("\n============ 初始化模型与环境 ============")
    print("正在加载 Fast-FoundationStereo 模型，这可能需要几秒钟...")
    stereo_model = FastStereoInference(ckpt_path=weight_path,max_disp=192)
    # 配置管道
    pipeline1 = rs.pipeline()
    pipeline2 = rs.pipeline()

    config1 = rs.config()
    config2 = rs.config()

    # 查找所有连接的设备
    ctx = rs.context()
    devices = ctx.query_devices()

    now_id=0

    config1.enable_device(devices[0].get_info(rs.camera_info.serial_number))
    config2.enable_device(devices[1].get_info(rs.camera_info.serial_number))

    config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline1.start(config1)
    pipeline2.start(config2)
    
    print("\n======== 进入实时推断 ======== \n按 's' 保存当前帧数据 (时间命名的文件夹) \n按 'q' 退出\n")
    try:
        while True:
            # 捕获双目实时帧 (需保持同步)
            frames1 = pipeline1.wait_for_frames()
            frames2 = pipeline2.wait_for_frames()
            
            c_frame1 = frames1.get_color_frame()
            c_frame2 = frames2.get_color_frame()
            if not c_frame1 or not c_frame2:
                continue

            # 将帧转换为 numpy 数组
            color1 = np.asanyarray(c_frame1.get_data())  # 假定为左图 (BGR)
            color2 = np.asanyarray(c_frame2.get_data())  # 假定为右图 (BGR)

            # 显示两侧摄像头的原始图
            cv2.imshow('Raw Left', color1)
            cv2.imshow('Raw Right', color2)

            # 极线校正 (Remap) - 网络要求左右图严格极线对其且无畸变
            # 如果深度图全是红色或其他异常，极有可能是左右图传反了，这里将 color2 赋值给 left，color1 赋值给 right 试试
            rectified_left = cv2.remap(color2, map1_x, map1_y, cv2.INTER_LINEAR)
            rectified_right = cv2.remap(color1, map2_x, map2_y, cv2.INTER_LINEAR)

            # 前向推理深度图
            results = stereo_model.infer(rectified_left, rectified_right, K_rect, baseline_m)
            depth_map = results['depth']
            
            # 深度效果可视化，通过色彩映射来显示 (JET)
            # FastStereoInference 中，由于之前计算出来的深度单位可能是毫米 (如果你的标定的 t 向量是毫米)
            # 根据 main.py 的逻辑可见：max_dis = 1500 (即1.5米)，depth * 0.2 被映射到了0-255，这说明深度值范围本身在0-1500左右（毫米单位）。
            # 所以做修改：如果 depth 超出 2.0，之前截断用的是 2.0米，如果目前结果其实是 2000 毫米，除以 2.0 再乘 255 当然全是红。
            # 根据 main.py 里的渲染习惯，使用 `depth * 0.2` 做截断，也就是说：
            # depth_vis = copy.deepcopy(depth_map)
            # depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis = np.clip(depth_map * 0.05, 0, 255).astype(np.uint8)
            # depth_color_map = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            # 将无效深度 (如背景或无效匹配) 涂成黑色
            # invalid_mask = depth_map <= 0
            # depth_color_map[invalid_mask] = 0
            cv2.imshow('Fast-FoundationStereo Depth', depth_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                save_current_frame(save_root, rectified_left, depth_map, K_rect)
            elif key == ord('q'):
                break

    finally:
        pipeline1.stop()
        pipeline2.stop()
        cv2.destroyAllWindows()
        print("程序已安全退出。")

if __name__ == "__main__":
    main()