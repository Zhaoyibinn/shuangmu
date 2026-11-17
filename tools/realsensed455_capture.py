import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# 初始化管道和配置
pipeline = rs.pipeline()
config = rs.config()

# 配置流：深度、彩色、左右红外
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)  # 左红外
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)  # 右红外

# 开始流
profile = pipeline.start(config)

try:
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
        print("激光器已关闭")
    else:
        print("设备不支持激光器控制")

    # 每次运行创建独立目录，保留已有采集结果。
    save_root_path = os.path.join("d455", datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    os.makedirs(save_root_path)
    print(f"保存目录：{save_root_path}；按 c 保存彩色和左右红外，按 q 退出")
    idx = 0
    while True:
        # 等待一组连贯的帧
        frames = pipeline.wait_for_frames()
        
        # 获取各帧
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir_left_frame = frames.get_infrared_frame(1)  # 左红外
        ir_right_frame = frames.get_infrared_frame(2) # 右红外
        if not all((depth_frame, color_frame, ir_left_frame, ir_right_frame)):
            continue
        
        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        ir_right_image = np.asanyarray(ir_right_frame.get_data())
        
        # 显示图像
        cv2.imshow('Left IR', ir_left_image)
        cv2.imshow('Right IR', ir_right_image)
        cv2.imshow('Color', color_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            idx = idx + 1
            for name, image in (("left", ir_left_image), ("right", ir_right_image), ("color", color_image)):
                path = os.path.join(save_root_path, f"{name}{idx}.png")
                if not cv2.imwrite(path, image):
                    raise OSError(f"无法保存图像：{path}")
            print(f"已保存第 {idx} 组：left{idx}.png、right{idx}.png、color{idx}.png")


finally:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()
