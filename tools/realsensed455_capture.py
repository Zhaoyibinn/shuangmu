import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 初始化管道和配置
pipeline = rs.pipeline()
config = rs.config()

# 配置流：深度、彩色、左右红外
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # 左红外
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # 右红外

# 开始流
profile = pipeline.start(config)

# 获取设备并控制激光器
device = profile.get_device()
depth_sensor = device.first_depth_sensor()

# 检查并设置激光器（发射器）开关
if depth_sensor.supports(rs.option.emitter_enabled):
    # 开启激光器（设置为1.0），关闭设置为0.0
    depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
    print("激光器已开启")
else:
    print("设备不支持激光器控制")



save_root_path = "d455"
# 检查路径是否存在
if os.path.exists(save_root_path):
    os.rmdir(save_root_path)

os.makedirs(save_root_path)


try:
    idx = 0
    while True:
        # 等待一组连贯的帧
        frames = pipeline.wait_for_frames()
        
        # 获取各帧
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir_left_frame = frames.get_infrared_frame(1)  # 左红外
        ir_right_frame = frames.get_infrared_frame(2) # 右红外
        
        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        ir_right_image = np.asanyarray(ir_right_frame.get_data())
        
        # 应用色图到深度图像（用于可视化）
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # 显示图像
        cv2.imshow('Left IR', ir_left_image)
        cv2.imshow('Right IR', ir_right_image)
        # cv2.imshow('Color', color_image)
        # cv2.imshow('Depth', depth_colormap)
        
        # 按'q'退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        if cv2.waitKey(1) & 0xFF == ord('c'):
            idx = idx + 1
            cv2.imwrite(os.path.join(save_root_path,f"left{idx}.png"), ir_left_image)
            cv2.imwrite(os.path.join(save_root_path,f"right{idx}.png"), ir_right_image)
            print(f"saved to {os.path.join(save_root_path,f'left{idx}.png')}")


finally:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()