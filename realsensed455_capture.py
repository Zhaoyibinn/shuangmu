# RealSense D455红外双目相机采集系统
# 采集左右红外图像用于双目立体视觉，支持激光器控制

import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 初始化RealSense管道和配置
pipeline = rs.pipeline()
config = rs.config()

# 配置多种数据流：深度、彩色、左右红外
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)     # 深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)    # 彩色流
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # 左红外
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # 右红外

# 启动数据流
profile = pipeline.start(config)

# 获取设备句柄并控制激光器
device = profile.get_device()
depth_sensor = device.first_depth_sensor()

# 检查并设置激光器（发射器）开关
if depth_sensor.supports(rs.option.emitter_enabled):
    # 关闭激光器（设置为0.0），开启设置为1.0
    # 关闭激光器可获得更纯净的红外图像
    depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
    print("激光器已关闭")  # 注意：原注释显示“已开启”，但实际设置为0.0是关闭
else:
    print("设备不支持激光器控制")

# 设置保存目录
save_root_path = "d455"

# 检查路径是否存在，如果存在则删除（清空旧数据）
if os.path.exists(save_root_path):
    os.rmdir(save_root_path)

# 创建新的保存目录
os.makedirs(save_root_path)


try:
    idx = 0  # 图像编号计数器

    # 主循环：连续采集和显示图像
    while True:
        # 等待一组连贯的帧数据
        frames = pipeline.wait_for_frames()

        # 获取各类型的帧数据
        depth_frame = frames.get_depth_frame()        # 深度帧
        color_frame = frames.get_color_frame()        # 彩色帧
        ir_left_frame = frames.get_infrared_frame(1)  # 左红外帧
        ir_right_frame = frames.get_infrared_frame(2) # 右红外帧

        # 转换为numpy数组以便处理
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        ir_right_image = np.asanyarray(ir_right_frame.get_data())

        # 应用色彩映射到深度图像（用于可视化）
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示左右红外图像
        cv2.imshow('Left IR', ir_left_image)
        cv2.imshow('Right IR', ir_right_image)

        # 可选：显示彩色和深度图像（已注释）
        # cv2.imshow('Color', color_image)
        # cv2.imshow('Depth', depth_colormap)

        # 按键事件处理
        key = cv2.waitKey(1) & 0xFF

        # 按'q'退出（已注释）
        # if key == ord('q'):
        #     break

        # 按'c'键抓取并保存当前帧的左右红外图像
        if key == ord('c'):
            idx = idx + 1
            # 保存左右红外图像
            cv2.imwrite(os.path.join(save_root_path, f"left{idx}.png"), ir_left_image)
            cv2.imwrite(os.path.join(save_root_path, f"right{idx}.png"), ir_right_image)
            print(f"saved to {os.path.join(save_root_path, f'left{idx}.png')}")


finally:
    # 清理资源：停止数据流并关闭所有窗口
    pipeline.stop()
    cv2.destroyAllWindows()