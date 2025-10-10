# RealSense单相机实时数据采集系统
# 同时采集彩色图像和深度图像，支持手动保存功能

import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 初始化变量
now_id = 0  # 图像ID计数器

# 配置深度图像对齐到彩色图像
align_to = rs.stream.color
align = rs.align(align_to)

# 深度图像滤波器，用于填充空洞
hole_filling_filter = rs.hole_filling_filter(2)

# 初始化RealSense管道和配置
pipe = rs.pipeline()
cfg_rs = rs.config()
profile = pipe.start(cfg_rs)

# D400系列相机参数配置
# 启用深度流：640x480分辨率，16位深度格式，30FPS
cfg_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# 启用彩色流：640x480分辨率，BGR8格式，30FPS
cfg_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    # 主循环：连续采集和显示图像
    while True:
        # 获取一帧数据
        frame = pipe.wait_for_frames()

        # 对齐深度图到彩色图，确保像素一一对应
        aligned_frame = align.process(frame)

        # 获取对齐后的数据帧
        depth_frame = aligned_frame.get_depth_frame()
        color_frame = aligned_frame.get_color_frame()

        # 获取彩色相机的内参，用于像素坐标系转相机坐标系
        # 注意：要使用彩色相机的内参，因为深度图已对齐到彩色相机
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        # print('color_intrin:', color_intrin)

        # 初始化深度图彩色化工具
        colorizer = rs.colorizer()

        # 将彩色图和深度图转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # 颜色空间转换（如需要）
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # 可选：输出视频流（已注释）
        # out.write(color_image)

        # 将深度图彩色化以便观察
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # 水平拼接彩色图和深度图
        all_images = np.hstack((color_image, colorized_depth))

        # 缩放图像以便显示（缩放到3分之1）
        height, width = all_images.shape[:2]
        new_width = int(width / 3)
        new_height = int(height / 3)
        resized_image = cv2.resize(all_images, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 显示拼接后的图像
        cv2.imshow('all_images', resized_image)

        # 设置帧率和键盘检测
        key = cv2.waitKey(30)
        now_id += 1

        # 按键事件处理
        if key == ord("q"):
            print('用户退出！')
            break

        # 按's'键保存当前帧的彩色图和深度图
        if key == ord("s"):
            folder_path = "realsense_capture"
            # 确保保存目录存在
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # 保存彩色图和深度图
            cv2.imwrite(f"realsense_capture/color_{now_id}.png", color_image)
            cv2.imwrite(f"realsense_capture/depth_{now_id}.png", depth_image)
            print('保存图像')
            



finally:
    # 清理资源：停止数据流管道
    pipe.stop()