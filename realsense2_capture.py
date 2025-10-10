# 双目RealSense相机采集系统
# 使用两个RealSense相机同时采集彩色图像，支持手动保存功能

import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 初始化双目相机的管道
pipeline1 = rs.pipeline()  # 第一个相机管道
pipeline2 = rs.pipeline()  # 第二个相机管道

# 初始化双目相机的配置
config1 = rs.config()
config2 = rs.config()

# 初始化RealSense上下文，查找所有连接的设备
ctx = rs.context()
devices = ctx.query_devices()

# 图像ID计数器
now_id = 0

# 检查是否连接了至少两个相机
if len(devices) < 2:
    print("未检测到足够的相机，请检查连接。")
else:
    # 为每个相机配置流，使用各自的序列号进行区分
    config1.enable_device(devices[0].get_info(rs.camera_info.serial_number))
    config2.enable_device(devices[1].get_info(rs.camera_info.serial_number))

    # 配置彩色流：640x480分辨率，30FPS，BGR8格式
    config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动两个相机的数据流
    pipeline1.start(config1)
    pipeline2.start(config2)

    try:
        # 主循环：连续采集和显示图像
        while True:
            now_id += 1

            # 等待两个相机的帧数据
            frames1 = pipeline1.wait_for_frames()
            frames2 = pipeline2.wait_for_frames()

            # 提取彩色帧
            color_frame1 = frames1.get_color_frame()
            color_frame2 = frames2.get_color_frame()

            # 检查帧数据有效性
            if not color_frame1 or not color_frame2:
                continue

            # 将帧数据转换为numpy数组
            color_image1 = np.asanyarray(color_frame1.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())

            # 显示两个相机的实时图像
            cv2.imshow('Camera 1', color_image1)
            cv2.imshow('Camera 2', color_image2)

            # 按键事件处理
            key = cv2.waitKey(1)

            # 按's'键抓取并保存当前帧
            if key & 0xFF == ord('s'):
                folder_path = "realsense_capture"
                # 确保保存目录存在
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # 保存两个相机的彩色图像
                cv2.imwrite(f"realsense_capture/color_1_{now_id}.png", color_image1)
                cv2.imwrite(f"realsense_capture/color_2_{now_id}.png", color_image2)

                print(f"图像已保存为 color_1_{now_id}.png 和 color_2_{now_id}.png")

            # 按'q'键退出循环
            elif key & 0xFF == ord('q'):
                break

    finally:
        # 清理资源：停止两个相机的数据流并关闭所有窗口
        pipeline1.stop()
        pipeline2.stop()
        cv2.destroyAllWindows()
    