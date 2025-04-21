import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shuangmu_lg

# 配置管道
pipeline1 = rs.pipeline()
pipeline2 = rs.pipeline()

config1 = rs.config()
config2 = rs.config()

# 查找所有连接的设备
ctx = rs.context()
devices = ctx.query_devices()

now_id=0


if len(devices) < 2:
    print("未检测到足够的相机，请检查连接。")
else:
    # 为每个相机配置流
    config1.enable_device(devices[0].get_info(rs.camera_info.serial_number))
    config2.enable_device(devices[1].get_info(rs.camera_info.serial_number))

    config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动管道
    pipeline1.start(config1)
    pipeline2.start(config2)

    try:
        while True:
            now_id+=1
            # 等待两个相机的帧
            frames1 = pipeline1.wait_for_frames()
            frames2 = pipeline2.wait_for_frames()

            # 提取彩色帧
            color_frame1 = frames1.get_color_frame()
            color_frame2 = frames2.get_color_frame()

            if not color_frame1 or not color_frame2:
                continue

            # 将帧转换为 numpy 数组
            color_image1 = np.asanyarray(color_frame1.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())

            # 显示图像
            # cv2.imshow('Camera 1', color_image1)
            # cv2.imshow('Camera 2', color_image2)

            shuangmu_lg.shuangmu_fun(color_image2,color_image1)


            # key = cv2.waitKey(1)
            # # 按 's' 键抓取一帧
            # if key & 0xFF == ord('s'):
            #     folder_path = "realsense_capture"
            #     if not os.path.exists(folder_path):
            #         os.makedirs(folder_path)
            #     cv2.imwrite(f"realsense_capture/color_1_{now_id}.png",color_image1)
            #     cv2.imwrite(f"realsense_capture/depth_2_{now_id}.png",color_image2)

            #     print(f"图像已保存为 color_1_{now_id}.png 和 depth_2_{now_id}.png")
            # # 按 'q' 键退出循环
            # elif key & 0xFF == ord('q'):
            #     break

    finally:
        # 停止管道并关闭窗口
        pipeline1.stop()
        pipeline2.stop()
        cv2.destroyAllWindows()
    