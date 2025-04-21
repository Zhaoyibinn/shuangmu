import pyrealsense2 as rs
import numpy as np
import cv2
import os
now_id=0
align_to = rs.stream.color
align = rs.align(align_to)

#深度图帧图像滤波器
hole_filling_filter=rs.hole_filling_filter(2)

#配置文件
pipe = rs.pipeline()
cfg_rs = rs.config()
profile = pipe.start(cfg_rs)
# D400相机开启参数
cfg_rs.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
cfg_rs.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)

try:
    while True:
        #获取帧图像
        frame = pipe.wait_for_frames()

        #对齐之后的frame
        aligned_frame = align.process(frame)

        #获得数据帧
        depth_frame = aligned_frame.get_depth_frame()
        color_frame = aligned_frame.get_color_frame()

        # 深度参数，像素坐标系转相机坐标系用到，要拿彩色作为内参，因为深度图会对齐到彩色相机
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        # print('color_intrin:', color_intrin)

        #将深度图彩色化的工具
        colorizer = rs.colorizer()

        #将彩色图和深度图进行numpy化
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2BGR)
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        #输入视频
        # out.write(color_image)
        #将深度图彩色化     
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        all_images = np.hstack((color_image, colorized_depth))
        
        height, width = all_images.shape[:2]
        new_width = int(width / 3)
        new_height = int(height / 3)
        resized_image = cv2.resize(all_images, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow('all_images', resized_image)

        
        

        
        #帧数设定
        key = cv2.waitKey(30)

        now_id+=1



        #按键事件
        if key == ord("q"):
            print('用户退出！')
            break

        if key == ord("s"):
            folder_path = "realsense_capture"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            cv2.imwrite(f"realsense_capture/color_{now_id}.png",color_image)
            cv2.imwrite(f"realsense_capture/depth_{now_id}.png",depth_image)
            print('保存图像')
            



finally:
    pipe.stop()