# 结构光立体视觉主程序
# 用于双目结构光三维重建和特征匹配

import cv2
import os
import numpy as np

from jiegouguang_class import JieGouGuang

# 旧的测试图像路径（已注释）
# img = cv2.imread('d455_jiegouguang_save/left2.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# H,W = img.shape

# 创建结构光处理对象
# 使用左右相机拍摄的图像对进行处理
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/left2.png','d455_jiegouguang_save/right2.png')
jiegouguang_class = JieGouGuang('whx_biaoding/left_0046.png','whx_biaoding/right_0046.png')

# 导入双目相机标定参数
# 包括内参矩阵、畸变系数、外参矩阵等
# jiegouguang_class.import_biaodin('biaoding/extrinsics_d435_20250915.yml','biaoding/intrinsics_d435_20250915.yml')
jiegouguang_class.import_biaodin('biaoding/extrinsics_whx_zyb.yml','biaoding/intrinsics_whx_zyb.yml')


# 绘制棋盘格标定效果图（已注释）
# 用于验证双目标定的准确性
# biaoding_img = jiegouguang_class.draw_chess_board(jiegouguang_class.img1,jiegouguang_class.img2)
# biaoding_img_rectify = jiegouguang_class.draw_chess_board(jiegouguang_class.img1_rectify,jiegouguang_class.img2_rectify)

# 提取结构光圆形光斑中心（已注释）
# img1_with_center,img2_with_center = jiegouguang_class.extract_circle()

# 执行特征匹配
# 使用SIFT算法在校正后的左右图像中寻找匹配特征点
img_matches = jiegouguang_class.feature_matching()

# 三角测量重建三维点云（待实现）
# jiegouguang_class.triangulate_points()

# 将左右图像水平拼接显示（已注释）
# img_out = np.hstack((img1_with_center, img2_with_center))

# 保存处理结果
# 注意：此处使用了未定义的变量img1_with_center，需要先执行extract_circle()方法
cv2.imwrite("test.png", img_matches)
print("处理完成")

