# 彩色图像转灰度图像批处理脚本
# 用于将指定文件夹中的彩色图像批量转换为灰度图像并保存到目标文件夹

import cv2
import os

# 初始化图像文件列表
img_list = []

# 输入和输出路径配置
input_path = r"whx_biaoding/R/"  # 要处理的图片所在的文件夹
output_path = r"whx_biaoding/gray/R/"  # 处理完的图片放在这里

# 遍历输入文件夹，获取所有图像文件路径
for item in os.listdir(input_path):
    img_list.append(os.path.join(input_path, item))

print(list)  # 打印文件列表

# 初始化计数器
count = 1

# 对文件列表进行排序，确保处理顺序一致
img_list.sort()

# 批量处理每张图像
for imagepath in img_list:
    # 以灰度模式读取图像
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

    # 显示保存文件的路径及保存的文件名
    print(output_path+'R%d.jpg' % count)

    # 保存灰度图像
    # 按一定路径将图片保存下来并命名，加号左边代表保存路径，右边代表文件名
    # %d代表后面的% count中的count的数值
    cv2.imwrite(output_path+'R%d.jpg' % count, image)

    # 显示处理进度
    print("-----------执行中，保存第{}张----------".format(count+1))

    # 计数器递增
    count += 1

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 可选：显示图像（已注释）
    # cv2.imshow('garyimg', image)
    # cv2.waitKey(0)