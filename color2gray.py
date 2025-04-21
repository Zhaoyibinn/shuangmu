import cv2
import os
 
img_list = []
input_path = r"C:\Users\ZhaoYibin\Desktop\SDK_MVCamCtrl_Win_STD_V4.5.1_V2.5.1\shuangmu\right"  # 要处理的图片所在的文件夹
output_path = r"C:\Users\ZhaoYibin\Desktop\SDK_MVCamCtrl_Win_STD_V4.5.1_V2.5.1\shuangmu\gray\right"  # 处理完的图片放在这里
for item in os.listdir(input_path):
    img_list.append(os.path.join(input_path, item))
print(list)
count = 1
for imagepath in img_list:
    # print(imagepath)
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    print(output_path+'%dR.jpg' % count)  # 显示保存文件的路径及保存的文件名
    cv2.imwrite(output_path+'%dR.jpg' % count, image)
    # 按一定路径将图片保存下来并命名，加号左边代表保存路径，右边代表文件命
    # %d代表后面的% count中的count的数值
    print("-----------执行中，保存第{}张----------".format(count+1))
    count += 1
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # cv2.imshow('garyimg', image)
    # cv2.waitKey(0)