#!/usr/bin/env python3
import cv2
import argparse
import os
import sys

# 全局变量
points = []
img_display = None
window_name = "SAM Prompt Labeler (q/ESC to save and exit)"

def mouse_callback(event, x, y, flags, param):
    global img_display, points
    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键正标签 (1)
        points.append((x, y, 1))
        print(f"添加正样本点 (x, y): ({x}, {y}) -> label: 1")
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)  # 绿色圆点
        cv2.imshow(window_name, img_display)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键负标签 (0)
        points.append((x, y, 0))
        print(f"添加负样本点 (x, y): ({x}, {y}) -> label: 0")
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)  # 红色圆点
        cv2.imshow(window_name, img_display)

def main():
    global img_display
    parser = argparse.ArgumentParser(description="为SAM模型提取点提示信息 (左键正样本，右键负样本)")
    parser.add_argument("image_path", help="输入图片的路径")
    parser.add_argument("--output", "-o", help="输出的txt文件路径 (可选，默认保存在原图片同目录下同名txt)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: 找不到图片文件 {args.image_path}")
        sys.exit(1)

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: 无法读取图像 {args.image_path}")
        sys.exit(1)

    img_display = img.copy()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("======================================================")
    print("操作说明：")
    print(" [鼠标左键]：添加 正样本点 (SAM label=1)，显示为绿色。")
    print(" [鼠标右键]：添加 负样本点 (SAM label=0)，显示为红色。")
    print(" [键盘 q/ESC]：退出并保存结果到txt文件中。")
    print("======================================================")

    while True:
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC 或 q
            break

    cv2.destroyAllWindows()

    if not points:
        print("未选择任何点，退出程序。")
        return

    # 确定输出路径
    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(args.image_path)
        out_path = base + ".txt"

    # 保存文件
    with open(out_path, "w", encoding="utf-8") as f:
        for p in points:
            f.write(f"{p[0]},{p[1]},{p[2]}\n")
            
    print(f"\n成功保存 {len(points)} 个点至文件: {out_path}")
    print("文件格式: x,y,label (1为正样本，0为负样本)")

if __name__ == "__main__":
    main()
