#!/usr/bin/env python3
import cv2
import argparse
import sys
import os

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"坐标 (x, y): ({x}, {y})")

def main():
    parser = argparse.ArgumentParser(description="点击图片获取像素坐标 (Click on image to get pixel coordinates)")
    parser.add_argument("image_path", help="Path to the input image file")
    
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: 找不到图片文件 {args.image_path}")
        sys.exit(1)

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: 无法读取图片文件 {args.image_path}")
        sys.exit(1)

    window_name = "Image - Click to get coordinates (Press 'q' or 'ESC' to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("========================================")
    print("操作说明：")
    print("1. 在弹出的图片窗口中点击鼠标左键，终端将输出点击的坐标。")
    print("2. 在弹出的窗口中按 'q' 键或 'ESC' 键退出。")
    print("========================================")

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
