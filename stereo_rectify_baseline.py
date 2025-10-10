# RealSense双目相机图像矫正与基线计算
# 实现图像矫正并计算校正前后的基线长度

import cv2
import numpy as np
import os

def rectify_and_compute_baseline(left_img_path, right_img_path, intri_path, extri_path, save_rectified=True):
    """
    对RealSense拍摄的双目图像进行矫正，并计算校正前后的基线长度

    Args:
        left_img_path: 左图路径
        right_img_path: 右图路径
        intri_path: 内参文件路径
        extri_path: 外参文件路径
        save_rectified: 是否保存矫正后的图像

    Returns:
        dict: 包含基线信息和矫正图像
    """

    # 读取图像
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)

    if img_left is None or img_right is None:
        raise ValueError("无法读取图像文件")

    h, w = img_left.shape[:2]
    print(f"图像尺寸: {w}x{h}")

    # 读取标定参数
    fs_intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    fs_extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)

    # 内参
    M1 = fs_intri.getNode('M1').mat()  # 左相机内参
    M2 = fs_intri.getNode('M2').mat()  # 右相机内参
    D1 = fs_intri.getNode('D1').mat()  # 左相机畸变
    D2 = fs_intri.getNode('D2').mat()  # 右相机畸变

    # 外参
    R = fs_extri.getNode('R').mat()   # 旋转矩阵
    T = fs_extri.getNode('T').mat()   # 平移向量

    fs_intri.release()
    fs_extri.release()

    # 计算原始基线（校正前）
    baseline_original = np.linalg.norm(T)
    print(f"\n=== 校正前 ===")
    print(f"基线长度: {baseline_original:.2f} mm")
    print(f"平移向量 T: [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}] mm")

    # 立体校正
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        M1, D1, M2, D2, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # 从投影矩阵P2提取校正后的基线
    # P2的形式为 [fx' 0 cx' -fx'*Tx; 0 fy' cy' 0; 0 0 1 0]
    # 其中Tx是校正后的基线长度
    baseline_rectified = abs(P2[0, 3] / P1[0, 0]) if P1[0, 0] != 0 else 0

    print(f"\n=== 校正后 ===")
    print(f"基线长度: {baseline_rectified:.2f} mm")
    print(f"基线变化: {abs(baseline_rectified - baseline_original):.2f} mm")

    # 计算重映射表
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        M1, D1, R1, P1, (w, h), cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        M2, D2, R2, P2, (w, h), cv2.CV_32FC1
    )

    # 应用重映射进行图像矫正
    img_left_rect = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

    # 绘制水平线验证极线对齐
    img_left_lines = img_left_rect.copy()
    img_right_lines = img_right_rect.copy()

    # 每隔30像素画一条水平线
    for y in range(0, h, 30):
        cv2.line(img_left_lines, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(img_right_lines, (0, y), (w, y), (0, 255, 0), 1)

    # 拼接显示
    combined = np.hstack([img_left_lines, img_right_lines])

    # 保存结果
    if save_rectified:
        output_dir = "rectified_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存矫正后的图像
        cv2.imwrite(os.path.join(output_dir, "left_rectified.png"), img_left_rect)
        cv2.imwrite(os.path.join(output_dir, "right_rectified.png"), img_right_rect)
        cv2.imwrite(os.path.join(output_dir, "combined_with_lines.png"), combined)

        # 保存对比图
        before = np.hstack([img_left, img_right])
        after = np.hstack([img_left_rect, img_right_rect])
        comparison = np.vstack([before, after])
        cv2.imwrite(os.path.join(output_dir, "before_after_comparison.png"), comparison)

        print(f"\n矫正后的图像已保存到 {output_dir}/ 目录")

    # 打印投影矩阵信息
    print(f"\n=== 投影矩阵 ===")
    print(f"P1 (左相机):\n{P1}")
    print(f"\nP2 (右相机):\n{P2}")

    # 计算视差-深度转换矩阵Q的相关参数
    print(f"\n=== 视差-深度转换参数 ===")
    print(f"焦距 fx: {P1[0,0]:.2f}")
    print(f"主点 cx: {P1[0,2]:.2f}, cy: {P1[1,2]:.2f}")
    print(f"Q矩阵用于将视差转换为3D坐标")

    return {
        'baseline_original': baseline_original,
        'baseline_rectified': baseline_rectified,
        'img_left_rect': img_left_rect,
        'img_right_rect': img_right_rect,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'R1': R1,
        'R2': R2
    }


if __name__ == "__main__":
    # 设置文件路径
    left_img = "whx_biaoding/rosL1.png"
    right_img = "whx_biaoding/rosR1.png"
    intri_file = "biaoding/intrinsics_d435_20250915.yml"
    extri_file = "biaoding/extrinsics_d435_20250915.yml"

    # 检查文件是否存在
    import os
    files_to_check = [left_img, right_img, intri_file, extri_file]
    for f in files_to_check:
        if not os.path.exists(f):
            print(f"警告: 文件 {f} 不存在")
            # 尝试其他可能的路径
            if "color_1_375" in f:
                alt_path = "realsense_capture/color_1_1.png"
                if os.path.exists(alt_path):
                    left_img = alt_path
                    print(f"使用替代文件: {alt_path}")
            elif "color_2_375" in f:
                alt_path = "realsense_capture/color_2_1.png"
                if os.path.exists(alt_path):
                    right_img = alt_path
                    print(f"使用替代文件: {alt_path}")

    try:
        # 执行矫正和基线计算
        result = rectify_and_compute_baseline(
            left_img, right_img,
            intri_file, extri_file,
            save_rectified=True
        )

        print("\n" + "="*50)
        print("处理完成！")
        print(f"原始基线: {result['baseline_original']:.2f} mm")
        print(f"矫正后基线: {result['baseline_rectified']:.2f} mm")

    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保以下文件存在:")
        print("1. 左右相机图像")
        print("2. 内参文件 intrinsics_realsense_20240429.yml")
        print("3. 外参文件 extrinsics_realsense_20240429.yml")