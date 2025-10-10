#!/usr/bin/env python3
# 一键双目立体矫正、基线计算、FoundationStereo预处理脚本
# 读取YAML → 去畸变+立体矫正 → 计算基线B → 生成intrinsics.txt

import cv2
import numpy as np
import argparse
import os
import sys
import textwrap
from pathlib import Path

def read_intr(path):
    """读取内参YAML文件"""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开内参文件: {path}")

    M1 = fs.getNode('M1').mat()
    M2 = fs.getNode('M2').mat()
    D1 = fs.getNode('D1').mat()
    D2 = fs.getNode('D2').mat()
    fs.release()

    if any(x is None for x in [M1,M2,D1,D2]):
        raise ValueError("YAML缺少M1/M2/D1/D2节点")
    return M1, M2, D1, D2

def read_extr(path):
    """读取外参YAML文件"""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开外参文件: {path}")

    R = fs.getNode('R').mat()
    T = fs.getNode('T').mat()
    fs.release()

    if any(x is None for x in [R,T]):
        raise ValueError("YAML缺少R/T节点")
    return R, T

def stereo_rectify_and_baseline(left_path, right_path, intri_path, extr_path,
                               out_dir="stereo_output", alpha=0.0):
    """
    双目立体矫正和基线计算主函数

    Args:
        left_path: 左图路径
        right_path: 右图路径
        intri_path: 内参YAML路径
        extr_path: 外参YAML路径
        out_dir: 输出目录
        alpha: 立体校正参数 (0=裁黑边对齐主点, 1=尽量保视场)
    """

    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 读取图像
    print(f"读取左图: {left_path}")
    print(f"读取右图: {right_path}")

    L = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if L is None or R is None:
        raise FileNotFoundError("左右图路径错误或图像读取失败")

    h, w = L.shape[:2]
    print(f"图像尺寸: {w}x{h}")

    # 读取标定参数
    print(f"读取内参: {intri_path}")
    print(f"读取外参: {extr_path}")

    M1, M2, D1, D2 = read_intr(intri_path)
    Rlr, Tlr = read_extr(extr_path)  # 右相机相对左相机

    print("\n=== 原始标定参数 ===")
    print(f"左相机内参 M1:\n{M1}")
    print(f"右相机内参 M2:\n{M2}")
    print(f"旋转矩阵 R:\n{Rlr}")
    print(f"平移向量 T: {Tlr.reshape(-1)}")

    # 立体校正
    print(f"\n执行立体校正 (alpha={alpha})...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        M1, D1, M2, D2, (w, h), Rlr, Tlr,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha
    )

    # 生成重映射表
    mx1, my1 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    mx2, my2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    # 应用重映射
    Lr = cv2.remap(L, mx1, my1, cv2.INTER_LINEAR)
    Rr = cv2.remap(R, mx2, my2, cv2.INTER_LINEAR)

    # 保存矫正后图像
    left_rect_path = os.path.join(out_dir, "left_rect.png")
    right_rect_path = os.path.join(out_dir, "right_rect.png")
    cv2.imwrite(left_rect_path, Lr)
    cv2.imwrite(right_rect_path, Rr)

    # 生成对比图
    combined_original = np.hstack([L, R])
    combined_rectified = np.hstack([Lr, Rr])

    # 在矫正后图像上画水平线验证
    Lr_lines = cv2.cvtColor(Lr.copy(), cv2.COLOR_GRAY2BGR)
    Rr_lines = cv2.cvtColor(Rr.copy(), cv2.COLOR_GRAY2BGR)
    for y in range(0, h, 30):
        cv2.line(Lr_lines, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(Rr_lines, (0, y), (w, y), (0, 255, 0), 1)
    combined_with_lines = np.hstack([Lr_lines, Rr_lines])

    cv2.imwrite(os.path.join(out_dir, "combined_original.png"), combined_original)
    cv2.imwrite(os.path.join(out_dir, "combined_rectified.png"), combined_rectified)
    cv2.imwrite(os.path.join(out_dir, "combined_with_epipolar_lines.png"), combined_with_lines)

    # 基线计算 - 三种方法互证
    print("\n=== 基线计算 (三种方法互证) ===")

    # 方法1: 直接从平移向量T计算
    B_T = float(np.linalg.norm(Tlr.reshape(-1)))
    print(f"方法1 - 从T向量: B = ||T|| = {B_T:.6f} mm")

    # 方法2: 从投影矩阵P2计算
    fx = float(P2[0, 0])
    Tx = float(P2[0, 3])  # P2[0,3] = -fx*B
    B_P = abs(-Tx / fx) if fx != 0 else 0
    print(f"方法2 - 从P2矩阵: B = |P2[0,3]/fx| = |{Tx:.3f}/{fx:.3f}| = {B_P:.6f} mm")

    # 方法3: 从Q矩阵计算
    B_Q = 1.0 / abs(float(Q[3, 2])) if Q[3, 2] != 0 else 0
    print(f"方法3 - 从Q矩阵: B = 1/|Q[3,2]| = 1/|{Q[3,2]:.6f}| = {B_Q:.6f} mm")

    # 检验一致性
    max_diff = max(abs(B_T - B_P), abs(B_T - B_Q), abs(B_P - B_Q))
    print(f"三种方法最大差异: {max_diff:.6f} mm")

    if max_diff < 0.01:  # 差异小于0.01mm认为一致
        print("✓ 基线计算结果一致")
        baseline_final = B_T  # 使用原始方法作为最终结果
    else:
        print("⚠ 基线计算结果存在差异，请检查标定参数")
        baseline_final = B_T

    # 输出校正后的投影矩阵信息
    print(f"\n=== 校正后投影矩阵 ===")
    print(f"P1 (左相机):\n{P1}")
    print(f"P2 (右相机):\n{P2}")
    print(f"Q (视差转深度):\n{Q}")

    # 生成FoundationStereo需要的intrinsics.txt
    # 第一行: K矩阵展平(9个数)
    # 第二行: 基线B(单位:米)
    K_flat = M1.astype(np.float32).flatten()  # K矩阵展平为9个数
    B_meters = np.array([baseline_final / 1000.0], dtype=np.float32)  # mm转米，单个数

    intrinsics_path = os.path.join(out_dir, "intrinsics.txt")

    # 手动写入两行，避免维度问题
    with open(intrinsics_path, 'w') as f:
        # 第一行：K矩阵9个参数
        f.write(' '.join([f'{x:.6f}' for x in K_flat]) + '\n')
        # 第二行：基线(米)
        f.write(f'{B_meters[0]:.6f}\n')

    # 生成详细报告
    report = f"""
=== 双目立体矫正与基线计算报告 ===

输入文件:
- 左图: {left_path}
- 右图: {right_path}
- 内参: {intri_path}
- 外参: {extr_path}

图像信息:
- 分辨率: {w}x{h}
- 校正参数alpha: {alpha}

基线计算结果:
- 方法1(从T): {B_T:.6f} mm
- 方法2(从P2): {B_P:.6f} mm
- 方法3(从Q): {B_Q:.6f} mm
- 最大差异: {max_diff:.6f} mm
- 最终基线: {baseline_final:.6f} mm ({baseline_final/1000:.6f} m)

输出文件:
- 左图矫正: {left_rect_path}
- 右图矫正: {right_rect_path}
- FoundationStereo内参: {intrinsics_path}
- 极线验证图: {os.path.join(out_dir, "combined_with_epipolar_lines.png")}

相机参数:
- 矫正后焦距fx: {P1[0,0]:.2f}
- 矫正后焦距fy: {P1[1,1]:.2f}
- 矫正后主点cx: {P1[0,2]:.2f}
- 矫正后主点cy: {P1[1,2]:.2f}
"""

    report_path = os.path.join(out_dir, "stereo_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"详细报告已保存: {report_path}")
    print(f"FoundationStereo内参已生成: {intrinsics_path}")

    return {
        'baseline_mm': baseline_final,
        'baseline_m': baseline_final / 1000.0,
        'left_rect': left_rect_path,
        'right_rect': right_rect_path,
        'intrinsics_txt': intrinsics_path,
        'P1': P1, 'P2': P2, 'Q': Q
    }

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""
        双目立体矫正、基线计算与FoundationStereo预处理

        功能:
        1. 读取左右图像和标定参数
        2. 执行去畸变和立体矫正
        3. 三种方法计算基线长度并互证
        4. 生成FoundationStereo需要的intrinsics.txt
        5. 输出矫正效果验证图像

        示例:
        python one_click_stereo.py \\
            --left whx_biaoding/rosL1.png \\
            --right whx_biaoding/rosR1.png \\
            --intri biaoding/intrinsics_d435_20250915.yml \\
            --extr biaoding/extrinsics_d435_20250915.yml
        """))

    parser.add_argument("--left", required=True, help="左图路径")
    parser.add_argument("--right", required=True, help="右图路径")
    parser.add_argument("--intri", required=True, help="内参YAML路径")
    parser.add_argument("--extr", required=True, help="外参YAML路径")
    parser.add_argument("--out_dir", default="stereo_output", help="输出目录")
    parser.add_argument("--alpha", type=float, default=0.0,
                       help="stereoRectify alpha参数 (0=裁黑边对齐主点, 1=尽量保视场)")

    args = parser.parse_args()

    try:
        result = stereo_rectify_and_baseline(
            args.left, args.right, args.intri, args.extr,
            args.out_dir, args.alpha
        )
        print(f"\n✓ 处理完成! 所有结果保存在: {args.out_dir}")

    except Exception as e:
        print(f"✗ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()