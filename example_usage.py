"""
BridgeDepthStereo 使用示例

作者: DX
创建时间: 2025-11-07
用途: 演示如何使用 BridgeDepthStereo 类进行立体匹配

运行方法:
    python example_usage.py --left data/left/left1.png --right data/right/right1.png
"""

import argparse
import imageio.v2 as imageio
import numpy as np
import open3d as o3d

from bridgedepth_stereo import BridgeDepthStereo


def read_camera_parameters(intrinsic_file: str):
    """
    读取相机参数文件

    文件格式:
        第1行: fx 0 cx 0 fy cy 0 0 1 (K 矩阵的 9 个值)
        第2行: baseline (基线，米)

    参数:
        intrinsic_file: 参数文件路径

    返回:
        K: 内参矩阵 (3, 3)
        baseline: 基线 (米)
    """
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
        baseline = float(lines[1])
    return K, baseline


def main():
    # ========== 参数解析 ==========
    parser = argparse.ArgumentParser(description='BridgeDepthStereo 使用示例')
    parser.add_argument('--left', type=str, default='data/left/left1.png',
                        help='左图路径')
    parser.add_argument('--right', type=str, default='data/right/right1.png',
                        help='右图路径')
    parser.add_argument('--intrinsic', type=str, default='K_d455.txt',
                        help='相机内参文件 (K 矩阵 + 基线)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/bridge_rvc_pretrain.pth',
                        help='模型权重路径')
    parser.add_argument('--output', type=str, default='output_example',
                        help='输出目录')
    parser.add_argument('--z_max', type=float, default=10.0,
                        help='最大深度 (米)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='计算设备')
    parser.add_argument('--visualize', action='store_true',
                        help='是否打开 Open3D 可视化窗口')
    args = parser.parse_args()

    print("=" * 70)
    print("BridgeDepthStereo 立体匹配示例")
    print("=" * 70)

    # ========== 第一步: 初始化模型并加载权重 ==========
    print("\n【第一步】初始化 BridgeDepthStereo 并加载模型权重...")
    stereo = BridgeDepthStereo(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    print("✓ 模型加载完成")

    # ========== 第二步: 加载图像和相机参数 ==========
    print("\n【第二步】加载图像和相机参数...")
    left_image = imageio.imread(args.left)
    right_image = imageio.imread(args.right)
    K, baseline = read_camera_parameters(args.intrinsic)

    print(f"  左图: {args.left} - 分辨率: {left_image.shape[:2]}")
    print(f"  右图: {args.right} - 分辨率: {right_image.shape[:2]}")
    print(f"  相机参数: fx={K[0,0]:.2f}, baseline={baseline:.5f}m")

    # ========== 第三步: 执行推理 ==========
    print("\n【第三步】执行立体匹配推理...")
    results = stereo.infer(
        left_image=left_image,
        right_image=right_image,
        K=K,
        baseline=baseline,
        z_min=0.1,
        z_max=args.z_max,
        return_pointcloud=True
    )
    print("✓ 推理完成")

    # ========== 第四步: 查看结果 ==========
    print("\n【第四步】推理结果:")
    print(f"  视差图: {results['disparity'].shape}, 范围: {results['disparity'].min():.2f} - {results['disparity'].max():.2f} 像素")
    print(f"  深度图: {results['depth'].shape}, 范围: {results['depth'].min():.3f} - {results['depth'].max():.3f} 米")
    print(f"  XYZ 坐标图: {results['xyz_map'].shape}")
    print(f"  点云: {len(results['pointcloud'].points)} 个点")

    # ========== 第五步: 保存结果 ==========
    print(f"\n【第五步】保存结果到 {args.output}/...")
    stereo.save_results(
        results=results,
        output_dir=args.output,
        save_ply=True,
        save_depth=True,
        save_disparity=True
    )
    print("✓ 所有结果已保存")

    # ========== 第六步: 可视化 (可选) ==========
    if args.visualize:
        print("\n【第六步】打开 Open3D 可视化窗口...")
        print("  提示: 按 ESC 关闭窗口")
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="BridgeDepth Point Cloud", width=1024, height=768)
            vis.add_geometry(results['pointcloud'])
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
            vis.run()
            vis.destroy_window()
        except Exception as e:
            print(f"  无法打开可视化窗口: {e}")
            print(f"  请使用其他工具查看点云: {args.output}/pointcloud.ply")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  - {args.output}/disparity.npy    (视差图)")
    print(f"  - {args.output}/depth.npy        (深度图)")
    print(f"  - {args.output}/pointcloud.ply   (点云)")
    print()


if __name__ == '__main__':
    main()
