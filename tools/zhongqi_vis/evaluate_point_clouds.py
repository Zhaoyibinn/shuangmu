#!/usr/bin/env python3
"""Voxel-downsample and evaluate two aligned point clouds.

The reconstruction is treated as the prediction and ``gt.ply`` as ground truth.
Distances are Euclidean nearest-neighbour distances in the input coordinate unit.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree


CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if CJK_FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(CJK_FONT_PATH))
    CJK_FONT_NAME = font_manager.FontProperties(fname=str(CJK_FONT_PATH)).get_name()
    plt.rcParams["font.family"] = CJK_FONT_NAME
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = REPO_ROOT / "huojian/zhongqi_149vis/gt.ply"
DEFAULT_RECON = REPO_ROOT / "huojian/zhongqi_149vis/recon.ply"
DEFAULT_OUTPUT = REPO_ROOT / "huojian/zhongqi_149vis/evaluation"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Voxel-downsample GT/reconstruction point clouds, compute Chamfer "
            "distance and thresholded precision/recall, and save visualizations."
        )
    )
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT, help="Ground-truth PLY.")
    parser.add_argument(
        "--recon", type=Path, default=DEFAULT_RECON, help="Reconstructed PLY."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory."
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Voxel size in the input coordinate unit (default: 1.0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Correct-match distance threshold (default: 5.0).",
    )
    parser.add_argument(
        "--unit", default="mm", help="Unit label written to results (default: mm)."
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=50000,
        help="Maximum plotted points per cloud; PLY outputs always contain all points.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Plot sampling seed.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    return parser.parse_args()


def load_and_downsample(path, voxel_size):
    if not path.is_file():
        raise FileNotFoundError("Point cloud does not exist: {}".format(path))
    cloud = o3d.io.read_point_cloud(str(path))
    input_count = len(cloud.points)
    if input_count == 0:
        raise ValueError("Point cloud is empty or unreadable: {}".format(path))
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("Point cloud became empty after downsampling: {}".format(path))
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinite coordinates: {}".format(path))
    return cloud, points, input_count


def nearest_distances(source_points, target_points):
    """Return the Euclidean distance from every source point to its nearest target."""
    tree = cKDTree(target_points)
    try:
        distances, _ = tree.query(source_points, k=1, workers=-1)
    except TypeError:  # Compatibility with older SciPy versions.
        distances, _ = tree.query(source_points, k=1)
    return np.asarray(distances, dtype=np.float64)


def make_colored_cloud(points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def error_to_white_red(distances, threshold):
    """Map zero error to white and threshold-or-larger error to pure red."""
    if threshold == 0:
        ratios = (distances > 0).astype(np.float64)
    else:
        ratios = np.clip(distances / threshold, 0.0, 1.0)
    colors = np.ones((len(distances), 3), dtype=np.float64)
    colors[:, 1] = 1.0 - ratios
    colors[:, 2] = 1.0 - ratios
    return colors


def write_cloud(path, cloud):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise OSError("Failed to write point cloud: {}".format(path))


def sampled_indices(count, maximum, rng):
    if count <= maximum:
        return np.arange(count)
    return np.sort(rng.choice(count, size=maximum, replace=False))


def set_axes_equal(ax, lower, upper):
    center = (lower + upper) / 2.0
    radius = float(np.max(upper - lower)) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X轴")
    ax.set_ylabel("Y轴")
    ax.set_zlabel("Z轴")
    ax.view_init(elev=22, azim=-60)


def scatter_cloud(ax, points, colors, title, lower, upper):
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=0.35,
        marker=".",
        linewidths=0,
        depthshade=False,
    )
    ax.set_title(title, fontsize=10)
    set_axes_equal(ax, lower, upper)


def save_figure(
    path,
    gt_points,
    recon_points,
    gt_to_recon,
    recon_to_gt,
    threshold,
    precision,
    recall,
    max_plot_points,
    seed,
    dpi,
    unit,
):
    rng = np.random.default_rng(seed)
    gt_idx = sampled_indices(len(gt_points), max_plot_points, rng)
    recon_idx = sampled_indices(len(recon_points), max_plot_points, rng)
    lower = np.minimum(gt_points.min(axis=0), recon_points.min(axis=0))
    upper = np.maximum(gt_points.max(axis=0), recon_points.max(axis=0))

    gt_status_colors = error_to_white_red(gt_to_recon[gt_idx], threshold)
    recon_status_colors = error_to_white_red(recon_to_gt[recon_idx], threshold)

    fig = plt.figure(figsize=(12.5, 5.8))
    ax = fig.add_subplot(121, projection="3d")
    scatter_cloud(
        ax,
        recon_points[recon_idx],
        recon_status_colors,
        "准确率：{:.2%}\n白色（误差为0）→ 红色（误差≥阈值）".format(precision),
        lower,
        upper,
    )

    ax = fig.add_subplot(122, projection="3d")
    scatter_cloud(
        ax,
        gt_points[gt_idx],
        gt_status_colors,
        "召回率：{:.2%}\n白色（误差为0）→ 红色（误差≥阈值）".format(recall),
        lower,
        upper,
    )
    # This color bar uses exactly the same mapping as the PLY vertex colors.
    error_cmap = LinearSegmentedColormap.from_list(
        "white_to_red_error", ["#ffffff", "#ff0000"]
    )
    error_norm = Normalize(vmin=0.0, vmax=threshold if threshold > 0 else 1.0)
    scalar_mappable = ScalarMappable(norm=error_norm, cmap=error_cmap)
    scalar_mappable.set_array([])

    # Reserve the right edge for one shared vertical error scale.
    fig.tight_layout(rect=[0.0, 0.0, 0.91, 1.0])
    colorbar_axis = fig.add_axes([0.925, 0.18, 0.018, 0.64])
    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="vertical",
    )
    display_unit = "毫米" if unit.lower() == "mm" else unit
    colorbar.set_label(
        "最近邻误差（{}）\n误差≥{:.6g}{}时为红色".format(
            display_unit, threshold, display_unit
        ),
        rotation=90,
        labelpad=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be greater than zero")
    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative")
    if args.max_plot_points <= 0:
        raise ValueError("--max-plot-points must be greater than zero")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_cloud, gt_points, gt_input_count = load_and_downsample(
        args.gt.resolve(), args.voxel_size
    )
    recon_cloud, recon_points, recon_input_count = load_and_downsample(
        args.recon.resolve(), args.voxel_size
    )

    # Precision direction: reconstruction -> GT. Recall direction: GT -> reconstruction.
    recon_to_gt = nearest_distances(recon_points, gt_points)
    gt_to_recon = nearest_distances(gt_points, recon_points)
    recon_correct = recon_to_gt <= args.threshold
    gt_correct = gt_to_recon <= args.threshold
    precision = float(recon_correct.mean())
    recall = float(gt_correct.mean())
    f_score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    mean_recon_to_gt = float(recon_to_gt.mean())
    mean_gt_to_recon = float(gt_to_recon.mean())
    mean_sq_recon_to_gt = float(np.mean(recon_to_gt**2))
    mean_sq_gt_to_recon = float(np.mean(gt_to_recon**2))

    gt_downsampled_path = output_dir / "gt_downsampled.ply"
    recon_downsampled_path = output_dir / "recon_downsampled.ply"
    accuracy_eval_path = output_dir / "accuracy_visualization.ply"
    recall_eval_path = output_dir / "recall_visualization.ply"
    figure_path = output_dir / "evaluation.png"
    results_path = output_dir / "metrics.json"

    recon_colors = error_to_white_red(recon_to_gt, args.threshold)
    gt_colors = error_to_white_red(gt_to_recon, args.threshold)
    recon_eval_cloud = make_colored_cloud(recon_points, recon_colors)
    gt_eval_cloud = make_colored_cloud(gt_points, gt_colors)

    write_cloud(gt_downsampled_path, gt_cloud)
    write_cloud(recon_downsampled_path, recon_cloud)
    write_cloud(accuracy_eval_path, recon_eval_cloud)
    write_cloud(recall_eval_path, gt_eval_cloud)
    save_figure(
        figure_path,
        gt_points,
        recon_points,
        gt_to_recon,
        recon_to_gt,
        args.threshold,
        precision,
        recall,
        args.max_plot_points,
        args.seed,
        args.dpi,
        args.unit,
    )

    results = {
        "inputs": {"ground_truth": str(args.gt.resolve()), "reconstruction": str(args.recon.resolve())},
        "parameters": {
            "voxel_size": float(args.voxel_size),
            "threshold": float(args.threshold),
            "coordinate_unit": args.unit,
            "threshold_comparison": "nearest_distance <= threshold",
        },
        "point_counts": {
            "ground_truth_before_downsampling": int(gt_input_count),
            "ground_truth_after_downsampling": int(len(gt_points)),
            "reconstruction_before_downsampling": int(recon_input_count),
            "reconstruction_after_downsampling": int(len(recon_points)),
        },
        "metrics": {
            "chamfer_distance_mean_l1": 0.5 * (mean_recon_to_gt + mean_gt_to_recon),
            "chamfer_distance_sum_l1": mean_recon_to_gt + mean_gt_to_recon,
            "chamfer_distance_mean_squared_l2": 0.5 * (mean_sq_recon_to_gt + mean_sq_gt_to_recon),
            "chamfer_distance_sum_squared_l2": mean_sq_recon_to_gt + mean_sq_gt_to_recon,
            "mean_reconstruction_to_ground_truth": mean_recon_to_gt,
            "mean_ground_truth_to_reconstruction": mean_gt_to_recon,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
        },
        "definitions": {
            "precision": "fraction of reconstruction points within threshold of ground truth",
            "recall": "fraction of ground-truth points within threshold of reconstruction",
            "chamfer_distance_mean_l1": "mean of the two directional mean Euclidean distances",
        },
        "visualization_colors": {
            "mapping": "linear white-to-red by nearest-neighbor error",
            "zero_error": "white [1, 1, 1]",
            "threshold_or_larger": "red [1, 0, 0]",
            "formula_below_threshold": "RGB = [1, 1-d/threshold, 1-d/threshold]",
        },
        "outputs": {
            "ground_truth_downsampled": str(gt_downsampled_path),
            "reconstruction_downsampled": str(recon_downsampled_path),
            "accuracy_visualization": str(accuracy_eval_path),
            "recall_visualization": str(recall_eval_path),
            "evaluation_figure": str(figure_path),
        },
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Evaluation complete")
    print("  Chamfer distance (mean L1): {:.6f} {}".format(results["metrics"]["chamfer_distance_mean_l1"], args.unit))
    print("  Precision @ {:.6g} {}: {:.4%}".format(args.threshold, args.unit, precision))
    print("  Recall    @ {:.6g} {}: {:.4%}".format(args.threshold, args.unit, recall))
    print("  F-score   @ {:.6g} {}: {:.4%}".format(args.threshold, args.unit, f_score))
    print("  Results: {}".format(results_path))


if __name__ == "__main__":
    main()
