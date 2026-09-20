#!/usr/bin/env python3
"""Smooth angle_deg and save the data and a comparison plot."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / (
    "depth_outputs/d455_penguan_20260525/zhuantai_angle/32shun/"
    "rotation_angles.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / (
    "depth_outputs/d455_penguan_20260525/zhuantai_angle/32shun/vis_pinghua"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 rotation_angles.csv 中的 angle_deg 做高斯加权平滑。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入 CSV（默认：{DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="高斯核标准差，数值越大越平滑（默认：2.0）",
    )
    return parser.parse_args()


def gaussian_weighted_smoothing(values: list[float], sigma: float) -> list[float]:
    """Smooth values with a normalized Gaussian kernel."""
    if not values:
        raise ValueError("没有可用于平滑的数据")
    if sigma <= 0.0:
        raise ValueError("--sigma 必须大于 0")

    radius = max(1, math.ceil(3.0 * sigma))
    weights = [
        math.exp(-(offset * offset) / (2.0 * sigma * sigma))
        for offset in range(-radius, radius + 1)
    ]

    smoothed = []
    for index in range(len(values)):
        weighted_sum = 0.0
        valid_weight_sum = 0.0
        for offset, weight in zip(range(-radius, radius + 1), weights):
            neighbor = index + offset
            if 0 <= neighbor < len(values):
                weighted_sum += weight * values[neighbor]
                valid_weight_sum += weight
        smoothed.append(weighted_sum / valid_weight_sum)
    return smoothed


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if "angle_deg" not in fieldnames:
            raise ValueError(f"CSV 中没有 angle_deg 列，现有列：{fieldnames}")
        rows = list(reader)

    if not rows:
        raise ValueError("输入 CSV 没有数据行")

    angles = []
    for csv_row, row in enumerate(rows, start=2):
        try:
            angle = float(row["angle_deg"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {csv_row} 行的 angle_deg 不是有效数字") from exc
        if not math.isfinite(angle):
            raise ValueError(f"第 {csv_row} 行的 angle_deg 不是有限数值")
        angles.append(angle)

    return rows, fieldnames, angles


def save_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    smoothed: list[float],
) -> None:
    output_field = "angle_deg_smoothed"
    output_fieldnames = [name for name in fieldnames if name != output_field]
    output_fieldnames.append(output_field)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fieldnames)
        writer.writeheader()
        for row, value in zip(rows, smoothed):
            output_row = dict(row)
            output_row[output_field] = f"{value:.10f}"
            writer.writerow(output_row)


def save_plot(
    path: Path,
    rows: list[dict[str, str]],
    angles: list[float],
    smoothed: list[float],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        x_values = [int(row["frame_index"]) for row in rows]
        x_label = "Frame index"
    except (KeyError, TypeError, ValueError):
        x_values = list(range(len(rows)))
        x_label = "Data index"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(
        x_values,
        angles,
        color="#8c8c8c",
        linewidth=1.2,
        alpha=0.75,
        label="Original",
    )
    ax.plot(
        x_values,
        smoothed,
        color="#d62728",
        linewidth=2.2,
        label="Smoothed",
    )
    ax.set_title("Rotation Angle: Original vs. Smoothed")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_original_plot(
    path: Path,
    rows: list[dict[str, str]],
    angles: list[float],
) -> None:
    """Save an original-data-only plot using the comparison plot style."""
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        x_values = [int(row["frame_index"]) for row in rows]
        x_label = "Frame index"
    except (KeyError, TypeError, ValueError):
        x_values = list(range(len(rows)))
        x_label = "Data index"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(
        x_values,
        angles,
        color="#8c8c8c",
        linewidth=1.2,
        alpha=0.75,
        label="Original",
    )
    ax.set_title("Rotation Angle: Original")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.sigma <= 0.0:
        raise ValueError("--sigma 必须大于 0")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames, angles = read_csv(input_path)
    smoothed = gaussian_weighted_smoothing(angles, args.sigma)

    csv_path = output_dir / "rotation_angles_smoothed.csv"
    plot_path = output_dir / "rotation_angles_comparison.png"
    original_plot_path = output_dir / "rotation_angles_original.png"
    save_csv(csv_path, rows, fieldnames, smoothed)
    save_plot(plot_path, rows, angles, smoothed)
    save_original_plot(original_plot_path, rows, angles)

    print(f"已处理 {len(rows)} 条 angle_deg 数据")
    print(f"高斯平滑 sigma：{args.sigma:g}")
    print(f"平滑 CSV：{csv_path}")
    print(f"对比折线图：{plot_path}")
    print(f"平滑前折线图：{original_plot_path}")


if __name__ == "__main__":
    main()
