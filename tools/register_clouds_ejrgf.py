import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
EJRGF_PYTHON_DIR = WORKSPACE_DIR / "EJRGF" / "src" / "python"
try:
    from EJRGF import EJRGF_register
except ImportError:
    if str(EJRGF_PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(EJRGF_PYTHON_DIR))
    from EJRGF import EJRGF_register


def parse_args():
    parser = argparse.ArgumentParser(description="Register PLY point clouds with EJRGF and save a fused voxel-downsampled cloud.")
    parser.add_argument("--input-dir", default="depth_outputs/d455_penguan_20260525/video/aruco_fusion/cloud_world_sam", help="Directory containing cloud_*.ply files.")
    parser.add_argument("--output-dir", default="depth_outputs/d455_penguan_20260525/video/ejrgf_registration", help="Output directory.")
    parser.add_argument("--registration-voxel-size", type=float, default=1.0, help="Voxel size in mm for clouds passed to EJRGF.")
    parser.add_argument("--final-voxel-size", type=float, default=1.0, help="Voxel size in mm for the final fused cloud.")
    parser.add_argument("--max-registration-points", type=int, default=30000, help="Maximum registration points kept per cloud after voxel downsampling.")
    parser.add_argument("--subgroup-size", type=int, default=10)
    parser.add_argument("--gmm-mean-local-num", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--local-sigma", type=float, default=0.0)
    parser.add_argument("--local-iteration-num", type=int, default=200)
    parser.add_argument("--global-refinement", action="store_true")
    parser.add_argument("--gmm-mean-global-num", type=int, default=1000)
    parser.add_argument("--global-sigma", type=float, default=0.0)
    parser.add_argument("--global-iteration-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--merge-downsample-every", type=int, default=0, help="Downsample the accumulating fused cloud every N frames.")
    return parser.parse_args()


def read_cloud_paths(input_dir):
    cloud_paths = sorted(Path(input_dir).glob("*.ply"))
    if not cloud_paths:
        raise FileNotFoundError(f"No .ply files found in {input_dir}")
    return cloud_paths


def load_registration_cloud(path, voxel_size, max_points, rng):
    point_cloud = o3d.io.read_point_cloud(str(path))
    if len(point_cloud.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")

    if voxel_size > 0:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.shape[0] > max_points:
        keep_indices = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep_indices]

    return torch.from_numpy(np.ascontiguousarray(points)).float().cuda()


def transform_cloud(point_cloud, transform):
    points = np.asarray(point_cloud.points, dtype=np.float64)
    colors = np.asarray(point_cloud.colors, dtype=np.float64) if point_cloud.has_colors() else None

    transformed_points = points @ transform[:3, :3].T + transform[:3, 3]
    transformed_cloud = o3d.geometry.PointCloud()
    transformed_cloud.points = o3d.utility.Vector3dVector(transformed_points)
    if colors is not None and colors.shape[0] == points.shape[0]:
        transformed_cloud.colors = o3d.utility.Vector3dVector(colors)
    return transformed_cloud


def save_transforms(output_dir, transforms, cloud_paths):
    transform_array = np.stack(transforms, axis=0)
    np.save(output_dir / "ejrgf_transforms.npy", transform_array)

    transform_records = []
    for cloud_path, transform in zip(cloud_paths, transforms):
        transform_records.append({
            "cloud": str(cloud_path),
            "transform": transform.tolist(),
        })
    with open(output_dir / "ejrgf_transforms.json", "w", encoding="utf-8") as file:
        json.dump(transform_records, file, indent=2)


def normalize_transforms_to_first_frame(transforms):
    if not transforms:
        return transforms

    first_inverse = np.linalg.inv(transforms[0])
    normalized_transforms = [first_inverse @ transform for transform in transforms]
    normalized_transforms[0] = np.eye(4, dtype=np.float64)
    return normalized_transforms


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EJRGF requires CUDA, but torch.cuda.is_available() is False in the current environment.")

    input_dir = WORKSPACE_DIR / args.input_dir
    output_dir = WORKSPACE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cloud_paths = read_cloud_paths(input_dir)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {len(cloud_paths)} clouds for EJRGF registration...")
    registration_tensors = []
    for index, cloud_path in enumerate(cloud_paths):
        tensor = load_registration_cloud(
            cloud_path,
            voxel_size=args.registration_voxel_size,
            max_points=args.max_registration_points,
            rng=rng,
        )
        registration_tensors.append(tensor)
        print(f"[{index + 1:04d}/{len(cloud_paths):04d}] {cloud_path.name}: {tensor.shape[0]} registration points")

    print("Running EJRGF registration...")
    transform_tensors = EJRGF_register(
        registration_tensors,
        args.subgroup_size,
        args.gmm_mean_local_num,
        args.epsilon,
        args.local_sigma,
        args.local_iteration_num,
        args.global_refinement,
        args.gmm_mean_global_num,
        args.global_sigma,
        args.global_iteration_num,
    )
    raw_transforms = [transform.detach().cpu().numpy().astype(np.float64) for transform in transform_tensors]
    np.save(output_dir / "ejrgf_transforms_raw.npy", np.stack(raw_transforms, axis=0))

    transforms = normalize_transforms_to_first_frame(raw_transforms)
    print("Normalized transforms so the first output frame stays aligned with the first input cloud.")
    save_transforms(output_dir, transforms, cloud_paths)

    print("Applying transforms to full-resolution clouds and fusing...")
    fused_cloud = o3d.geometry.PointCloud()
    for index, (cloud_path, transform) in enumerate(zip(cloud_paths, transforms)):
        point_cloud = o3d.io.read_point_cloud(str(cloud_path))
        transformed_cloud = transform_cloud(point_cloud, transform)
        fused_cloud += transformed_cloud
        if args.merge_downsample_every > 0 and (index + 1) % args.merge_downsample_every == 0:
            fused_cloud = fused_cloud.voxel_down_sample(voxel_size=args.final_voxel_size)
        print(f"[{index + 1:04d}/{len(cloud_paths):04d}] merged {cloud_path.name}")

    fused_cloud = fused_cloud.voxel_down_sample(voxel_size=args.final_voxel_size)
    output_cloud_path = output_dir / "merged_cloud_ejrgf_voxel.ply"
    o3d.io.write_point_cloud(str(output_cloud_path), fused_cloud)
    print(f"Saved final fused cloud: {output_cloud_path}")
    print(f"Final points: {len(fused_cloud.points)}")


if __name__ == "__main__":
    main()
