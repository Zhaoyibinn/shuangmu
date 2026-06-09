import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d


WORKSPACE_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register sequential PLY point clouds with Open3D GICP and save the fused cloud."
    )
    parser.add_argument(
        "--input-dir",
        default="depth_outputs/d455_penguan_20260525/video/aruco_fusion/cloud_aruco_sam",
        help="Directory containing .ply files.",
    )
    parser.add_argument(
        "--output-dir",
        default="depth_outputs/d455_penguan_20260525/video/gicp_registration",
        help="Output directory.",
    )
    parser.add_argument(
        "--registration-voxel-size",
        type=float,
        default=1.0,
        help="Voxel size in mm for clouds passed to GICP.",
    )
    parser.add_argument(
        "--final-voxel-size",
        type=float,
        default=1.0,
        help="Voxel size in mm for the final fused cloud.",
    )
    parser.add_argument(
        "--max-registration-points",
        type=int,
        default=300000,
        help="Maximum registration points kept per cloud after voxel downsampling.",
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=10.0,
        help="Maximum correspondence distance in mm for GICP.",
    )
    parser.add_argument(
        "--gicp-epsilon",
        type=float,
        default=1e-3,
        help="Epsilon value for Open3D TransformationEstimationForGeneralizedICP.",
    )
    parser.add_argument("--max-iteration", type=int, default=100)
    parser.add_argument("--relative-fitness", type=float, default=1e-6)
    parser.add_argument("--relative-rmse", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--merge-downsample-every",
        type=int,
        default=10,
        help="Downsample the accumulating fused cloud every N frames. Use 0 to only downsample at the end.",
    )
    parser.add_argument(
        "--skip-registered-frames",
        action="store_true",
        help="Do not save each transformed full-resolution frame.",
    )
    return parser.parse_args()


def read_cloud_paths(input_dir):
    cloud_paths = sorted(Path(input_dir).glob("*.ply"))
    if not cloud_paths:
        raise FileNotFoundError(f"No .ply files found in {input_dir}")
    return cloud_paths


def load_cloud(path):
    point_cloud = o3d.io.read_point_cloud(str(path))
    if len(point_cloud.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return point_cloud


def prepare_registration_cloud(path, voxel_size, max_points, rng):
    point_cloud = load_cloud(path)

    if voxel_size > 0:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    point_count = len(point_cloud.points)
    if point_count == 0:
        raise ValueError(f"Empty registration cloud after downsampling: {path}")

    if max_points > 0 and point_count > max_points:
        keep_indices = rng.choice(point_count, size=max_points, replace=False)
        point_cloud = point_cloud.select_by_index(keep_indices.tolist())

    return point_cloud


def transform_cloud(point_cloud, transform):
    points = np.asarray(point_cloud.points, dtype=np.float64)
    colors = np.asarray(point_cloud.colors, dtype=np.float64) if point_cloud.has_colors() else None

    transformed_points = points @ transform[:3, :3].T + transform[:3, 3]
    transformed_cloud = o3d.geometry.PointCloud()
    transformed_cloud.points = o3d.utility.Vector3dVector(transformed_points)
    if colors is not None and colors.shape[0] == points.shape[0]:
        transformed_cloud.colors = o3d.utility.Vector3dVector(colors)
    return transformed_cloud


def run_gicp(source, target, args):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=args.relative_fitness,
        relative_rmse=args.relative_rmse,
        max_iteration=args.max_iteration,
    )
    estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
        epsilon=args.gicp_epsilon
    )
    return o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        args.max_correspondence_distance,
        np.eye(4, dtype=np.float64),
        estimation,
        criteria,
    )


def save_transforms(output_dir, transforms, metrics, cloud_paths):
    transform_array = np.stack(transforms, axis=0)
    np.save(output_dir / "gicp_transforms.npy", transform_array)

    transform_records = []
    for cloud_path, transform, metric in zip(cloud_paths, transforms, metrics):
        transform_records.append({
            "cloud": str(cloud_path),
            "transform": transform.tolist(),
            "fitness": metric["fitness"],
            "inlier_rmse": metric["inlier_rmse"],
        })
    with open(output_dir / "gicp_transforms.json", "w", encoding="utf-8") as file:
        json.dump(transform_records, file, indent=2)


def save_pairwise_transforms(output_dir, pairwise_transforms, cloud_paths):
    transform_array = np.stack(pairwise_transforms, axis=0)
    np.save(output_dir / "gicp_pairwise_transforms.npy", transform_array)

    transform_records = []
    for index, (cloud_path, transform) in enumerate(zip(cloud_paths, pairwise_transforms)):
        target_cloud = None if index == 0 else str(cloud_paths[index - 1])
        transform_records.append({
            "source_cloud": str(cloud_path),
            "target_cloud": target_cloud,
            "transform_source_to_target": transform.tolist(),
        })
    with open(output_dir / "gicp_pairwise_transforms.json", "w", encoding="utf-8") as file:
        json.dump(transform_records, file, indent=2)


def save_metrics(output_dir, metrics):
    fieldnames = ["index", "source_cloud", "target_cloud", "fitness", "inlier_rmse"]
    with open(output_dir / "gicp_metrics.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def main():
    args = parse_args()

    input_dir = WORKSPACE_DIR / args.input_dir
    output_dir = WORKSPACE_DIR / args.output_dir
    registered_dir = output_dir / "registered_clouds"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_registered_frames:
        registered_dir.mkdir(parents=True, exist_ok=True)

    cloud_paths = read_cloud_paths(input_dir)
    rng = np.random.default_rng(args.seed)

    print(f"Loading first frame as sequence anchor: {cloud_paths[0].name}")
    previous_registration_cloud = prepare_registration_cloud(
        cloud_paths[0],
        voxel_size=args.registration_voxel_size,
        max_points=args.max_registration_points,
        rng=rng,
    )
    print(f"First frame registration points: {len(previous_registration_cloud.points)}")

    transforms = [np.eye(4, dtype=np.float64)]
    pairwise_transforms = [np.eye(4, dtype=np.float64)]
    metrics = [{
        "index": 0,
        "source_cloud": str(cloud_paths[0]),
        "target_cloud": "",
        "fitness": 1.0,
        "inlier_rmse": 0.0,
    }]

    for index, cloud_path in enumerate(cloud_paths[1:], start=1):
        source_cloud = prepare_registration_cloud(
            cloud_path,
            voxel_size=args.registration_voxel_size,
            max_points=args.max_registration_points,
            rng=rng,
        )
        result = run_gicp(source_cloud, previous_registration_cloud, args)
        pairwise_transform = result.transformation.astype(np.float64)
        cumulative_transform = transforms[-1] @ pairwise_transform
        pairwise_transforms.append(pairwise_transform)
        transforms.append(cumulative_transform)
        metrics.append({
            "index": index,
            "source_cloud": str(cloud_path),
            "target_cloud": str(cloud_paths[index - 1]),
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
        })
        print(
            f"[{index + 1:04d}/{len(cloud_paths):04d}] registered {cloud_path.name} -> "
            f"{cloud_paths[index - 1].name}: "
            f"fitness={result.fitness:.6f}, rmse={result.inlier_rmse:.6f}"
        )
        previous_registration_cloud = source_cloud

    save_transforms(output_dir, transforms, metrics, cloud_paths)
    save_pairwise_transforms(output_dir, pairwise_transforms, cloud_paths)
    save_metrics(output_dir, metrics)

    print("Applying GICP transforms to full-resolution clouds and fusing...")
    fused_cloud = o3d.geometry.PointCloud()
    for index, (cloud_path, transform) in enumerate(zip(cloud_paths, transforms)):
        point_cloud = load_cloud(cloud_path)
        transformed_cloud = transform_cloud(point_cloud, transform)

        if not args.skip_registered_frames:
            registered_path = registered_dir / f"registered_{index:04d}.ply"
            o3d.io.write_point_cloud(str(registered_path), transformed_cloud)

        fused_cloud += transformed_cloud
        if args.merge_downsample_every > 0 and (index + 1) % args.merge_downsample_every == 0:
            fused_cloud = fused_cloud.voxel_down_sample(voxel_size=args.final_voxel_size)
        print(f"[{index + 1:04d}/{len(cloud_paths):04d}] merged {cloud_path.name}")

    if args.final_voxel_size > 0:
        fused_cloud = fused_cloud.voxel_down_sample(voxel_size=args.final_voxel_size)

    output_cloud_path = output_dir / "merged_cloud_gicp_voxel.ply"
    o3d.io.write_point_cloud(str(output_cloud_path), fused_cloud)
    print(f"Saved final fused cloud: {output_cloud_path}")
    print(f"Final points: {len(fused_cloud.points)}")


if __name__ == "__main__":
    main()
