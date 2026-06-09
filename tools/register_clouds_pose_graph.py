import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


WORKSPACE_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Open3D pose graph optimization on a folder of point clouds and save one fused result."
    )
    parser.add_argument(
        "--input-dir",
        default="depth_outputs/d455_penguan_20260525/video/aruco_fusion/cloud_world_sam",
        help="Directory containing input point clouds.",
    )
    parser.add_argument(
        "--output-path",
        default="depth_outputs/d455_penguan_20260525/video/pose_graph_registration/pcd_combined.ply",
        help="Path for the final fused point cloud.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Voxel size in mm for registration and final fusion. Use 0 to disable downsampling.",
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=10.0,
        help="Maximum correspondence distance in mm for pairwise ICP.",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=300,
        help="Maximum ICP iterations for each pairwise registration.",
    )
    parser.add_argument(
        "--loop-closure-stride",
        type=int,
        default=30,
        help="Add non-neighbor loop closure edges every N frames. Use 0 to disable loop closures.",
    )
    parser.add_argument(
        "--preference-loop-closure",
        type=float,
        default=0.1,
        help="Open3D global optimization preference for uncertain loop closure edges.",
    )
    parser.add_argument(
        "--edge-prune-threshold",
        type=float,
        default=0.25,
        help="Open3D global optimization edge prune threshold.",
    )
    parser.add_argument(
        "--reference-node",
        type=int,
        default=0,
        help="Pose graph node kept fixed during global optimization.",
    )
    return parser.parse_args()


def natural_sort_key(path):
    parts = []
    for text in path.stem.replace("-", "_").split("_"):
        if text.isdigit():
            parts.append((0, int(text)))
        else:
            parts.append((1, text))
    return parts, path.suffix


def read_cloud_paths(input_dir):
    point_cloud_extensions = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}
    input_path = Path(input_dir)
    cloud_paths = sorted(
        [
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in point_cloud_extensions
        ],
        key=natural_sort_key,
    )
    if not cloud_paths:
        raise FileNotFoundError(f"No point cloud files found in {input_path}")
    return cloud_paths


def load_cloud(path, voxel_size):
    point_cloud = o3d.io.read_point_cloud(str(path))
    if len(point_cloud.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")

    if voxel_size > 0:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    if len(point_cloud.points) == 0:
        raise ValueError(f"Empty point cloud after voxel downsampling: {path}")

    if not point_cloud.has_normals():
        normal_radius = voxel_size * 2.0 if voxel_size > 0 else 2.0
        point_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
    return point_cloud


def pairwise_registration(source, target, args):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=args.max_iteration
    )
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        args.max_correspondence_distance,
        init=np.eye(4, dtype=np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=criteria,
    )
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        args.max_correspondence_distance,
        result.transformation,
    )
    return result.transformation, information, result.fitness, result.inlier_rmse


def should_add_loop_edge(source_id, target_id, stride):
    if stride <= 0:
        return False
    if target_id <= source_id + 1:
        return False
    return (target_id - source_id) % stride == 0


def build_pose_graph(point_clouds, cloud_paths, args):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.eye(4, dtype=np.float64)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for source_id in range(len(point_clouds)):
        for target_id in range(source_id + 1, len(point_clouds)):
            is_neighbor = target_id == source_id + 1
            is_loop = should_add_loop_edge(source_id, target_id, args.loop_closure_stride)
            if not is_neighbor and not is_loop:
                continue

            transformation, information, fitness, rmse = pairwise_registration(
                point_clouds[source_id],
                point_clouds[target_id],
                args,
            )
            print(
                f"{cloud_paths[source_id].name} -> {cloud_paths[target_id].name}: "
                f"fitness={fitness:.6f}, rmse={rmse:.6f}, uncertain={not is_neighbor}"
            )

            if is_neighbor:
                odometry = transformation @ odometry
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)
                    )
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation,
                        information,
                        uncertain=False,
                    )
                )
            else:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation,
                        information,
                        uncertain=True,
                    )
                )
    return pose_graph


def optimize_pose_graph(pose_graph, args):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=args.max_correspondence_distance,
        edge_prune_threshold=args.edge_prune_threshold,
        preference_loop_closure=args.preference_loop_closure,
        reference_node=args.reference_node,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )


def fuse_clouds(cloud_paths, poses, voxel_size):
    fused_cloud = o3d.geometry.PointCloud()
    for cloud_path, pose in zip(cloud_paths, poses):
        point_cloud = o3d.io.read_point_cloud(str(cloud_path))
        if len(point_cloud.points) == 0:
            continue
        point_cloud.transform(pose)
        fused_cloud += point_cloud

    if voxel_size > 0 and len(fused_cloud.points) > 0:
        fused_cloud = fused_cloud.voxel_down_sample(voxel_size=voxel_size)
    return fused_cloud


def main():
    args = parse_args()
    input_dir = WORKSPACE_DIR / args.input_dir
    output_path = WORKSPACE_DIR / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud_paths = read_cloud_paths(input_dir)
    print(f"Loaded {len(cloud_paths)} input paths from {input_dir}")

    point_clouds = []
    for index, cloud_path in enumerate(cloud_paths, start=1):
        point_cloud = load_cloud(cloud_path, args.voxel_size)
        point_clouds.append(point_cloud)
        print(f"[{index:04d}/{len(cloud_paths):04d}] {cloud_path.name}: {len(point_cloud.points)} registration points")

    print("Building pose graph...")
    pose_graph = build_pose_graph(point_clouds, cloud_paths, args)
    print(f"Pose graph nodes: {len(pose_graph.nodes)}, edges: {len(pose_graph.edges)}")

    print("Running Open3D global pose graph optimization...")
    optimize_pose_graph(pose_graph, args)

    print("Applying optimized poses and fusing full-resolution clouds...")
    poses = [node.pose for node in pose_graph.nodes]
    fused_cloud = fuse_clouds(cloud_paths, poses, args.voxel_size)
    o3d.io.write_point_cloud(str(output_path), fused_cloud)
    print(f"Saved final fused cloud: {output_path}")
    print(f"Final points: {len(fused_cloud.points)}")


if __name__ == "__main__":
    main()
