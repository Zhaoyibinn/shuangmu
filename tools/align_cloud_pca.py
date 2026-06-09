import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_INPUT_PATH = (
    "depth_outputs/d455_penguan_20260525/video/aruco_fusion/"
    "pose_graph_registration/merged_cloud_pose_graph_sam_cut.ply"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move a point-cloud centroid to the origin and align its PCA axes to XYZ."
    )
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Input point-cloud path.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Aligned point-cloud path. Defaults to <input>_pca_aligned.ply.",
    )
    parser.add_argument(
        "--axis-output-path",
        default=None,
        help="Coordinate-frame point-cloud path. Defaults to <output>_axes.ply.",
    )
    parser.add_argument(
        "--transform-path",
        default=None,
        help="Output JSON path for the PCA axes and input-to-aligned transform.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Voxel size in mm applied before centering and PCA alignment.",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=100.0,
        help="Coordinate-frame axis length in mm. Use 0 to disable it.",
    )
    parser.add_argument(
        "--axis-points",
        type=int,
        default=30000,
        help="Number of colored points sampled on the coordinate frame.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write an ASCII PLY instead of the default binary PLY.",
    )
    return parser.parse_args()


def orient_pca_basis(eigenvectors):
    basis = eigenvectors.copy()

    # Resolve the sign ambiguity of the first two PCA axes deterministically.
    for axis_index in range(2):
        axis = basis[:, axis_index]
        dominant_index = int(np.argmax(np.abs(axis)))
        if axis[dominant_index] < 0:
            basis[:, axis_index] *= -1.0

    # Construct the final axis from the first two so the basis is right-handed.
    basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    basis[:, 2] /= np.linalg.norm(basis[:, 2])
    return basis


def compute_pca_transform(points):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point array must have shape (N, 3)")
    if len(points) < 3:
        raise ValueError("At least three points are required for PCA alignment")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinite coordinates")

    centroid = points.mean(axis=0)
    centered_points = points - centroid
    covariance = centered_points.T @ centered_points / (len(points) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_order]
    eigenvectors = eigenvectors[:, descending_order]
    basis = orient_pca_basis(eigenvectors)

    # Columns of basis are the PCA axes in the input coordinate system.
    # The transpose rotates those axes onto output X, Y, and Z.
    rotation = basis.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return centroid, eigenvalues, basis, transform


def default_output_path(input_path):
    return input_path.with_name("{}_pca_aligned.ply".format(input_path.stem))


def default_transform_path(output_path):
    return output_path.with_name("{}_transform.json".format(output_path.stem))


def default_axis_output_path(output_path):
    return output_path.with_name("{}_axes.ply".format(output_path.stem))


def create_coordinate_frame_cloud(size, number_of_points):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size,
        origin=np.zeros(3, dtype=np.float64),
    )
    coordinate_frame_cloud = coordinate_frame.sample_points_uniformly(
        number_of_points=number_of_points,
        use_triangle_normal=False,
    )
    colors = np.asarray(coordinate_frame_cloud.colors)
    np.clip(colors, 0.0, 1.0, out=colors)
    return coordinate_frame_cloud


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else default_output_path(input_path)
    )
    transform_path = (
        Path(args.transform_path)
        if args.transform_path is not None
        else default_transform_path(output_path)
    )
    axis_output_path = (
        Path(args.axis_output_path)
        if args.axis_output_path is not None
        else default_axis_output_path(output_path)
    )

    if not input_path.is_file():
        raise FileNotFoundError("Input point cloud does not exist: {}".format(input_path))
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be greater than zero")
    if args.axis_size < 0:
        raise ValueError("--axis-size must be non-negative")
    if args.axis_size > 0 and args.axis_points <= 0:
        raise ValueError("--axis-points must be greater than zero")

    point_cloud = o3d.io.read_point_cloud(str(input_path))
    input_point_count = len(point_cloud.points)
    if input_point_count == 0:
        raise ValueError("Input point cloud is empty: {}".format(input_path))

    point_cloud = point_cloud.voxel_down_sample(voxel_size=args.voxel_size)
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(
            "Point cloud is empty after {:.6g} mm voxel downsampling: {}".format(
                args.voxel_size,
                input_path,
            )
        )

    centroid, eigenvalues, basis, transform = compute_pca_transform(points)
    point_cloud.transform(transform)
    aligned_centroid = np.asarray(point_cloud.points).mean(axis=0)

    coordinate_frame_cloud = None
    axis_point_count = 0
    if args.axis_size > 0:
        coordinate_frame_cloud = create_coordinate_frame_cloud(
            size=args.axis_size,
            number_of_points=args.axis_points,
        )
        axis_point_count = len(coordinate_frame_cloud.points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    if coordinate_frame_cloud is not None:
        axis_output_path.parent.mkdir(parents=True, exist_ok=True)

    write_ok = o3d.io.write_point_cloud(
        str(output_path),
        point_cloud,
        write_ascii=args.ascii,
    )
    if not write_ok:
        raise OSError("Failed to write aligned point cloud: {}".format(output_path))

    if coordinate_frame_cloud is not None:
        axis_write_ok = o3d.io.write_point_cloud(
            str(axis_output_path),
            coordinate_frame_cloud,
            write_ascii=args.ascii,
        )
        if not axis_write_ok:
            raise OSError(
                "Failed to write coordinate-frame point cloud: {}".format(
                    axis_output_path
                )
            )

    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "axis_output_path": (
            str(axis_output_path) if coordinate_frame_cloud is not None else None
        ),
        "input_point_count": int(input_point_count),
        "downsampled_point_count": int(len(points)),
        "axis_point_count": int(axis_point_count),
        "output_point_count": int(len(point_cloud.points)),
        "voxel_size_mm": float(args.voxel_size),
        "coordinate_frame_axis_size_mm": float(args.axis_size),
        "coordinate_frame_colors": {
            "x_axis": "red",
            "y_axis": "green",
            "z_axis": "blue",
        },
        "original_centroid": centroid.tolist(),
        "aligned_cloud_centroid": aligned_centroid.tolist(),
        "pca_eigenvalues_descending": eigenvalues.tolist(),
        "pca_axes_columns_in_input_coordinates": basis.tolist(),
        "transform_input_to_aligned": transform.tolist(),
    }
    transform_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Loaded {} points from {}".format(input_point_count, input_path))
    print(
        "Downsampled to {} points with {:.6g} mm voxels".format(
            len(points),
            args.voxel_size,
        )
    )
    print("Original centroid: {}".format(np.array2string(centroid, precision=6)))
    print("PCA eigenvalues: {}".format(np.array2string(eigenvalues, precision=6)))
    print(
        "Aligned centroid: {}".format(
            np.array2string(aligned_centroid, precision=6)
        )
    )
    if axis_point_count > 0:
        print(
            "Created {} coordinate-frame points with {:.6g} mm axes".format(
                axis_point_count,
                args.axis_size,
            )
        )
        print("Axis colors: X=red, Y=green, Z=blue")
        print("Saved coordinate frame: {}".format(axis_output_path))
    print("Saved aligned point cloud: {}".format(output_path))
    print("Saved transform metadata: {}".format(transform_path))


if __name__ == "__main__":
    main()
