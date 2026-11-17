"""Fit a nozzle axis from robust circular sections, rather than point variance."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares


def axis_basis(axis):
    x = np.asarray(axis, dtype=float).copy()
    x /= np.linalg.norm(x)
    if x[0] < 0:
        x *= -1
    seed = np.array([0., 1., 0.]) if abs(x[1]) < .9 else np.array([0., 0., 1.])
    y = seed - seed @ x * x
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def fit_circle(points):
    points = np.asarray(points)
    xy, _, _, _ = np.linalg.lstsq(
        np.column_stack([2 * points, np.ones(len(points))]),
        np.sum(points * points, axis=1), rcond=None)
    initial = [xy[0], xy[1], np.sqrt(max(xy[2] + xy[:2] @ xy[:2], 1e-6))]
    fit = least_squares(lambda v: np.linalg.norm(points - v[:2], axis=1) - v[2],
                        initial, loss='soft_l1', f_scale=.5)
    error = np.abs(np.linalg.norm(points - fit.x[:2], axis=1) - fit.x[2])
    theta = np.mod(np.arctan2(points[:, 1] - fit.x[1], points[:, 0] - fit.x[0]), 2*np.pi)
    coverage = len(np.unique(np.floor(theta * 36 / (2*np.pi)).astype(int))) / 36.
    return fit.x, float(np.median(error)), float(np.percentile(error, 90)), coverage


def fit_sections(points, origin, basis, width=2., spacing=4.):
    local = (points - origin) @ basis
    low, high = np.percentile(local[:, 0], [2, 98])
    sections = []
    for x in np.arange(low, high, spacing):
        mask = np.abs(local[:, 0] - x) <= width / 2
        if mask.sum() < 100:
            continue
        circle, median, p90, coverage = fit_circle(local[mask, 1:])
        sections.append(dict(x_mm=float(x), center_y_mm=float(circle[0]),
                             center_z_mm=float(circle[1]), radius_mm=float(circle[2]),
                             points=int(mask.sum()), median_radial_error_mm=median,
                             p90_radial_error_mm=p90, angular_coverage=coverage,
                             quality_ok=bool(p90 <= 2.5 and median <= 1.0
                                             and coverage >= .75 and circle[2] > 5.)))
    return sections


def fit_center_line(sections, threshold=.6):
    candidates = [s for s in sections if s['quality_ok']]
    if len(candidates) < 6:
        raise ValueError('Fewer than six reliable circular sections; axis fit is not supported')
    centers = np.array([[s['x_mm'], s['center_y_mm'], s['center_z_mm']] for s in candidates])
    design = np.column_stack([centers[:, 0], np.ones(len(centers))])
    rng = np.random.default_rng(31)
    best = None
    best_score = (-1, 0.)
    for _ in range(500):
        pair = rng.choice(len(centers), 2, replace=False)
        if abs(np.diff(centers[pair, 0])[0]) < 15:
            continue
        coefficients = np.linalg.solve(design[pair], centers[pair, 1:])
        errors = np.linalg.norm(design @ coefficients - centers[:, 1:], axis=1)
        inliers = errors <= threshold
        score = (int(inliers.sum()), float(np.ptp(centers[inliers, 0])))
        if score > best_score:
            best, best_score = inliers, score
    if best is None or best.sum() < 6 or best_score[1] < 30:
        raise ValueError('No consistent center line spanning at least 30 mm')
    for _ in range(5):
        coefficients = np.linalg.lstsq(design[best], centers[best, 1:], rcond=None)[0]
        errors = np.linalg.norm(design @ coefficients - centers[:, 1:], axis=1)
        updated = errors <= threshold
        if updated.sum() < 6 or np.array_equal(best, updated):
            break
        best = updated
    for section in sections:
        predicted = np.array([section['x_mm'], 1.]) @ coefficients
        error = float(np.linalg.norm(predicted - [section['center_y_mm'], section['center_z_mm']]))
        section['center_line_error_mm'] = error
        section['used_for_axis'] = bool(section['quality_ok'] and error <= threshold)
    return coefficients


def fit_axis(points, width=2., spacing=4., iterations=20):
    centroid = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    axis = axis_basis(vectors[:, -1])[:, 0]
    initial_axis = axis.copy()
    origin = centroid.copy()
    history = []
    for _ in range(iterations):
        basis = axis_basis(axis)
        sections = fit_sections(points, origin, basis, width, spacing)
        coefficients = fit_center_line(sections)
        direction = basis @ np.r_[1., coefficients[0]]
        direction /= np.linalg.norm(direction)
        anchor = origin + basis @ np.r_[0., coefficients[1]]
        origin = anchor + direction * ((centroid - anchor) @ direction)
        change = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(axis, direction)), axis @ direction)))
        axis = direction
        history.append(dict(direction=axis.tolist(), change_deg=change,
                            used_sections=sum(s['used_for_axis'] for s in sections)))
        if change < 1e-3:
            break
    basis = axis_basis(axis)
    sections = fit_sections(points, origin, basis, width, spacing)
    fit_center_line(sections)
    transform = np.eye(4)
    transform[:3, :3] = basis.T
    transform[:3, 3] = -basis.T @ origin
    return origin, axis, initial_axis, transform, sections, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output')
    parser.add_argument('--width', type=float, default=2.)
    parser.add_argument('--spacing', type=float, default=4.)
    args = parser.parse_args()
    if args.width <= 0 or args.spacing <= 0:
        parser.error('width and spacing must be positive')
    source = Path(args.input)
    output = Path(args.output) if args.output else source.with_name(source.stem + '_section_axis_aligned.ply')
    cloud = o3d.io.read_point_cloud(str(source))
    points = np.asarray(cloud.points).copy()
    if len(points) < 1000 or not np.isfinite(points).all():
        raise ValueError('Need at least 1000 finite points')
    origin, axis, pca_axis, transform, sections, history = fit_axis(points, args.width, args.spacing)
    cloud.transform(transform)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise OSError(output)
    stem = output.with_suffix('')
    # One PLY contains both the transformed object and XYZ axes, avoiding mixed frames.
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.).sample_points_uniformly(15000)
    np.clip(np.asarray(frame.colors), 0., 1., out=np.asarray(frame.colors))
    overlay = o3d.geometry.PointCloud(cloud)
    if not overlay.has_colors():
        overlay.paint_uniform_color([.65, .65, .65])
    overlay += frame
    overlay_path = Path(str(stem) + '_with_axes.ply')
    if not o3d.io.write_point_cloud(str(overlay_path), overlay):
        raise OSError(overlay_path)
    used = [s for s in sections if s['used_for_axis']]
    metadata = dict(input_path=str(source), output_path=str(output), overlay_path=str(overlay_path),
                    units='mm', input_point_count=len(points), output_point_count=len(cloud.points),
                    axis_origin_in_input=origin.tolist(), axis_direction_in_input=axis.tolist(),
                    pca_axis_in_input=pca_axis.tolist(),
                    angle_to_old_X_deg=float(np.degrees(np.arccos(np.clip(axis[0], -1, 1)))),
                    angle_to_PCA_deg=float(np.degrees(np.arccos(np.clip(axis @ pca_axis, -1, 1)))),
                    transform_input_to_aligned=transform.tolist(),
                    transform_aligned_to_input=np.linalg.inv(transform).tolist(),
                    slice_width_mm=args.width, slice_spacing_mm=args.spacing,
                    line_inlier_threshold_mm=.6, circle_p90_threshold_mm=2.5,
                    used_sections=len(used), total_sections=len(sections),
                    used_section_span_mm=float(np.ptp([s['x_mm'] for s in used])),
                    center_line_rmse_mm=float(np.sqrt(np.mean([s['center_line_error_mm']**2 for s in used]))),
                    angle_convergence_tolerance_deg=.001,
                    converged=bool(history[-1]['change_deg'] < .001),
                    sections=sections, iterations=history)
    report = Path(str(stem) + '_transform.json')
    report.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('sections','iterations')}, indent=2), flush=True)
    print('REPORT', report, flush=True)


if __name__ == '__main__':
    main()
