import copy
import numpy as np
import open3d as o3d
from probreg import cpd
from probreg import filterreg
from probreg import callbacks

# load source and target point cloud
source = o3d.io.read_point_cloud('depth_outputs/d455_huojian_ce_nojgg/cloud_sam/cloud_sam_0003.ply')
# source.points = o3d.utility.Vector3dVector(np.asarray(source.points) / 1000)
source.remove_non_finite_points()

target = o3d.io.read_point_cloud('depth_outputs/d455_huojian_ce_nojgg/cloud_sam/cloud_sam_0004.ply')
# target.points = o3d.utility.Vector3dVector(np.asarray(target.points) / 1000)
target.remove_non_finite_points()
# transform target point cloud

source = source.voxel_down_sample(voxel_size=3)
target = target.voxel_down_sample(voxel_size=3)

objective_type = 'pt2pt'
# cbs = [callbacks.Open3dVisualizerCallback(source, target)]

# compute cpd registration
tf_param, _, _ = filterreg.registration_filterreg(source, target,
                                                  objective_type=objective_type,
                                                  sigma2=None,
                                                  update_sigma2=True)
R = tf_param.rot  # 3x3
angle_rad = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)
print(f"配准旋转角: {angle_rad:.6f} rad ({angle_deg:.3f}°)")




result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

# draw result
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])

# Merge point clouds and save to local file
merged_pcd = target + result
o3d.io.write_point_cloud("test.ply", merged_pcd)