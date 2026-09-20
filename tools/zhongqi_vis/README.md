# Zhongqi point-cloud evaluation

This tool evaluates the already aligned `gt.ply` and `recon.ply` point clouds.
By default, coordinates are interpreted as millimetres, voxel downsampling uses
`1 mm`, and precision/recall use a fixed `5 mm` nearest-neighbour threshold.

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/zhaoyibin/miniforge3/envs/vggt/bin/python \
tools/zhongqi_vis/evaluate_point_clouds.py
```

To change the voxel size, threshold, or inputs:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/zhaoyibin/miniforge3/envs/vggt/bin/python \
tools/zhongqi_vis/evaluate_point_clouds.py \
  --gt huojian/zhongqi_149vis/gt.ply \
  --recon huojian/zhongqi_149vis/recon.ply \
  --voxel-size 1.0 \
  --threshold 5.0 \
  --unit mm
```

Definitions:

- Precision: fraction of reconstruction points whose nearest GT point is within
  the threshold.
- Recall: fraction of GT points whose nearest reconstruction point is within the
  threshold.
- Chamfer distance (mean L1): the average of the two directional mean Euclidean
  nearest-neighbour distances.

Outputs are written to `huojian/zhongqi_149vis/evaluation/`:

- `metrics.json`: parameters, point counts, Chamfer variants, precision, recall,
  and F-score.
- `gt_downsampled.ply`, `recon_downsampled.ply`: downsampled source clouds.
- `accuracy_visualization.ply`: reconstruction-to-GT errors; zero error is white,
  errors below the threshold transition linearly from white to red, and errors
  at or above the threshold are pure red.
- `recall_visualization.ply`: GT-to-reconstruction errors using the same
  white-to-red mapping.
- `evaluation.png`: accuracy and recall point-cloud panels, with a shared
  white-to-red error color bar on the right.
