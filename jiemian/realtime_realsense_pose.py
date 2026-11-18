"""D455 实时物体位姿：Fast FoundationStereo + YOLO + 刚性 ICP。

在仓库根目录运行 python jiemian/realtime_realsense_pose.py。
相机固定，移动目标；GT 点云及所有平移、深度均使用毫米。
"""

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly

from Pose.Pose_est import (
    IcpConfig, PoseEstimate, _TemporalRigidIcp, _depth_to_point_cloud,
    _filter_and_center_cloud, matrix_to_rpy_degrees, resolve_workspace_path,
)
from Recon.config_loader import load_config
from jiemian.cameras.realsense_stereo_camera import (
    RealSenseStereoCamera, STEREO_METHODS, default_checkpoint,
)
from main_pose import (
    IncrementalRotationPointEstimator, open_pose_csv, gt_axis_rotation_angles,
    save_angle_plot, save_estimate, save_rotation_angle_csv, validate_config,
)

DEFAULT_TARGET = (
    'depth_outputs/chongjian/video_shu/aruco_fusion/pose_graph_registration/'
    'merged_cloud_pose_graph_sam_pca_aligned_clean_section_axis_aligned.ply'
)


class LivePoseEstimator:
    """Use the live camera's RGB-aligned depth with main_pose's ICP pipeline."""

    def __init__(self, config, camera, keep_x_rotation=False):
        self.config, self.camera = config, camera
        target = resolve_workspace_path(config['paths']['icp_target_path'])
        if not target.is_file():
            raise FileNotFoundError(f'GT 点云不存在：{target}')
        self.icp = _TemporalRigidIcp(
            o3d.io.read_point_cloud(str(target)), IcpConfig.from_config(config['icp']))

    def read_inputs(self):
        """Capture and filter a preview frame without changing ICP tracking."""
        started = time.perf_counter()
        depth, gray, mask, right, extra = self.camera.capture()
        capture_seconds = time.perf_counter() - started
        color = extra['rgb']
        foreground = extra['foreground'] > 0
        cloud_config = self.config['point_cloud']
        valid = ((mask == 0) & foreground & np.isfinite(depth)
                 & (depth >= cloud_config['min_depth_mm'])
                 & (depth <= cloud_config['max_depth_mm']))
        brightness = self.config['reconstruction']['brightness_mask']
        if brightness['enabled']:
            # Depth is in the undistorted RGB frame, so filter RGB brightness here.
            valid &= gray >= brightness['threshold']
        return dict(color=color, full_depth=depth.copy(),
                    depth=np.where(valid, depth, 0).astype(np.float32),
                    foreground=extra['foreground'], left=extra['left'], right=right,
                    capture_seconds=capture_seconds)

    def estimate(self, index, inputs=None):
        if inputs is None:
            inputs = self.read_inputs()
        started = time.perf_counter()
        capture_seconds = inputs['capture_seconds']
        color, depth = inputs['color'], inputs['depth']
        foreground = inputs['foreground'] > 0
        cloud_config = self.config['point_cloud']
        if not foreground.any():
            return None, inputs
        stage = time.perf_counter()
        raw = _depth_to_point_cloud(depth, color, self.camera.rgb_k)
        segmentation = self.config['segmentation']
        centered, centroid = _filter_and_center_cloud(
            raw, float(cloud_config['voxel_size_mm']),
            int(segmentation['statistical_nb_neighbors']),
            float(segmentation['statistical_std_ratio']))
        timings = dict(capture=capture_seconds,
                       build_filter_center_cloud=time.perf_counter() - stage)
        initialization, result, icp_timings = self.icp.register(centered)
        timings.update(icp_timings)
        transform = np.asarray(result.transformation, dtype=np.float64)
        if result.fitness <= 0 or not np.isfinite(transform).all():
            self.icp.reset()
            raise ValueError('ICP 没有有效对应点，不能建立位姿参考')
        centering = np.eye(4)
        centering[:3, 3] = -centroid
        registered = copy.deepcopy(centered)
        registered.transform(transform)
        estimate = PoseEstimate(
            frame_index=index, frame_name=f'{index:06d}', frame_paths={},
            centered_cloud=centered, registered_cloud=registered,
            centered_to_target=transform, original_to_target=transform @ centering,
            centroid_before_translation=centroid,
            translation_mm=transform[:3, 3].copy(),
            rpy_degrees=matrix_to_rpy_degrees(transform[:3, :3]),
            initialization=initialization, icp_fitness=float(result.fitness),
            icp_rmse_mm=float(result.inlier_rmse), yolo_instance_count=None,
            yolo_mask_pixels=int(foreground.sum()), raw_point_count=len(raw.points),
            elapsed_seconds=capture_seconds + time.perf_counter() - started, stage_timings=timings)
        return estimate, inputs


def frame_images(inputs, maximum):
    """Use one fixed colour scale for the full and segmented depth images."""
    images = [cv2.cvtColor(inputs['color'], cv2.COLOR_BGR2RGB)]
    for key in ('full_depth', 'depth'):
        depth = inputs[key]
        valid = np.isfinite(depth) & (depth > 0)
        scaled = np.where(valid, np.clip(depth / maximum, 0, 1), 0)
        preview = cv2.applyColorMap(np.uint8(scaled * 255), cv2.COLORMAP_TURBO)
        preview[~valid] = 0
        images.append(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    return tuple(images)


def registration_plot(target=None, registered=None):
    figure = go.Figure()
    for cloud, name, color in ((target, 'GT', '#999999'),
                               (registered, '当前帧配准', '#ff6633')):
        if cloud is None:
            continue
        points = np.asarray(cloud.points)
        points = points[::max(1, int(np.ceil(len(points) / 6000)))]
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
            name=name, marker=dict(size=2, color=color)))
    figure.update_layout(scene=dict(aspectmode='data', xaxis_title='X / mm',
                                    yaxis_title='Y / mm', zaxis_title='Z / mm'),
                         uirevision='realsense-pose', margin=dict(l=0, r=0, t=25, b=0))
    return figure


ANGLE_FIELDS = (
    ('gt_axis_pitch_deg', '去主轴自转：绕 Y', '#27ae60'),
    ('gt_axis_yaw_deg', '去主轴自转：绕 Z', '#2980b9'),
    ('gt_axis_tilt_deg', '总体：主轴偏转角', '#8e44ad'),
)

HISTORY_CURVES = ('Y', 'Z', '总体')
CURVE_FIELDS = {
    'gt_axis_pitch_deg': 'Y', 'relative_to_first_pitch_deg': 'Y',
    'gt_axis_yaw_deg': 'Z', 'relative_to_first_yaw_deg': 'Z',
    'gt_axis_tilt_deg': '总体', 'rotation_angle_about_center_deg': '总体',
}

FULL_ANGLE_FIELDS = (
    ('relative_to_first_roll_deg', '绕 X', '#e74c3c'),
    ('relative_to_first_pitch_deg', '绕 Y', '#27ae60'),
    ('relative_to_first_yaw_deg', '绕 Z', '#2980b9'),
    ('rotation_angle_about_center_deg', '总体旋转角', '#8e44ad'),
)


def angle_fields(record):
    if all(field in record for field, _, _ in ANGLE_FIELDS[:2]):
        return tuple(item for item in ANGLE_FIELDS if item[0] in record)
    return FULL_ANGLE_FIELDS


def history_plot(records, index, selected_curves=None):
    figure = go.Figure()
    selected = set(HISTORY_CURVES if selected_curves is None else selected_curves)
    if records:
        frames = [item['frame_index'] for item in records]
        fields = [item for item in angle_fields(records[0]) if CURVE_FIELDS.get(item[0]) in selected]
        for field, name, color in fields:
            figure.add_trace(go.Scatter(
                x=frames, y=[item[field] for item in records], mode='lines+markers',
                name=name, line=dict(color=color), marker=dict(size=4)))
        index = max(0, min(index, len(records) - 1))
        figure.add_vline(x=frames[index], line_dash='dash', line_color='#555555')
        if fields:
            figure.add_trace(go.Scatter(
                x=[frames[index]] * len(fields),
                y=[records[index][field] for field, _, _ in fields],
                mode='markers', name='回放帧', marker=dict(size=10, symbol='circle-open', color='#333333')))
    figure.update_layout(xaxis_title='原始帧序号', yaxis_title='角度 / °',
                         yaxis=dict(range=[-45, 45], autorange=False, fixedrange=False),
                         hovermode='x unified', uirevision='pose-angle-history',
                         margin=dict(l=55, r=20, t=25, b=45))
    return figure


class PoseController:
    def __init__(self, config, output, device_index=0,
                 stereo_method='fast_foundation_stereo', checkpoint=None,
                 valid_iters=None, keep_x_rotation=False, inference_scale=None, emitter_enabled=False):
        self.config, self.output = config, resolve_workspace_path(output)
        self.settings = dict(device_index=device_index, stereo_method=stereo_method,
                             checkpoint=checkpoint or default_checkpoint(stereo_method),
                             valid_iters=valid_iters if valid_iters is not None else
                             (8 if stereo_method == 'fast_foundation_stereo' else 32),
                             inference_scale=inference_scale if inference_scale is not None else (
                                 .5 if stereo_method == 'foundation_stereo' else 1.))
        segmentation = config['segmentation']
        self.settings['emitter_enabled'] = bool(emitter_enabled)
        self.settings['foreground_settings'] = dict(
            weights=segmentation['yolo_model_path'], conf=segmentation['yolo_conf'],
            imgsz=segmentation['yolo_imgsz'], device=segmentation['yolo_device'])
        self.keep_x_rotation = keep_x_rotation
        self.lock = threading.RLock()
        self.camera = self.estimator = self.session = None
        self.csv_file = self.jsonl_file = None
        self.records = []
        self.display_record = None
        self.replay_index = 0
        self.playing = False
        self.history_records = None
        self.history_curves = list(HISTORY_CURVES)
        self.continuous = False
        self.preview_images = (None, None, None)
        self.captured_images = (None, None, None)
        self.detail = None
        self.plot = registration_plot()
        self.status = '请启动相机，再单帧或连续拍摄。首次成功估计自动建立参考位姿。'

    def render(self):
        record = self.display_record
        rows = [] if record is None else [[
            record[f'relative_to_first_{key}'] for key in
            ('tx_mm', 'ty_mm', 'tz_mm')
        ] + [record.get('gt_axis_pitch_deg', record.get('relative_to_first_pitch_deg')),
             record.get('gt_axis_yaw_deg', record.get('relative_to_first_yaw_deg')),
             record.get('gt_axis_tilt_deg', record.get('rotation_angle_about_center_deg')), record['icp_fitness'],
             record['icp_inlier_rmse_mm']]]
        status = self.status + (f'\n保存目录：{self.session}' if self.session else '')
        return (*self.preview_images, *self.captured_images, self.plot.to_json(), rows, self.detail,
                status, self.continuous,
                gr.update(value=self.replay_index, maximum=max(1, len(self.playback_records) - 1)),
                self.playing, history_plot(self.playback_records, self.replay_index, self.history_curves).to_json())

    def set_history_curves(self, selected):
        with self.lock:
            self.history_curves = [curve for curve in (selected or []) if curve in HISTORY_CURVES]
            return history_plot(self.playback_records, self.replay_index, self.history_curves).to_json()

    def _history_render(self, include_preview=False):
        # Replaying angles must not reload the latest captured images or 3D scene.
        result = self.render()
        previews = result[:3] if include_preview else tuple(gr.skip() for _ in range(3))
        return (*previews, *(gr.skip() for _ in range(6)), *result[9:])

    def _create_session(self):
        session = self.output / datetime.now().strftime('pose_%Y%m%d_%H%M%S_%f')
        session.mkdir(parents=True, exist_ok=False)
        directories = {name: session / name for name in
                       ('centered', 'registered', 'poses', 'color', 'full_depth_mm',
                        'depth_mm', 'foreground', 'left', 'right')}
        for directory in directories.values():
            directory.mkdir()
        metadata = dict(config=self.config, device=self.camera.detail,
                        keep_x_rotation=self.keep_x_rotation,
                        rotation_removal='none' if self.keep_x_rotation else 'GT-X post-ICP axis angles only',
                        symmetry_axis_gt=[1., 0., 0.], symmetry_axis_origin_gt_mm=[0., 0., 0.],
                        relative_pose_convention='inverse(first camera-to-GT pose) @ current camera-to-GT pose',
                        rotation_parameterization='unconstrained six-DoF ICP',
                        brightness_coordinate_system='undistorted_rgb', units='mm / degrees')
        (session / 'session.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        csv_file, csv_writer = open_pose_csv(session / 'poses.csv')
        try:
            jsonl_file = (session / 'poses.jsonl').open('w', encoding='utf-8')
        except Exception:
            csv_file.close()
            raise
        self.session, self.directories = session, directories
        self.csv_file, self.csv_writer, self.jsonl_file = csv_file, csv_writer, jsonl_file

    def _reset_history(self):
        self.continuous = False
        self.records = []
        self.display_record = None
        self.replay_index = 0
        self.playing = False
        self.history_records = None
        self.frame_index = 0
        self.first_inverse = self.reference = None
        self.rotation_estimator = IncrementalRotationPointEstimator()
        self.captured_images = (None, None, None)
        self.detail = None
        if self.estimator is not None:
            self.estimator.icp.reset()
        self.plot = registration_plot(self.estimator.icp.target if self.estimator else None)

    def _finish_recording(self):
        recording = self.csv_file is not None or self.jsonl_file is not None
        for stream in (self.csv_file, self.jsonl_file):
            if stream is not None:
                stream.close()
        self.csv_file = self.jsonl_file = None
        if recording and self.session is not None:
            save_rotation_angle_csv(self.records, self.session / 'rotation_angles.csv')
            save_angle_plot(self.records, self.session / 'rotation_angle.png')

    def _close(self):
        self.continuous = False
        try:
            if self.camera is not None:
                self.camera.close()
        finally:
            self.camera = None
            self._finish_recording()

    def start(self, stereo_method=None, checkpoint=None, valid_iters=None, inference_scale=None, emitter_enabled=None):
        with self.lock:
            try:
                if self.session is not None:
                    self.stop()
                else:
                    self._close()
                if stereo_method is not None:
                    default_path = default_checkpoint(stereo_method)
                    iterations = valid_iters if valid_iters is not None else (
                        8 if stereo_method == 'fast_foundation_stereo' else 32)
                    if int(iterations) != float(iterations) or int(iterations) < 1:
                        raise ValueError('双目迭代次数必须为 ≥ 1 的整数')
                    scale = float(inference_scale if inference_scale is not None else (
                        .5 if stereo_method == 'foundation_stereo' else 1.))
                    if not 0 < scale <= 1:
                        raise ValueError('双目推理缩放必须在 (0, 1] 内')
                    self.settings.update(stereo_method=stereo_method,
                                         checkpoint=(checkpoint or '').strip() or default_path,
                                         valid_iters=int(iterations), inference_scale=scale)
                if emitter_enabled is not None:
                    self.settings['emitter_enabled'] = bool(emitter_enabled)
                self.config['paths']['stereo_model_path'] = self.settings['checkpoint']
                self.config['stereo']['valid_iters'] = self.settings['valid_iters']
                validate_config(self.config)
                # Validate the target before loading models or opening hardware.
                self.estimator = LivePoseEstimator(self.config, None, self.keep_x_rotation)
                self.camera = RealSenseStereoCamera(**self.settings)
                self.estimator.camera = self.camera
                self._create_session()
                self._reset_history()
                self.preview_images = (None, None, None)
                self.status = (f'相机已启动：{self.settings["stereo_method"]}，'
                               f'{self.settings["valid_iters"]} 次迭代，'
                               f'推理缩放 {self.settings["inference_scale"]}。'
                               f'散斑投射：{"开启" if self.settings["emitter_enabled"] else "关闭"}。'
                               '第一行持续预览；可以单帧或连续拍摄。')
            except Exception as exc:
                self._close()
                self.status = f'启动失败：{exc}'
            return self.render()

    def _capture(self, inputs=None):
        if self.camera is None:
            raise RuntimeError('请先启动相机')
        self.playing = False
        self.history_records = None
        self.replay_index = max(0, len(self.records) - 1)
        index = self.frame_index
        self.frame_index += 1
        if inputs is None:
            inputs = self.estimator.read_inputs()
        self.preview_images = frame_images(inputs, float(self.config['point_cloud']['max_depth_mm']))
        self.captured_images = tuple(image.copy() for image in self.preview_images)
        estimate, inputs = self.estimator.estimate(index, inputs)
        depth = inputs['depth']
        if estimate is None:
            self.detail = dict(frame_index=index, frame_name=f'{index:06d}',
                               status='skipped', error='YOLO 未检测到目标')
            self.jsonl_file.write(json.dumps(self.detail, ensure_ascii=False) + '\n')
            self.jsonl_file.flush()
            self.csv_writer.writerow(self.detail)
            self.csv_file.flush()
            self.status = f'第 {index} 帧未检测到目标，跳过；表格保留上次成功结果。'
            return
        for name in ('color', 'foreground', 'left', 'right'):
            path = self.directories[name] / f'{estimate.frame_name}.png'
            if not cv2.imwrite(str(path), inputs[name]):
                raise OSError(f'图像保存失败：{path}')
            estimate.frame_paths[name] = path
        depth_path = self.directories['depth_mm'] / f'{estimate.frame_name}.npy'
        np.save(depth_path, depth)
        estimate.frame_paths['depth_mm'] = depth_path
        full_depth_path = self.directories['full_depth_mm'] / f'{estimate.frame_name}.npy'
        np.save(full_depth_path, inputs['full_depth'])
        estimate.frame_paths['full_depth_mm'] = full_depth_path
        analysis_pose = estimate.original_to_target
        if self.first_inverse is None:
            self.first_inverse = np.linalg.inv(analysis_pose)
            self.reference = dict(frame_index=index, frame_name=estimate.frame_name)
        relative = self.first_inverse @ analysis_pose
        analysis = self.rotation_estimator.add(relative)
        analysis['removed_x_rotation_deg'] = 0.0
        analysis['rotation_removal'] = 'none'
        if not self.keep_x_rotation:
            analysis.update(gt_axis_rotation_angles(np.linalg.inv(self.first_inverse), analysis_pose))
        record = save_estimate(
            estimate, self.directories, self.config['output']['write_ascii'],
            relative, self.reference, analysis)
        record['timestamp'] = datetime.now().astimezone().isoformat()
        self.jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.jsonl_file.flush()
        row = {key: record[key] for key in self.csv_writer.fieldnames if key in record}
        row.update(fitness=estimate.icp_fitness, rmse_mm=estimate.icp_rmse_mm)
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.records.append(record)
        self.display_record = record
        self.replay_index = len(self.records) - 1
        self.detail = record
        self.plot = registration_plot(self.estimator.icp.target, estimate.registered_cloud)
        angles = (f'主轴倾角 Y={analysis["gt_axis_pitch_deg"]:.3f}°、Z={analysis["gt_axis_yaw_deg"]:.3f}°'
                  if 'gt_axis_pitch_deg' in analysis else f'旋转角 {analysis["rotation_angle_about_center_deg"]:.3f}°')
        self.status = (f'第 {index} 帧：{angles}，'
                       f'fitness={estimate.icp_fitness:.3f}，RMSE={estimate.icp_rmse_mm:.3f} mm，'
                       f'估计耗时 {estimate.elapsed_seconds:.3f} s')

    def capture(self, inputs=None):
        with self.lock:
            try:
                self._capture(inputs)
            except Exception as exc:
                self.continuous = False
                self.status = f'检测暂停：{exc}；表格保留上次成功结果。'
                if self.jsonl_file is not None and not self.jsonl_file.closed:
                    self.detail = dict(frame_index=self.frame_index - 1,
                                       frame_name=f'{self.frame_index - 1:06d}',
                                       status='failed', error=str(exc))
                    self.jsonl_file.write(json.dumps(self.detail, ensure_ascii=False) + '\n')
                    self.jsonl_file.flush()
                    self.csv_writer.writerow(self.detail)
                    self.csv_file.flush()
                self._handle_oom(exc)
            return self.render()

    def _handle_oom(self, exc):
        if 'cuda out of memory' not in str(exc).lower():
            return
        self.playing = False
        self._close()
        if self.estimator is not None:
            self.estimator.camera = None
        self.status = ('显存不足，已停止相机并释放模型，已保存记录保留。'
                       '请将“双目推理缩放”降至 0.5（仍不足可用 0.25），'
                       '或选择 Fast FoundationStereo，再重新启动。')

    def set_continuous(self, enabled):
        with self.lock:
            self.playing = False
            self.continuous = bool(enabled) and self.camera is not None
            self.status = ('连续拍摄中，每帧估计并保存。' if self.continuous else
                           '连续拍摄已暂停，实时预览继续。' if self.camera else '请先启动相机。')
            return self.render()

    def tick(self):
        with self.lock:
            try:
                if self.camera is not None:
                    inputs = self.estimator.read_inputs()
                    self.preview_images = frame_images(
                        inputs, float(self.config['point_cloud']['max_depth_mm']))
                    if self.continuous:
                        return self.capture(inputs)
                if self.playing:
                    elapsed = self.play_record_time + time.monotonic() - self.play_started
                    index = self.replay_index
                    while index + 1 < len(self.playback_records) and self._record_time(index + 1) <= elapsed:
                        index += 1
                    if index != self.replay_index:
                        self._show_record(index)
                    if index == len(self.playback_records) - 1:
                        self.playing = False
                    return self._history_render(include_preview=self.camera is not None)
                # Preview never saves a frame or changes the captured images / pose reference.
                return (*self.preview_images, *(gr.skip() for _ in range(11))) if self.camera else tuple(
                    gr.skip() for _ in range(14))
            except Exception as exc:
                self.continuous = self.playing = False
                self.status = f'实时预览失败：{exc}'
                self._handle_oom(exc)
                return self.render()

    @property
    def playback_records(self):
        return self.records if self.history_records is None else self.history_records

    def _record_time(self, index):
        records = self.playback_records
        try:
            return (datetime.fromisoformat(records[index]['timestamp']) -
                    datetime.fromisoformat(records[0]['timestamp'])).total_seconds()
        except (KeyError, ValueError, TypeError):
            return float(index) * .2

    def _show_record(self, index):
        record = self.playback_records[index]
        self.replay_index = index
        angles = '，'.join(f'{name} {record[field]:.3f}°' for field, name, _ in angle_fields(record))
        self.status = f'角度回放 {index + 1}/{len(self.playback_records)}，原始帧 {record["frame_index"]}：{angles}'

    def seek(self, index):
        with self.lock:
            self.continuous = self.playing = False
            try:
                if not self.playback_records:
                    raise ValueError('尚无成功位姿，请先拍摄或载入历史会话')
                self._show_record(max(0, min(int(index), len(self.playback_records) - 1)))
            except Exception as exc:
                self.status = f'回放失败：{exc}'
            return self._history_render()

    def playback(self, enabled):
        with self.lock:
            self.continuous = self.playing = False
            try:
                if enabled:
                    if not self.playback_records:
                        raise ValueError('尚无成功位姿，请先拍摄或载入历史会话')
                    index = 0 if self.replay_index >= len(self.playback_records) - 1 else self.replay_index
                    self._show_record(index)
                    self.play_record_time = self._record_time(index)
                    self.play_started = time.monotonic()
                    self.playing = len(self.playback_records) > 1
                else:
                    self.status = '回放已暂停。'
            except Exception as exc:
                self.status = f'回放失败：{exc}'
            return self._history_render()

    def load_session(self, value):
        with self.lock:
            try:
                folder = resolve_workspace_path(value).resolve()
                records = [json.loads(line) for line in (folder / 'poses.jsonl').read_text(
                    encoding='utf-8').splitlines() if line.strip()]
                records = [record for record in records if record.get('status') == 'ok']
                if not records:
                    raise ValueError('会话没有成功位姿记录')
                for record in records:
                    angles = np.asarray([record[field] for field, _, _ in angle_fields(record)], dtype=float)
                    if not np.isfinite(angles).all():
                        raise ValueError('历史角度无效')
                    if not isinstance(record['frame_index'], int):
                        raise ValueError('历史帧序号无效')
                self.continuous = self.playing = False
                self.history_records = records
                self._show_record(0)
            except Exception as exc:
                self.status = f'载入失败：{exc}'
            return self._history_render()

    def clear_history(self):
        with self.lock:
            self.continuous = False
            try:
                self._finish_recording()
                self._reset_history()
                self.session = None
                if self.camera is not None:
                    self._create_session()
                self.status = '已清除捕捉图像、位姿与参考，已保存文件保留。' + (
                    '实时预览继续，下次成功拍摄建立新参考。' if self.camera else '请启动相机。')
            except Exception as exc:
                self.status = f'清除失败：{exc}'
            return self.render()

    def stop(self):
        with self.lock:
            self.playing = False
            try:
                self._close()
                self.status = '相机已停止，结果已保存。'
            except Exception as exc:
                self.status = f'停止 / 汇总失败：{exc}'
            return self.render()


def build_ui(controller):
    plotly_asset = Path(plotly.__file__).parent / 'package_data/plotly.min.js'
    gr.set_static_paths(paths=[plotly_asset])
    plot_script = (Path(__file__).parent / 'assets/persistent_pose_plot.js').read_text(encoding='utf-8')

    def persistent_plot(value, label, elem_id):
        return gr.HTML(
            value=value.to_json(), label=label, show_label=True, container=True,
            html_template='<div class="pose-graph" style="width:100%;height:440px"></div>',
            apply_default_css=False,
            head=f'<script src="/gradio_api/file={quote(str(plotly_asset.resolve()))}"></script>',
            js_on_load=plot_script, elem_id=elem_id)

    with gr.Blocks(title='RealSense 实时物体位姿') as ui:
        gr.Markdown('## RealSense D455 实时物体位姿\n'
                    '固定相机，移动目标；双目深度 → YOLO 前景 → 刚性 ICP。'
                    '首次成功估计作为参考，重新启动创建新会话。')
        gr.Markdown(f'GT：`{controller.config["paths"]["icp_target_path"]}`。\n\n'
                    + ('保留完整 XYZ 旋转。' if controller.keep_x_rotation else
                       '六自由度 ICP；仅在配准后的角度计算中消除 GT-X 主轴自转，输出 Y、Z 两个倾角。'
                       '角度参考系由首帧主轴与 GT-X 对齐的最短摆转确定。'))
        with gr.Row():
            stereo_method = gr.Dropdown(
                label='深度估计模型',
                choices=[('FoundationStereo', 'foundation_stereo'),
                         ('Fast FoundationStereo', 'fast_foundation_stereo')],
                value=controller.settings['stereo_method'])
            checkpoint = gr.Textbox(label='模型权重路径', value=controller.settings['checkpoint'])
            valid_iters = gr.Number(label='双目迭代次数', value=controller.settings['valid_iters'],
                                    precision=0, minimum=1)
            inference_scale = gr.Dropdown(
                label='双目推理缩放', choices=[1., .75, .5, .25],
                value=controller.settings['inference_scale'])
            emitter = gr.Checkbox(label='开启 RealSense 散斑投射',
                                  value=controller.settings['emitter_enabled'])
        gr.Markdown('切换模型自动填入默认权重及迭代次数（Foundation：32；Fast Foundation：8）。'
                    '可自行修改，点击“启动 / 重新启动相机”生效，并创建新会话。')
        gr.Markdown('8GB 显卡使用 Foundation 时默认以 0.5 倍尺寸推理；Fast 默认 1.0。'
                    '缩小推理尺寸能降低显存占用，但会减少深度细节；'
                    '采集、RGB 标定和保存尺寸仍为 1280×720，深度按正确尺度恢复后对齐 RGB。')
        with gr.Row():
            start = gr.Button('启动 / 重新启动相机', variant='primary')
            capture = gr.Button('拍摄一帧并估计')
            continuous = gr.Checkbox(label='连续拍摄并估计', value=False)
            stop = gr.Button('停止并保存汇总')
        status = gr.Textbox(label='状态', value=controller.status, interactive=False, lines=3, max_lines=3)
        gr.Markdown('### 实时预览\n启动后持续刷新，暂停连续拍摄时仍更新；预览不保存、不建立参考。')
        with gr.Row():
            preview_rgb = gr.Image(label='实时 RGB', type='numpy', interactive=False, height=280)
            preview_depth = gr.Image(label='实时深度图', type='numpy', interactive=False, height=280)
            preview_segmented = gr.Image(label='实时分割后的深度图', type='numpy', interactive=False, height=280)
        gr.Markdown('### 最新捕捉\n仅在单帧或连续拍摄时更新，同一行三张图来自同一帧。')
        with gr.Row():
            image = gr.Image(label='最新捕捉 RGB', type='numpy', interactive=False, height=280)
            depth = gr.Image(label='最新捕捉深度图', type='numpy', interactive=False, height=280)
            segmented = gr.Image(label='最新捕捉分割后的深度图', type='numpy', interactive=False, height=280)
        plot = persistent_plot(controller.plot, 'GT 与当前帧配准点云', 'realsense-registration')
        table = gr.Dataframe(headers=['ΔX / mm', 'ΔY / mm', 'ΔZ / mm',
                                      '倾角 Y / °', '倾角 Z / °',
                                      '主轴偏转角 / °', 'ICP fitness', 'ICP RMSE / mm'],
                             interactive=False)
        gr.Markdown('平移来自原始六自由度相对位姿。默认倾角描述主轴相对首帧的方向，采用 u = Rz · Ry · X；'
                    '--keep-x-rotation 模式显示原始相对 pitch、yaw 和总体旋转角。'
                    '3D 图展示 GT 坐标系中的配准结果，平移和点云单位均为 mm。')
        with gr.Accordion('本帧结果与矩阵', open=False):
            detail = gr.JSON()
        gr.Markdown('### 历史角度回放\n可勾选 Y、Z、总体曲线。默认模式的“总体”为去自转后的主轴偏转角；'
                    '旧会话或保留完整旋转模式沿用原总体旋转角。虚线标记回放帧。'
                    '拖动滑块或自动播放只更新角度图，最新捕捉图像与点云保持不变。')
        history_curves = gr.CheckboxGroup(label='显示曲线', choices=list(HISTORY_CURVES),
                                          value=controller.history_curves)
        angles = persistent_plot(history_plot([], 0), '历史旋转角 / °', 'realsense-angle-history')
        with gr.Row():
            slider = gr.Slider(label='角度回放序号（仅成功帧，从 0 开始）', minimum=0,
                               maximum=1, value=0, step=1)
            playing = gr.Checkbox(label='自动回放角度（按拍摄时间）', value=False)
        with gr.Row():
            folder = gr.Textbox(label='历史会话目录', placeholder='depth_outputs/realsense_icp_pose/pose_...')
            load = gr.Button('载入历史角度')
        clear = gr.Button('清除历史与捕捉结果')
        events = dict(outputs=[preview_rgb, preview_depth, preview_segmented,
                               image, depth, segmented, plot, table, detail, status, continuous,
                               slider, playing, angles],
                      concurrency_id='realsense-pose', concurrency_limit=1,
                      scroll_to_output=False, show_progress='hidden')
        stereo_method.change(
            lambda method: (default_checkpoint(method), 8 if method == 'fast_foundation_stereo' else 32,
                            .5 if method == 'foundation_stereo' else 1.),
            inputs=[stereo_method], outputs=[checkpoint, valid_iters, inference_scale],
            concurrency_id='realsense-pose', concurrency_limit=1,
            scroll_to_output=False, show_progress='hidden')
        start.click(controller.start, inputs=[stereo_method, checkpoint, valid_iters, inference_scale, emitter], **events)
        capture.click(controller.capture, **events)
        continuous.input(controller.set_continuous, inputs=[continuous], **events)
        stop.click(controller.stop, **events)
        clear.click(controller.clear_history, **events)
        slider.release(controller.seek, inputs=[slider], **events)
        playing.input(controller.playback, inputs=[playing], **events)
        history_curves.input(controller.set_history_curves, inputs=[history_curves], outputs=[angles],
                             concurrency_id='realsense-pose', concurrency_limit=1,
                             scroll_to_output=False, show_progress='hidden')
        load.click(controller.load_session, inputs=[folder], **events)
        gr.Timer(0.2).tick(controller.tick, **events)
    return ui


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/pose/main.yaml')
    parser.add_argument('--gt-ply', default=DEFAULT_TARGET)
    parser.add_argument('--output', default='depth_outputs/realsense_icp_pose')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--stereo-method', choices=STEREO_METHODS, default='fast_foundation_stereo')
    parser.add_argument('--checkpoint')
    parser.add_argument('--valid-iters', type=int)
    parser.add_argument('--emitter', action=argparse.BooleanOptionalAction, default=False,
                        help='开启散斑投射；--no-emitter 关闭（默认）')
    parser.add_argument('--inference-scale', type=float, choices=[1., .75, .5, .25],
                        help='双目推理缩放；默认 Foundation=0.5、Fast=1.0')
    parser.add_argument('--yolo-weights')
    parser.add_argument('--keep-x-rotation', action='store_true', help='保留完整旋转，不去除 GT-X 主轴自转')
    parser.add_argument('--port', type=int, default=7862)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device_index < 0 or (args.valid_iters is not None and args.valid_iters < 1):
        raise ValueError('设备序号须 ≥ 0，迭代次数须 ≥ 1')
    config = load_config(resolve_workspace_path(args.config))
    config['paths']['icp_target_path'] = args.gt_ply
    config['paths']['stereo_model_path'] = args.checkpoint or default_checkpoint(args.stereo_method)
    config['stereo']['valid_iters'] = args.valid_iters if args.valid_iters is not None else (
        8 if args.stereo_method == 'fast_foundation_stereo' else 32)
    if args.yolo_weights:
        config['segmentation']['yolo_model_path'] = args.yolo_weights
    controller = PoseController(
        config, args.output, args.device_index, args.stereo_method, args.checkpoint,
        args.valid_iters, args.keep_x_rotation, args.inference_scale, args.emitter)
    try:
        build_ui(controller).queue().launch(server_name='127.0.0.1', server_port=args.port)
    finally:
        controller.stop()


if __name__ == '__main__':
    main()
