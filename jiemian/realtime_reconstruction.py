"""Live Shengxiang / RealSense D455 capture and one-frame-at-a-time reconstruction UI."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import sys
import threading
from time import perf_counter
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import open3d as o3d
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Recon.timing import StageTimer, timing_summary
from Recon.Reconstruction import Reconstruction
from Recon.config_loader import load_config
from jiemian.cameras.realsense_stereo_camera import STEREO_METHODS, default_checkpoint
from jiemian.processing.foreground import DEFAULT_YOLO


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _camera_storage_root(value, camera_type):
    """Keep existing camera directory layouts; scope custom roots by camera."""
    camera_folders = {'shengxiang': 'shengxiang', 'realsense_d455': 'realsense'}
    if camera_type not in camera_folders:
        raise ValueError('不支持的相机类型')
    folder = camera_folders[camera_type]
    root = _project_path(value)
    parts = list(root.parts)
    aliases = {'shengxiang', 'shenxiang', 'realsense', 'realsense_d455'}
    # Replace the nearest camera directory, preserving suffixes such as continuous.
    for index in range(len(parts) - 1, 0, -1):
        if parts[index] in aliases:
            parts[index] = folder
            return Path(*parts)
    return root / folder


def _camera_settings(device_index, binning, working_mode, exposure_mode,
                     exposure_3d, exposure_gray, user_gain, exposure_number,
                     auto_hdr_priority, auto_phdr_quality):
    settings = {
        'device_index': int(device_index),
        'binning': bool(binning),
        'working_mode': str(working_mode),
        'exposure_mode': str(exposure_mode),
        'exposure_3d': float(exposure_3d),
        'exposure_gray': float(exposure_gray),
        'user_gain': int(user_gain),
        'exposure_number': int(exposure_number),
        'auto_hdr_priority': int(auto_hdr_priority),
        'auto_phdr_quality': int(auto_phdr_quality),
    }
    if settings['device_index'] < 0:
        raise ValueError('设备序号不能为负数')
    if settings['working_mode'] not in {'fast_3d', 'super_precise_3d'}:
        raise ValueError('工作模式必须是 fast_3d 或 super_precise_3d')
    if settings['exposure_mode'] not in {'manual', 'manual_repeat', 'auto_nhdr', 'auto_phdr'}:
        raise ValueError('无效的曝光模式')
    for key in ('exposure_3d', 'exposure_gray'):
        if not 0.1 <= settings[key] <= 100:
            raise ValueError(f'{key} 须在 0.1–100 之间')
    if not 0 <= settings['user_gain'] <= 5:
        raise ValueError('用户增益须在 0–5 之间')
    if not 1 <= settings['exposure_number'] <= 3:
        raise ValueError('曝光次数须在 1–3 之间')
    if not 0 <= settings['auto_hdr_priority'] <= 7:
        raise ValueError('自动 HDR 优先级须在 0–7 之间')
    if not 0 <= settings['auto_phdr_quality'] <= 100:
        raise ValueError('自动 PHDR 质量须在 0–100 之间')
    return settings


def _software_settings(neighbors, std_ratio, joint_frames, voxel_size,
                       final_voxel_size, correspondence_distance, max_iteration,
                       loop_stride, loop_preference, edge_prune, reference_node):
    settings = {
        'statistical_nb_neighbors': int(neighbors),
        'statistical_std_ratio': float(std_ratio),
        'aruco_min_joint_calibration_frames': int(joint_frames),
        'pose_graph': {
            'voxel_size_mm': float(voxel_size),
            'final_voxel_size_mm': float(final_voxel_size),
            'max_correspondence_distance_mm': float(correspondence_distance),
            'max_iteration': int(max_iteration),
            'loop_closure_stride': int(loop_stride),
            'preference_loop_closure': float(loop_preference),
            'edge_prune_threshold': float(edge_prune),
            'reference_node': int(reference_node),
        },
    }
    graph = settings['pose_graph']
    if settings['statistical_nb_neighbors'] < 0:
        raise ValueError('统计滤波邻居数须 ≥ 0；0 表示关闭滤波')
    if not np.isfinite(settings['statistical_std_ratio']) or settings['statistical_std_ratio'] <= 0:
        raise ValueError('统计滤波标准差倍数须 > 0')
    if settings['aruco_min_joint_calibration_frames'] < 1:
        raise ValueError('共同观测最少帧数须 ≥ 1')
    if not np.isfinite(graph['voxel_size_mm']) or graph['voxel_size_mm'] <= 0:
        raise ValueError('图优化体素大小须 > 0')
    if not np.isfinite(graph['final_voxel_size_mm']) or graph['final_voxel_size_mm'] < 0:
        raise ValueError('融合体素大小须 ≥ 0')
    if not np.isfinite(graph['max_correspondence_distance_mm']) or graph['max_correspondence_distance_mm'] <= 0:
        raise ValueError('最大对应距离须 > 0')
    if graph['max_iteration'] < 1 or graph['loop_closure_stride'] < 0:
        raise ValueError('ICP 迭代次数须 ≥ 1；回环连接间隔须 ≥ 0')
    if not np.isfinite(graph['preference_loop_closure']) or graph['preference_loop_closure'] < 0:
        raise ValueError('回环连接权重须 ≥ 0')
    if not np.isfinite(graph['edge_prune_threshold']) or not 0 <= graph['edge_prune_threshold'] <= 1:
        raise ValueError('连接剪枝阈值须在 0–1 之间')
    if graph['reference_node'] < 0:
        raise ValueError('固定节点序号须 ≥ 0')
    return settings


def _camera_worker(frame_queue, status_pipe, stop_event, settings):
    """Keep the vendor SDK in a child process; its USB teardown can crash."""
    try:
        from mpsizector import Camera, DataOutMode, ExposureMode, WorkingMode

        from jiemian.processing.foreground import ForegroundSegmenter
        foreground_settings = settings['foreground']
        segmenter = ForegroundSegmenter(**{
            key: foreground_settings[key] for key in ('weights', 'conf', 'imgsz', 'device')})
        camera = Camera()
        camera.open(settings['device_index'])
        camera.set_binning(settings['binning'])
        camera.set_working_mode(getattr(WorkingMode, settings['working_mode']))
        camera.use_camera_coordinate_system()
        camera.set_zmap_scale(z_zero_mm=0.0, z_increment_mm=10_000.0 / 65_534.0)
        camera.set_data_out_mode(DataOutMode.fix_z_map)
        camera.set_exposure_3d(settings['exposure_3d'])
        camera.set_exposure_2d(settings['exposure_gray'])
        camera.set_user_gain(settings['user_gain'])
        camera.set_exposure_basic(
            getattr(ExposureMode, settings['exposure_mode']),
            exposure_number=settings['exposure_number'],
            auto_hdr_priority=settings['auto_hdr_priority'],
            auto_phdr_quality=settings['auto_phdr_quality'],
        )
        camera.start_stream()
        status_pipe.send(('ready', camera.current_device()))
        sequence = 0
        while not stop_event.is_set():
            timer = StageTimer()
            frame = camera.get_latest_zmap_frame(timeout_ms=1000)
            if frame is None:
                continue
            timer.mark('wait_for_frames')
            depth, gray, sdk_mask = frame[:3]
            foreground = segmenter.segment(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            timer.mark('foreground_segment')
            # Retain SDK invalid flags, and mark background as invalid for reconstruction.
            mask = np.asarray(sdk_mask, dtype=np.uint8).copy()
            mask[~foreground] |= 1
            aligned = np.where(((sdk_mask & 0x07) == 0) & np.isfinite(depth) & (depth > 0), depth, 0)
            extra = dict(foreground=foreground.astype(np.uint8) * 255, aligned_depth_mm=aligned)
            timer.mark('mask_pack')
            extra['timings_ms'] = timer.finish()
            extra['produced_at'] = perf_counter()
            sequence += 1
            print(f'[Shengxiang frame {sequence}] {timing_summary(extra["timings_ms"])}', flush=True)
            item = (sequence, depth, gray, mask, None, extra)
            try:
                frame_queue.put_nowait(item)
            except Full:
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
                try:
                    frame_queue.put_nowait(item)
                except Full:
                    pass
    except Exception as exc:
        try:
            status_pipe.send(('error', str(exc)))
        except (BrokenPipeError, OSError):
            pass
    finally:
        status_pipe.close()
        # The vendor SDK may crash while destructing its background USB thread.
        # Exiting this child releases the device without taking down Gradio.
        os._exit(0)


def _realsense_worker(frame_queue, status_pipe, stop_event, settings):
    from jiemian.cameras.realsense_stereo_camera import RealSenseStereoCamera
    camera = None
    try:
        camera = RealSenseStereoCamera(settings['device_index'], settings['checkpoint'], settings['valid_iters'], settings.get('foreground'), settings['stereo_method'])
        status_pipe.send(('ready', camera.detail))
        sequence = 0
        while not stop_event.is_set():
            depth, left, mask, right, extra = camera.capture()
            sequence += 1
            print(f'[RealSense frame {sequence}] {timing_summary(extra["timings_ms"])}', flush=True)
            print(f'[{settings["stereo_method"]} frame {sequence}] {timing_summary(extra["stereo_timings_ms"])}', flush=True)
            item = (sequence, depth, left, mask, right, extra)
            try:
                frame_queue.put_nowait(item)
            except Full:
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
                try:
                    frame_queue.put_nowait(item)
                except Full:
                    pass
    except Exception as exc:
        try:
            status_pipe.send(('error', str(exc)))
        except (BrokenPipeError, OSError):
            pass
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            status_pipe.close()


def _prepare_frame(raw_frame, image_size, min_depth, max_depth):
    sequence, depth_mm, gray, sdk_mask = raw_frame[:4]
    if depth_mm.shape != gray.shape or gray.shape != sdk_mask.shape:
        raise ValueError('相机的深度、灰度和 mask 尺寸不一致')
    valid = ((sdk_mask & 0x07) == 0) & np.isfinite(depth_mm)
    valid &= (depth_mm > 0) & (depth_mm >= min_depth) & (depth_mm <= min(max_depth, 65535))
    depth = np.zeros(depth_mm.shape, dtype=np.uint16)
    depth[valid] = np.rint(depth_mm[valid]).astype(np.uint16)
    mask = valid.astype(np.uint8) * 255
    width, height = image_size
    if (gray.shape[1], gray.shape[0]) != image_size:
        camera_ratio = gray.shape[1] / gray.shape[0]
        calibration_ratio = width / height
        if abs(camera_ratio - calibration_ratio) > 0.02:
            raise ValueError('相机图像比例与标定文件不一致，不能安全缩放')
        gray = cv2.resize(gray, image_size, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, image_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    return sequence, gray, depth, mask


def _raw_depth_stats(depth_mm, sdk_mask, min_depth, max_depth):
    finite_positive = np.isfinite(depth_mm) & (depth_mm > 0)
    sdk_valid = (sdk_mask & 0x07) == 0
    accepted = finite_positive & sdk_valid & (depth_mm >= min_depth) & (depth_mm <= max_depth)
    raw_values = depth_mm[finite_positive]
    sdk_values = depth_mm[finite_positive & sdk_valid]
    return {
        'raw_min_mm': float(raw_values.min()) if raw_values.size else None,
        'sdk_valid_min_mm': float(sdk_values.min()) if sdk_values.size else None,
        'raw_below_min_count': int(np.count_nonzero(finite_positive & (depth_mm < min_depth))),
        'sdk_rejected_positive_count': int(np.count_nonzero(finite_positive & ~sdk_valid)),
        'accepted_count': int(np.count_nonzero(accepted)),
        'pixel_count': int(depth_mm.size),
    }


def _unsegmented_depth(raw_frame, image_size):
    """Use sensor-valid depth before foreground masking, in calibrated pixels."""
    extra = raw_frame[5] if len(raw_frame) > 5 else {}
    if 'aligned_depth_mm' in extra:
        # Both camera workers retain valid, unsegmented depth here. Their
        # outgoing SDK mask already includes foreground rejection flags.
        depth = np.asarray(extra['aligned_depth_mm'], dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0)
    else:
        depth = np.asarray(raw_frame[1], dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0) & ((raw_frame[3] & 0x07) == 0)
    depth = np.where(valid, depth, 0)
    if (depth.shape[1], depth.shape[0]) != image_size:
        depth = cv2.resize(depth, image_size, interpolation=cv2.INTER_NEAREST)
    return depth


def _aruco_summary(frame, pose, estimator):
    corners, detected_ids, _ = estimator.detector.detectMarkers(frame)
    marker_ids = set() if detected_ids is None else {int(value) for value in detected_ids.ravel()}
    board_poses = pose.get('board_poses') if pose is not None else None
    if board_poses is None:
        board_poses = estimator.estimate_board_poses(corners, detected_ids)
    registered = {int(board_id) for board_id in estimator.marker_to_anchor_transforms}
    boards = []
    for board_id, layout in sorted(estimator.board_marker_layouts.items()):
        visible_ids = sorted(marker_ids.intersection(int(marker_id) for marker_id in layout))
        if visible_ids:
            boards.append({
                '板子 ID': int(board_id),
                '检测到的码数': len(visible_ids),
                '码 ID': visible_ids,
                '已注册到全局坐标系': int(board_id) in registered,
                '本帧位姿解算成功': board_id in board_poses,
                '共同观测所需码数': estimator.minimum_markers_for_joint_calibration(board_id),
                '本帧码数满足板间标定': len(visible_ids) >= estimator.minimum_markers_for_joint_calibration(board_id),
                '已累计共同观测帧数': max((len(frames) for (reference, candidate), frames
                    in estimator.board_calibration_frames.items() if candidate == board_id), default=0),
                '共同观测所需帧数': estimator.MIN_JOINT_CALIBRATION_FRAMES,
            })
    if pose is not None:
        status = '已得到世界位姿'
    elif not marker_ids:
        status = '未检测到 ArUco 码'
    elif boards and not any(board['已注册到全局坐标系'] for board in boards):
        status = '检测到板子，但尚未完成板间标定，无法得到世界位姿'
    else:
        status = '检测到 ArUco 码，但未能解算世界位姿'
    summary = {
        '位姿状态': status,
        '检测到的码总数': len(marker_ids),
        '检测到的码 ID': sorted(marker_ids),
        '未配置的码 ID': sorted(marker_ids - {int(marker_id) for layout in estimator.board_marker_layouts.values() for marker_id in layout}),
        '重复码 ID': [] if detected_ids is None else [int(marker_id) for marker_id, count in zip(*np.unique(detected_ids, return_counts=True)) if count > 1],
        '注册板子 ID': sorted(registered),
        '检测到的板子数': len(boards),
        '板子': boards,
        '锚点板子 ID': None if estimator.anchor_marker_id is None else int(estimator.anchor_marker_id),
        '本帧参考板子 ID': None if pose is None else int(pose['reference_board_id']),
    }
    if pose is not None:
        summary['相机到世界平移_mm'] = np.asarray(pose['translation_cam_to_world']).reshape(3).tolist()
        summary['相机到世界旋转矩阵'] = np.asarray(pose['rotation_cam_to_world']).tolist()
    return summary


def _depth_color_range(depth):
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if not valid.size:
        return None
    low, high = np.percentile(valid, (2, 98))
    return low, max(high, low + 1)


def _depth_preview(depth, color_range):
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if color_range is not None:
        low, high = color_range
        scaled = np.clip((depth.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    color[depth == 0] = 0
    return color


class LiveReconstruction:
    def __init__(self, config_path, device_index=0, exposure_3d=100.0,
                 camera_type='shengxiang', checkpoint=None, valid_iters=None, foreground_settings=None,
                 stereo_method='foundation_stereo'):
        self.config_path = _project_path(config_path)
        self.config = load_config(self.config_path)
        if self.config['reconstruction']['input_mode'] != 'direct_depth':
            raise ValueError('实时界面要求 reconstruction.input_mode=direct_depth')
        if not self.config['reconstruction']['use_aruco']:
            raise ValueError('实时界面要求 reconstruction.use_aruco=true')
        self.camera_settings = _camera_settings(
            device_index, True, 'super_precise_3d', 'manual',
            exposure_3d, 30.0, 3, 1, 3, 90,
        )
        if camera_type not in {'shengxiang', 'realsense_d455'}:
            raise ValueError('不支持的相机类型')
        checkpoint = checkpoint or default_checkpoint(stereo_method)
        valid_iters = valid_iters if valid_iters is not None else (8 if stereo_method == 'fast_foundation_stereo' else 32)
        self.camera_settings.update(camera_type=camera_type, checkpoint=str(checkpoint),
                                    valid_iters=int(valid_iters), stereo_method=stereo_method,
                                    aruco_board_yaml_path=self.config['paths']['aruco_board_yaml_path'])
        seg = self.config['segmentation']
        self.camera_settings['foreground'] = foreground_settings or dict(
            enabled=True,
            weights=seg['yolo_model_path'] or DEFAULT_YOLO, conf=seg['yolo_conf'],
            imgsz=seg['yolo_imgsz'], device=seg['yolo_device'])
        self.camera_settings['foreground']['enabled'] = True
        recon_config = self.config['reconstruction']
        graph_config = self.config['global_registration']['pose_graph']
        self.software_settings = _software_settings(
            recon_config['statistical_nb_neighbors'],
            recon_config['statistical_std_ratio'],
            recon_config['aruco_min_joint_calibration_frames'],
            graph_config['voxel_size_mm'],
            graph_config['final_voxel_size_mm'],
            graph_config['max_correspondence_distance_mm'],
            graph_config['max_iteration'],
            graph_config['loop_closure_stride'],
            graph_config['preference_loop_closure'],
            graph_config['edge_prune_threshold'],
            graph_config['reference_node'],
        )
        self.camera_matrix = None
        self.image_size = None
        if camera_type == 'shengxiang':
            self._load_shengxiang_calibration()
        self.lock = threading.RLock()
        self.context = mp.get_context('spawn')
        self.process = None
        self.frame_queue = None
        self.status_pipe = None
        self.stop_event = None
        self.latest = None
        self.latest_raw = None
        self.latest_stats = None
        self.last_captured_sequence = -1
        self.reconstruction = None
        self.raw_dir = None
        self.output_dir = None
        self.captured_count = 0
        self.continuous_capture = False
        self.capture_interval = 0.0
        self.next_capture_at = 0.0
        self.global_display_path = None
        self.full_global_cloud = o3d.geometry.PointCloud()
        self.full_global_display_path = None
        self.pose_summaries = {}
        self.message = '未启动'

    def _load_shengxiang_calibration(self):
        intrinsics_path = _project_path(self.config['paths']['int_yaml_path'])
        storage = cv2.FileStorage(str(intrinsics_path), cv2.FILE_STORAGE_READ)
        try:
            matrix = storage.getNode('M1').mat()
            width = int(storage.getNode('image_width').real())
            height = int(storage.getNode('image_height').real())
        finally:
            storage.release()
        if matrix is None or np.asarray(matrix).shape != (3, 3) or width <= 0 or height <= 0:
            raise ValueError(f'标定文件缺少 M1/image_width/image_height: {intrinsics_path}')
        self.camera_matrix = np.asarray(matrix, dtype=np.float64)
        self.image_size = (width, height)

    def _drain_frames(self):
        if self.frame_queue is None:
            return
        timer = StageTimer()
        raw_frame = None
        # Bound queue draining so a fast producer cannot keep this callback busy.
        # The camera queue holds at most two frames; prepare only the latest one.
        for _ in range(2):
            try:
                raw_frame = self.frame_queue.get_nowait()
            except Empty:
                break
        if raw_frame is None:
            return
        timer.mark('queue_get_deserialize')
        self.latest_raw = raw_frame
        extra = raw_frame[5] if len(raw_frame) > 5 else {}
        if 'produced_at' in extra:
            extra['delivery_age_ms'] = (perf_counter() - extra['produced_at']) * 1000.
        self.latest_stats = _raw_depth_stats(
            raw_frame[1], raw_frame[3],
            float(self.config['reconstruction']['depth_range_mm']['min']),
            float(self.config['reconstruction']['depth_range_mm']['max']),
        )
        timer.mark('depth_stats')
        self.latest = _prepare_frame(
            raw_frame,
            self.image_size,
            float(self.config['reconstruction']['depth_range_mm']['min']),
            float(self.config['reconstruction']['depth_range_mm']['max']),
        )

        timer.mark('prepare_frame')
        extra['receive_timings_ms'] = timer.finish()
        if 'timings_ms' in extra:
            print(f'[Receive frame {raw_frame[0]}] {timing_summary(extra["receive_timings_ms"])}; '
                  f'delivery_age={extra["delivery_age_ms"]:.1f} ms', flush=True)

    def _status(self):
        diagnostics = ''
        if self.latest_stats is not None:
            stats = self.latest_stats
            raw_min = '无' if stats['raw_min_mm'] is None else f'{stats["raw_min_mm"]:.0f} mm'
            sdk_min = '无' if stats['sdk_valid_min_mm'] is None else f'{stats["sdk_valid_min_mm"]:.0f} mm'
            diagnostics = (
                f'｜原始最近深度：{raw_min}；有效最近深度：{sdk_min}'
                f'；低于软件下限：{stats["raw_below_min_count"]} 像素'
            )
        if self.latest_raw is not None and len(self.latest_raw) > 5:
            timings = self.latest_raw[5].get('timings_ms', {})
            if timings:
                total = timings['total_ms']
                slowest = max((k for k in timings if k != 'total_ms'), key=timings.get)
                diagnostics += (f'｜采集处理：{total:.0f} ms（{1000. / total:.2f} FPS）'
                                f'；最慢：{slowest} {timings[slowest]:.0f} ms')
        mode = '｜连续拍摄中' if self.continuous_capture else ''
        return f'{self.message}{mode}｜已拍摄 {self.captured_count} 帧{diagnostics}｜输出：{self.output_dir or "未创建"}'

    def start(self, settings=None, software_settings=None):
        try:
            return self._start_impl(settings, software_settings)
        except Exception:
            self.stop()
            raise

    def _launch_camera(self, settings):
        self.frame_queue = self.context.Queue(maxsize=2)
        parent_pipe, child_pipe = self.context.Pipe(duplex=False)
        self.status_pipe = parent_pipe
        self.stop_event = self.context.Event()
        self.process = self.context.Process(
            target=(_realsense_worker if settings.get('camera_type') == 'realsense_d455' else _camera_worker),
            args=(self.frame_queue, child_pipe, self.stop_event, settings),
            daemon=True,
        )
        self.process.start()
        child_pipe.close()
        if not parent_pipe.poll(180):
            self._stop_camera()
            raise RuntimeError('相机启动超时，请检查设备连接和 SDK')
        try:
            kind, detail = parent_pipe.recv()
        except EOFError as exc:
            self._stop_camera()
            raise RuntimeError('相机进程意外退出') from exc
        if kind != 'ready':
            self._stop_camera()
            raise RuntimeError(f'相机启动失败：{detail}')
        self.latest = self.latest_raw = self.latest_stats = None
        return detail

    def _stop_camera(self):
        self.continuous_capture = False
        if self.stop_event is not None:
            self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        if self.status_pipe is not None:
            self.status_pipe.close()
        if self.frame_queue is not None:
            self.frame_queue.close()
        self.process = self.frame_queue = self.status_pipe = self.stop_event = None
        self.latest = self.latest_raw = self.latest_stats = None

    def _start_impl(self, settings=None, software_settings=None):
        with self.lock:
            if self.process is not None and self.process.is_alive():
                self.message = '相机正在运行；请先停止，再点击开始使新参数生效'
                return self._status()
            self.stop()
            self.camera_settings = {**self.camera_settings, **(settings or {})}
            self.software_settings = software_settings or self.software_settings
            self.camera_settings['foreground']['enabled'] = True
            default_checkpoint(self.camera_settings['stereo_method'])
            detail = self._launch_camera(self.camera_settings)

            if self.camera_settings['camera_type'] == 'realsense_d455':
                calibration = detail['calibration']
                self.camera_matrix = np.asarray(calibration['camera_matrix'], dtype=np.float64)
                self.image_size = tuple(calibration['image_size'])
            else:
                self._load_shengxiang_calibration()
            stamp = datetime.now().strftime('realtime_%Y%m%d_%H%M%S_%f')
            camera_type = self.camera_settings['camera_type']
            self.raw_dir = _camera_storage_root(self.config['paths']['data_root_path'], camera_type) / stamp
            self.output_dir = _camera_storage_root(self.config['paths']['save_dir'], camera_type) / stamp
            for name in ('gray', 'depth', 'depth_full', 'mask', 'sdk_depth_mm', 'sdk_mask', 'right', 'rgb', 'foreground_mask', 'aligned_depth_mm', 'stereo_depth_mm', 'left_ir'):
                (self.raw_dir / name).mkdir(parents=True, exist_ok=True)
            calibration_path = _project_path(self.config['paths']['int_yaml_path']) if self.camera_settings['camera_type'] == 'shengxiang' else self.raw_dir / 'intrinsics.yaml'
            if self.camera_settings['camera_type'] == 'realsense_d455':
                with calibration_path.open('w', encoding='utf-8') as calibration_file:
                    yaml.safe_dump(dict(rgb_intrinsics=dict(
                        fx=float(self.camera_matrix[0, 0]), fy=float(self.camera_matrix[1, 1]),
                        cx=float(self.camera_matrix[0, 2]), cy=float(self.camera_matrix[1, 2]),
                        width=self.image_size[0], height=self.image_size[1]),
                        distortion_coefficients=[0.] * 5), calibration_file)
            recon = self.config['reconstruction']
            seg = self.config['segmentation']
            reconstruction = Reconstruction(
                data_root_path=str(self.raw_dir),
                color_ext_yaml_path=str(calibration_path),
                aruco_board_yaml_path=str(_project_path(self.camera_settings['aruco_board_yaml_path'])),
                aruco_length=recon['aruco_length'],
                world_voxel_size_mm=recon['world_voxel_size_mm'],
                method=recon['stereo_method'],
                use_aruco=True,
                brightness_mask_enabled=recon['brightness_mask']['enabled'],
                brightness_threshold=recon['brightness_mask']['threshold'],
                use_sem=False,
                segmentation_method=seg['method'],
                sam_model_path=seg['sam_model_path'],
                yolo_model_path=seg['yolo_model_path'],
                yolo_conf=seg['yolo_conf'],
                yolo_imgsz=seg['yolo_imgsz'],
                yolo_device=seg['yolo_device'],
                sem_statistical_nb_neighbors=seg['statistical_nb_neighbors'],
                sem_statistical_std_ratio=seg['statistical_std_ratio'],
                statistical_nb_neighbors=self.software_settings['statistical_nb_neighbors'],
                statistical_std_ratio=self.software_settings['statistical_std_ratio'],
                input_mode='direct_depth',
            )
            reconstruction.aruco_estimater.MIN_JOINT_CALIBRATION_FRAMES = (
                self.software_settings['aruco_min_joint_calibration_frames']
            )
            reconstruction.prepare_outputs(str(self.output_dir))
            self.reconstruction = reconstruction
            with (self.output_dir / 'session.json').open('w', encoding='utf-8') as session_file:
                json.dump({
                    'config': str(self.config_path),
                    'device': detail,
                    'aruco_board_yaml_path': str(_project_path(self.camera_settings['aruco_board_yaml_path'])),
                    'camera_matrix': self.camera_matrix.tolist(),
                    'image_size': list(self.image_size),
                    'camera_settings': self.camera_settings,
                    'software_settings': self.software_settings,
                    'raw_dir': str(self.raw_dir),
                    'output_dir': str(self.output_dir),
                }, session_file, ensure_ascii=False, indent=2, default=str)
            self.captured_count = 0
            self.last_captured_sequence = -1
            self.global_display_path = None
            self.full_global_cloud = o3d.geometry.PointCloud()
            self.full_global_display_path = None
            self.pose_summaries = {}
            self.latest = self.latest_raw = self.latest_stats = None
            self._last_preview_raw = None
            self.message = f'相机运行中：{detail.get("name", "设备已连接")}'
            return self._status()

    def poll(self):
        timer = StageTimer()
        with self.lock:
            timer.mark('lock_wait')
            if self.process is None:
                return None, None, None, self._status()
            if self.status_pipe.poll():
                try:
                    kind, detail = self.status_pipe.recv()
                except EOFError:
                    kind, detail = 'error', '相机进程已退出'
                if kind == 'error':
                    self.message = f'相机采集失败：{detail}'
            if not self.process.is_alive():
                if '采集失败' not in self.message:
                    self.message = '相机进程已退出，请重新开始'
                return None, None, None, self._status()
            self._drain_frames()
            timer.mark('receive_frame')
            if self.latest is None:
                return None, None, None, self._status()
            _, gray, depth, mask = self.latest
            raw_frame = self.latest_raw
            extra = raw_frame[5] if len(raw_frame) > 5 else {}
            depth_range = self.config['reconstruction']['depth_range_mm'].copy()
            status = self._status()
            log_preview = 'timings_ms' in extra and getattr(self, '_last_preview_raw', None) is not raw_frame
            if log_preview:
                self._last_preview_raw = raw_frame

        # Frame arrays are read-only here. Render outside the capture lock and
        # resize first; full-resolution arrays remain available to reconstruction.
        height, width = depth.shape
        scale = min(1., 640. / max(height, width))
        preview_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = extra.get('rgb', gray)
        if (image.shape[1], image.shape[0]) != preview_size:
            image = cv2.resize(image, preview_size, interpolation=cv2.INTER_AREA)
        preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if 'rgb' in extra else image
        original_depth = np.asarray(extra.get('aligned_depth_mm', raw_frame[1]))
        if (original_depth.shape[1], original_depth.shape[0]) != preview_size:
            original_depth = cv2.resize(original_depth, preview_size, interpolation=cv2.INTER_NEAREST)
        if (width, height) != preview_size:
            depth = cv2.resize(depth, preview_size, interpolation=cv2.INTER_NEAREST)
        timer.mark('resize_preview')
        original_depth = np.where(
            np.isfinite(original_depth) & (original_depth > 0)
            & (original_depth >= float(depth_range['min']))
            & (original_depth <= min(float(depth_range['max']), 65535)), original_depth, 0)
        color_range = _depth_color_range(original_depth)
        timer.mark('depth_color_range')
        depth_preview = _depth_preview(original_depth, color_range)
        segmented_preview = _depth_preview(depth, color_range)
        timer.mark('colorize_preview')
        if log_preview:
            extra['preview_timings_ms'] = timer.finish()
            print(f'[Preview frame {raw_frame[0]}] {timing_summary(extra["preview_timings_ms"])}', flush=True)
        return preview, depth_preview, segmented_preview, status

    def start_continuous_capture(self, interval):
        interval = float(interval)
        if not np.isfinite(interval) or interval < 0:
            raise ValueError('连续拍摄间隔须为不小于 0 秒的有限数值')
        with self.lock:
            if self.reconstruction is None or self.process is None or not self.process.is_alive():
                raise RuntimeError('请先点击开始并等待相机画面')
            self.capture_interval = interval
            self.next_capture_at = 0.0
            self.continuous_capture = True
            self.message = f'连续拍摄已开启，间隔 {interval:g} 秒'
            return self._status()

    def stop_continuous_capture(self):
        with self.lock:
            self.continuous_capture = False
            self.message = '连续拍摄已停止，相机继续预览'
            return self._status()

    def continuous_capture_tick(self):
        with self.lock:
            if not self.continuous_capture or perf_counter() < self.next_capture_at:
                return (gr.skip(),) * 10
            if self.process is None or not self.process.is_alive():
                self.continuous_capture = False
                self.message = '连续拍摄已停止：相机未运行'
                return (self._status(),) + (gr.skip(),) * 9
            try:
                self._drain_frames()
                if self.latest is None or self.latest[0] == self.last_captured_sequence:
                    return (gr.skip(),) * 10
                return self.capture()
            except Exception as exc:
                self.continuous_capture = False
                self.message = f'连续拍摄已停止：{exc}'
                return (self._status(),) + (gr.skip(),) * 9
            finally:
                # Zero interval captures the next available frame without an added delay.
                self.next_capture_at = perf_counter() + self.capture_interval

    def capture(self):
        timer = StageTimer()
        with self.lock:
            timer.mark('lock_wait')
            if self.reconstruction is None or self.process is None or not self.process.is_alive():
                raise RuntimeError('请先点击开始并等待相机画面')
            self._drain_frames()
            if self.latest is None:
                raise RuntimeError('尚未收到相机帧')
            sequence, gray, depth, mask = self.latest
            if sequence == self.last_captured_sequence:
                raise RuntimeError('尚未收到新帧，请稍后再拍摄')
            raw_depth = self.latest_raw[1]
            raw_mask = self.latest_raw[3]
            right = self.latest_raw[4] if len(self.latest_raw) > 4 else None
            extra = self.latest_raw[5] if len(self.latest_raw) > 5 else {}
            raw_stats = self.latest_stats.copy()
            idx = self.captured_count
            sample = {
                'idx': idx,
                'gray_left': gray.copy(),
                'depth': depth.copy(),
                'mask': mask.copy(),
                'rgb': extra.get('rgb'),
                'camera_params': {
                    'K1': self.camera_matrix,
                    'min_dis': float(self.config['reconstruction']['depth_range_mm']['min']),
                    'max_dis': float(self.config['reconstruction']['depth_range_mm']['max']),
                },
            }
            timer.mark('receive_prepare_sample')
            # Save every captured RGB-D frame before checking foreground or poses.
            for folder, image in (('gray', gray), ('depth', depth), ('mask', mask)):
                path = self.raw_dir / folder / f'frame_{idx:04d}.png'
                if not cv2.imwrite(str(path), image):
                    raise OSError(f'无法保存 {path}')
            if right is not None and not cv2.imwrite(str(self.raw_dir / 'right' / f'frame_{idx:04d}.png'), right):
                raise OSError('无法保存右红外图')
            for folder, key in (('rgb', 'rgb'), ('foreground_mask', 'foreground'), ('left_ir', 'left')):
                if key in extra and not cv2.imwrite(str(self.raw_dir / folder / f'frame_{idx:04d}.png'), extra[key]):
                    raise OSError(f'无法保存 {folder}')
            for key in ('aligned_depth_mm', 'stereo_depth_mm'):
                if key in extra:
                    np.save(self.raw_dir / key / f'frame_{idx:04d}.npy', extra[key])
            np.save(self.raw_dir / 'sdk_depth_mm' / f'frame_{idx:04d}.npy', raw_depth)
            sdk_mask_path = self.raw_dir / 'sdk_mask' / f'frame_{idx:04d}.png'
            if not cv2.imwrite(str(sdk_mask_path), raw_mask):
                raise OSError(f'无法保存 {sdk_mask_path}')
            full_depth_mm = np.asarray(extra.get('aligned_depth_mm', raw_depth))
            full_depth = np.zeros(full_depth_mm.shape, dtype=np.uint16)
            valid = np.isfinite(full_depth_mm) & (full_depth_mm > 0) & (full_depth_mm <= 65535)
            full_depth[valid] = np.rint(full_depth_mm[valid]).astype(np.uint16)
            full_depth_path = self.raw_dir / 'depth_full' / f'frame_{idx:04d}.png'
            if not cv2.imwrite(str(full_depth_path), full_depth):
                raise OSError(f'无法保存 {full_depth_path}')
            # Keep captured images visible beside session.json, including failed frames.
            output_images = [('gray', gray), ('depth_full', full_depth)]
            if 'rgb' in extra:
                output_images.append(('rgb', extra['rgb']))
            for folder, image in output_images:
                output_folder = self.output_dir / folder
                output_folder.mkdir(parents=True, exist_ok=True)
                output_path = output_folder / f'frame_{idx:04d}.png'
                if not cv2.imwrite(str(output_path), image):
                    raise OSError(f'无法保存 {output_path}')
            timer.mark('save_raw_inputs')
            # A saved frame owns its index even if reconstruction later fails.
            self.captured_count += 1
            self.last_captured_sequence = sequence
            self.message = f'第 {idx + 1} 帧 RGB-D 已保存到 {self.raw_dir}'
            try:
                result = self.reconstruction.process_frame(sample)
                full_cloud_path, full_world_path = self._save_unsegmented_cloud(
                    idx, gray, extra.get('rgb'), result['frame_pose'])
                timer.mark('reconstruct')
                pose_summary = _aruco_summary(
                    result['aruco_est_frame'], result['frame_pose'], self.reconstruction.aruco_estimater
                )
                timer.mark('aruco_summary')
            except Exception as exc:
                return self._capture_without_cloud(
                    idx, sequence, raw_stats, f'重建失败：{exc}')
            self.reconstruction.save_frame_result(result)
            timer.mark('save_frame_result')
            if len(self.reconstruction.global_cloud.points) > 0:
                fused = self.reconstruction.global_cloud.voxel_down_sample(
                    voxel_size=self.reconstruction.world_voxel_size_mm
                )
                timer.mark('global_voxel_downsample')
                if not o3d.io.write_point_cloud(self.reconstruction.global_cloud_path, fused):
                    raise OSError(f'无法保存全局点云 {self.reconstruction.global_cloud_path}')
                self.global_display_path = self.reconstruction.global_cloud_path
            timer.mark('save_global_cloud')
            # Save poses after each capture without closing the video writer.
            with open(self.reconstruction.pose_yaml_path, 'w', encoding='utf-8') as pose_file:
                yaml.safe_dump(
                    {'camera_poses': self.reconstruction.pose_records},
                    pose_file,
                    allow_unicode=True,
                    sort_keys=False,
                )
            timer.mark('save_pose_yaml')
            capture_timings = timer.finish()
            print(f'[Capture {idx}] {timing_summary(capture_timings)}', flush=True)
            print(f'[Reconstruction {idx}] {timing_summary(result["timings_ms"])}', flush=True)
            with (self.output_dir / 'captures.jsonl').open('a', encoding='utf-8') as log_file:
                log_file.write(json.dumps({
                    'idx': idx,
                    'camera_sequence': sequence,
                    'aruco_pose_found': result['frame_pose'] is not None,
                    'aruco': pose_summary,
                    'raw_depth_stats': raw_stats,
                    'timings_ms': {
                        'camera': extra.get('timings_ms'),
                        'stereo': extra.get('stereo_timings_ms'),
                        'delivery_age_ms': extra.get('delivery_age_ms'),
                        'receive': extra.get('receive_timings_ms'),
                        'preview': extra.get('preview_timings_ms'),
                        'reconstruction': result['timings_ms'],
                        'capture_before_log': capture_timings,
                    },
                    'cloud': str(self.output_dir / 'cloud' / f'cloud_{idx:04d}.ply'),
                    'cloud_full': full_cloud_path,
                    'cloud_world_full': full_world_path,
                    'global_cloud_full': self.full_global_display_path,
                }, ensure_ascii=False) + '\n')
            self.pose_summaries[idx] = pose_summary
            cloud_path = self.output_dir / 'cloud' / f'cloud_{idx:04d}.ply'
            cloud_display = str(cloud_path) if cloud_path.is_file() else None
            foreground_status = '前景深度为空；' if not np.any(depth) else ''
            self.message = (f'已拍摄第 {idx + 1} 帧；RGB-D 已保存；{foreground_status}'
                            f'{pose_summary["位姿状态"]}；未分割全局点云 {len(self.full_global_cloud.points)} 个点')
            choices = [(f'第 {i + 1} 帧 · {i:04d}', i) for i in range(self.captured_count)]
            pose_image = None if result['pose_vis'] is None else cv2.cvtColor(result['pose_vis'], cv2.COLOR_BGR2RGB)
            timer.mark('log_and_ui_prepare')
            print(f'[Capture complete {idx}] {timing_summary(timer.finish())}', flush=True)
            return (
                self._status(), gr.update(choices=choices, value=idx),
                cloud_display, cloud_display,
                self.global_display_path, self.global_display_path,
                pose_image, pose_summary,
                self.full_global_display_path, self.full_global_display_path,
            )

    def _save_unsegmented_cloud(self, idx, gray, rgb, frame_pose):
        """Retain every valid depth pixel, independently of foreground filters."""
        depth = _unsegmented_depth(self.latest_raw, self.image_size)
        color = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if rgb is not None else gray
        cloud = Reconstruction.depth_to_point_cloud(depth, color, self.camera_matrix)
        if not len(cloud.points):
            return None, None
        folder = self.output_dir / 'cloud_full'
        folder.mkdir(parents=True, exist_ok=True)
        cloud_path = folder / f'cloud_full_{idx:04d}.ply'
        if not o3d.io.write_point_cloud(str(cloud_path), cloud):
            raise OSError(f'无法保存 {cloud_path}')
        if frame_pose is None:
            return str(cloud_path), None
        world = self.reconstruction.aruco_estimater.transform_point_cloud(cloud, frame_pose)
        world_folder = self.output_dir / 'cloud_world_full'
        world_folder.mkdir(parents=True, exist_ok=True)
        world_path = world_folder / f'cloud_world_full_{idx:04d}.ply'
        if not o3d.io.write_point_cloud(str(world_path), world):
            raise OSError(f'无法保存 {world_path}')
        self.full_global_cloud += world
        global_path = self.output_dir / 'global_cloud_full.ply'
        if not o3d.io.write_point_cloud(str(global_path), self.full_global_cloud):
            raise OSError(f'无法保存 {global_path}')
        self.full_global_display_path = str(global_path)
        return str(cloud_path), str(world_path)

    def _capture_without_cloud(self, idx, sequence, raw_stats, reason):
        summary = {'位姿状态': reason, 'RGB-D已保存': True, '原始数据目录': str(self.raw_dir)}
        self.pose_summaries[idx] = summary
        self.message = f'已拍摄第 {idx + 1} 帧；RGB-D 已保存；{reason}'
        with (self.output_dir / 'captures.jsonl').open('a', encoding='utf-8') as log_file:
            log_file.write(json.dumps({
                'idx': idx,
                'camera_sequence': sequence,
                'aruco_pose_found': False,
                'aruco': summary,
                'raw_depth_stats': raw_stats,
                'cloud': None,
                'reconstruction_error': reason,
                'raw_dir': str(self.raw_dir),
            }, ensure_ascii=False) + '\n')
        choices = [(f'第 {i + 1} 帧 · {i:04d}', i) for i in range(self.captured_count)]
        return (
            self._status(), gr.update(choices=choices, value=idx), None, None,
            self.global_display_path, self.global_display_path, None, summary,
            self.full_global_display_path, self.full_global_display_path,
        )

    def select_frame(self, idx):
        with self.lock:
            if idx is None or self.output_dir is None:
                return None, None, None, None
            frame_idx = int(idx)
            if not 0 <= frame_idx < self.captured_count:
                raise ValueError('所选帧不在当前会话中')
            cloud_path = self.output_dir / 'cloud' / f'cloud_{frame_idx:04d}.ply'
            pose_path = self.output_dir / 'aruco_fusion' / 'pose_vis' / f'frame_{frame_idx:04d}.png'
            pose_image = cv2.imread(str(pose_path)) if pose_path.is_file() else None
            if pose_image is not None:
                pose_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            cloud_display = str(cloud_path) if cloud_path.is_file() else None
            return cloud_display, cloud_display, pose_image, self.pose_summaries.get(frame_idx)

    def optimize(self, *graph_values):
        with self.lock:
            if self.reconstruction is None:
                raise RuntimeError('请先开始采集')
            if not self.config['global_registration']['enabled']:
                raise RuntimeError('配置中 global_registration.enabled 为 false')
            if len(self.reconstruction.pose_graph_entries) < 2:
                raise RuntimeError('至少需要两帧成功识别 ArUco 并保存点云后才能优化')
            method = self.config['global_registration']['method']
            graph_settings = self.software_settings['pose_graph']
            if method == 'pose_graph' and graph_values:
                software_settings = _software_settings(
                    self.software_settings['statistical_nb_neighbors'],
                    self.software_settings['statistical_std_ratio'],
                    self.software_settings['aruco_min_joint_calibration_frames'],
                    *graph_values,
                )
                graph_settings = software_settings['pose_graph']
            if method == 'pose_graph' and graph_settings['reference_node'] >= len(self.reconstruction.pose_graph_entries):
                raise ValueError('固定节点序号不能超过当前位姿图节点数')
            result = self.reconstruction.run_global_registration(
                method=method,
                use_sem=self.reconstruction.use_sem,
                **(graph_settings if method == 'pose_graph'
                   else self.config['global_registration'][method]),
            )
            if result is None:
                raise RuntimeError('图优化未生成结果，请检查有效点云和 ArUco 位姿')
            if method == 'pose_graph':
                self.software_settings['pose_graph'] = graph_settings
                with (self.output_dir / 'pose_graph_runs.jsonl').open('a', encoding='utf-8') as log_file:
                    log_file.write(json.dumps({
                        'time': datetime.now().isoformat(timespec='seconds'),
                        'settings': graph_settings,
                        'output_path': result['output_path'],
                        'points': result['points'],
                    }, ensure_ascii=False) + '\n')
            self.global_display_path = result['output_path']
            self.message = f'全局图优化完成：{result["points"]} 个点'
            return self._status(), result['output_path'], result['output_path']

    def stop(self):
        with self.lock:
            self._stop_camera()
            if self.reconstruction is not None:
                self.reconstruction.save_final_result()
            self.reconstruction = None
            self.message = '已停止采集'
            return self._status()


def build_ui(controller):
    with gr.Blocks(title='双目实时重建') as demo:
        gr.Markdown('# 盛相 / RealSense D455 实时重建\n开始后预览相机；拍摄时记录一帧灰度、深度、mask、点云和 ArUco 位姿。')
        camera_type = gr.Dropdown(label='相机类型', choices=[('盛相相机', 'shengxiang'), ('RealSense D455', 'realsense_d455')], value=controller.camera_settings['camera_type'])
        with gr.Accordion('双目深度推理参数（D455）', open=False):
            stereo_method = gr.Dropdown(label='双目推理模式', choices=list(STEREO_METHODS), value=controller.camera_settings['stereo_method'])
            checkpoint = gr.Textbox(label='模型权重路径（FoundationStereo 需同目录 cfg.yaml；Fast 使用序列化权重）', value=controller.camera_settings['checkpoint'])
            valid_iters = gr.Number(label='推理迭代次数', value=controller.camera_settings['valid_iters'], precision=0)
            gr.Markdown('D455 固定使用 1280×720 左右红外流和彩色流，关闭投射器，自动曝光；下方盛相专用参数对 D455 不生效。模型加载最多等待 180 秒，预览速度取决于推理耗时。')
        stereo_method.change(
            lambda method: (default_checkpoint(method), 8 if method == 'fast_foundation_stereo' else 32),
            inputs=stereo_method, outputs=[checkpoint, valid_iters], queue=False,
        )
        board_choices = [
            ('四码双板：ID 0–3 / 4–7；码边长 24 mm，中心距 32 mm', 'config/aruco_board/main.yaml'),
            ('十二码双板：ID 0–11 / 12–23；码边长 27 mm，中心距 36 mm', 'config/aruco_board/shenxiang.yaml'),
        ]
        current_board = controller.camera_settings['aruco_board_yaml_path']
        if current_board not in {value for _, value in board_choices}:
            board_choices.append((f'配置中的布局：{current_board}', current_board))
        board_type = gr.Dropdown(label='ArUco 标定板类型', choices=board_choices, value=current_board)
        gr.Markdown('开始前选择与实物一致的板类型，点击“开始”即加载对应码 ID、布局和尺寸；选择板型无需预先拍摄标定帧。运行中更换板型，请先停止再开始，新会话重新建立世界坐标系，原会话文件保留。')
        fg = controller.camera_settings['foreground']
        with gr.Accordion('YOLO 前景分割（两种相机默认启用）', open=True):
            yolo_weights = gr.Textbox(label='YOLO 分割权重路径', value=fg['weights'])
            with gr.Row():
                yolo_conf = gr.Number(label='YOLO 置信度', value=fg['conf'])
                yolo_imgsz = gr.Number(label='YOLO 推理尺寸', value=fg['imgsz'], precision=0)
                yolo_device = gr.Textbox(label='YOLO 设备（0 为首张 GPU，cpu 为 CPU，留空自动）', value=str(fg['device']) if fg['device'] is not None else '')
            gr.Markdown('RealSense 在完整 RGB 上分割，盛相在灰度图转三通道后分割。前景点云与未分割点云分别保存；未分割点云使用全部有效深度，不受前景分割、软件深度范围、亮度掩码、统计滤波或体素降采样影响。ArUco 在完整原图上定位；未解算世界位姿的帧只保存单帧点云，不加入全局融合。参数在停止后重新开始时生效。')
        foreground_inputs = [yolo_weights, yolo_conf, yolo_imgsz, yolo_device]
        with gr.Accordion('相机拍摄参数（停止后调整，点击开始生效）', open=True):
            with gr.Row():
                device_index = gr.Number(label='设备序号', value=controller.camera_settings['device_index'], precision=0)
                binning = gr.Checkbox(label='2×2 Binning', value=controller.camera_settings['binning'])
                working_mode = gr.Dropdown(
                    label='3D 工作模式', choices=['super_precise_3d', 'fast_3d'],
                    value=controller.camera_settings['working_mode'],
                )
                exposure_mode = gr.Dropdown(
                    label='曝光模式', choices=['manual', 'manual_repeat', 'auto_nhdr', 'auto_phdr'],
                    value=controller.camera_settings['exposure_mode'],
                )
            with gr.Row():
                exposure_3d = gr.Slider(0.1, 100, value=controller.camera_settings['exposure_3d'], step=0.1, label='3D 曝光强度')
                exposure_gray = gr.Slider(0.1, 100, value=controller.camera_settings['exposure_gray'], step=0.1, label='灰度曝光强度')
                user_gain = gr.Slider(0, 5, value=controller.camera_settings['user_gain'], step=1, label='用户增益')
            with gr.Row():
                exposure_number = gr.Slider(1, 3, value=controller.camera_settings['exposure_number'], step=1, label='曝光次数')
                auto_hdr_priority = gr.Slider(0, 7, value=controller.camera_settings['auto_hdr_priority'], step=1, label='自动 HDR 优先级')
                auto_phdr_quality = gr.Slider(0, 100, value=controller.camera_settings['auto_phdr_quality'], step=1, label='自动 PHDR 质量')
            gr.Markdown('自动曝光模式下，SDK 会自行决定 3D 曝光；手动 3D 曝光值主要用于 manual / manual_repeat。')
        camera_inputs = [
            device_index, binning, working_mode, exposure_mode, exposure_3d,
            exposure_gray, user_gain, exposure_number, auto_hdr_priority, auto_phdr_quality,
        ]
        software = controller.software_settings
        graph = software['pose_graph']
        with gr.Accordion('软件设置', open=False):
            gr.Markdown('统计滤波作用于每帧普通点云；邻居数设为 0 可关闭。板间标定要求两块板在同一帧都达到各自所需的码数。')
            with gr.Row():
                neighbors = gr.Number(label='每帧统计滤波邻居数', value=software['statistical_nb_neighbors'], precision=0)
                std_ratio = gr.Number(label='每帧统计滤波标准差倍数', value=software['statistical_std_ratio'])
                joint_frames = gr.Number(label='板间共同观测最少帧数', value=software['aruco_min_joint_calibration_frames'], precision=0)
            gr.Markdown('统计滤波和共同观测帧数在停止后再次点击“开始”时生效。以下图优化参数在点击“全局图优化”时直接生效。')
            with gr.Row():
                voxel_size = gr.Number(label='配准体素大小（mm）', value=graph['voxel_size_mm'])
                final_voxel_size = gr.Number(label='融合体素大小（mm；0 不降采样）', value=graph['final_voxel_size_mm'])
                correspondence_distance = gr.Number(label='连接最大对应距离（mm）', value=graph['max_correspondence_distance_mm'])
            with gr.Row():
                max_iteration = gr.Number(label='每条连接 ICP 最大迭代次数', value=graph['max_iteration'], precision=0)
                loop_stride = gr.Number(label='回环连接帧间隔（0 关闭）', value=graph['loop_closure_stride'], precision=0)
                loop_preference = gr.Number(label='回环连接权重', value=graph['preference_loop_closure'])
            with gr.Row():
                edge_prune = gr.Number(label='连接剪枝阈值', value=graph['edge_prune_threshold'])
                reference_node = gr.Number(label='固定节点序号', value=graph['reference_node'], precision=0)
        software_inputs = [
            neighbors, std_ratio, joint_frames, voxel_size, final_voxel_size,
            correspondence_distance, max_iteration, loop_stride, loop_preference,
            edge_prune, reference_node,
        ]
        graph_inputs = software_inputs[3:]
        with gr.Row():
            start = gr.Button('开始', variant='primary')
            capture = gr.Button('拍摄一帧')
            continuous_start = gr.Button('连续拍摄')
            continuous_stop = gr.Button('停止连续拍摄')
            optimize = gr.Button('全局图优化')
            stop = gr.Button('停止')
        capture_interval = gr.Number(label='连续拍摄间隔（秒，0 表示不等待）', value=0.0, minimum=0)
        status = gr.Textbox(label='状态', value='未启动', interactive=False)
        with gr.Row():
            gray = gr.Image(label='原图像（D455 RGB / 盛相灰度）', interactive=False)
            depth = gr.Image(label='深度图（伪彩）', interactive=False)
            segmented_depth = gr.Image(label='分割后的深度图（伪彩）', interactive=False)
        with gr.Accordion('ArUco 位姿估计', open=True):
            with gr.Row():
                pose_image = gr.Image(label='pose_vis（当前选中帧）', interactive=False)
                pose_summary = gr.JSON(label='检测到的板子与码数、世界位姿状态')
        frame_selector = gr.Dropdown(label='选择单帧点云', choices=[], value=None)
        with gr.Row():
            single_cloud = gr.Model3D(label='单帧前景点云', display_mode='point_cloud', interactive=False)
            global_cloud = gr.Model3D(label='前景全局点云', display_mode='point_cloud', interactive=False)
        with gr.Row():
            single_file = gr.File(label='下载单帧点云', interactive=False)
            global_file = gr.File(label='下载全局点云', interactive=False)
        full_global_cloud = gr.Model3D(label='未语义分割全局点云（全部有效深度像素）', display_mode='point_cloud', interactive=False)
        full_global_file = gr.File(label='下载未语义分割全局点云', interactive=False)
        timer = gr.Timer(0.7)
        capture_timer = gr.Timer(0.2)

        def start_from_ui(*values):
            selected_type, method, model_path, iterations, board_path = values[:5]
            weights, conf, imgsz, device = values[5:9]
            camera_values = values[9:9 + len(camera_inputs)]
            if selected_type == 'realsense_d455':
                settings = dict(camera_type=selected_type, device_index=int(camera_values[0]),
                                checkpoint=model_path, valid_iters=int(iterations))
                if settings['device_index'] < 0 or settings['valid_iters'] < 1:
                    raise ValueError('设备序号须 ≥ 0，推理迭代次数须 ≥ 1')
            else:
                settings = _camera_settings(*camera_values)
                settings.update(camera_type=selected_type, checkpoint=model_path, valid_iters=int(iterations))
            settings['stereo_method'] = method
            settings['aruco_board_yaml_path'] = board_path
            settings['foreground'] = dict(enabled=True, weights=weights, conf=float(conf), imgsz=int(imgsz), device=device or None)
            software_settings = _software_settings(*values[9 + len(camera_inputs):])
            already_running = controller.process is not None and controller.process.is_alive()
            message = controller.start(settings, software_settings)
            if already_running:
                return (message,) + (gr.skip(),) * 9
            return message, gr.update(choices=[], value=None), None, None, None, None, None, None, None, None

        def stop_from_ui():
            message = controller.stop()
            return message, controller.global_display_path, controller.global_display_path

        start.click(
            start_from_ui, inputs=[camera_type, stereo_method, checkpoint, valid_iters, board_type] + foreground_inputs + camera_inputs + software_inputs,
            outputs=[status, frame_selector, single_cloud, single_file, global_cloud, global_file, pose_image, pose_summary, full_global_cloud, full_global_file],
            concurrency_limit=1,
        )
        capture.click(
            controller.capture,
            outputs=[status, frame_selector, single_cloud, single_file, global_cloud, global_file, pose_image, pose_summary, full_global_cloud, full_global_file],
            concurrency_limit=1,
        )
        continuous_start.click(
            controller.start_continuous_capture, inputs=capture_interval, outputs=status,
            concurrency_limit=1,
        )
        continuous_stop.click(
            controller.stop_continuous_capture, outputs=status, concurrency_limit=1,
        )
        capture_timer.tick(
            controller.continuous_capture_tick,
            outputs=[status, frame_selector, single_cloud, single_file, global_cloud, global_file, pose_image, pose_summary, full_global_cloud, full_global_file],
            concurrency_limit=1, show_progress='hidden',
        )
        frame_selector.input(
            controller.select_frame, inputs=frame_selector,
            outputs=[single_cloud, single_file, pose_image, pose_summary], concurrency_limit=1,
        )
        optimize.click(controller.optimize, inputs=graph_inputs, outputs=[status, global_cloud, global_file], concurrency_limit=1)
        stop.click(stop_from_ui, outputs=[status, global_cloud, global_file], concurrency_limit=1)
        timer.tick(controller.poll, outputs=[gray, depth, segmented_depth, status], concurrency_limit=1, show_progress='hidden')
    return demo


def main():
    parser = argparse.ArgumentParser(description='Shengxiang / RealSense D455 live reconstruction UI')
    parser.add_argument('--config', default=None, help='软件配置，默认 config/recon/shenxiang.yaml；D455 双目标定读取 biaoding 中的 720p 文件')
    parser.add_argument('--camera-type', choices=['shengxiang', 'realsense_d455'], default='shengxiang')
    parser.add_argument('--stereo-method', choices=STEREO_METHODS, default='foundation_stereo')
    parser.add_argument('--checkpoint', default=None, help='留空时按双目模式选择默认权重')
    parser.add_argument('--valid-iters', type=int, default=None, help='默认 FoundationStereo 32 次，Fast 8 次')
    parser.add_argument('--yolo', action='store_true', help=argparse.SUPPRESS)  # Legacy flag; YOLO is always enabled.
    parser.add_argument('--yolo-weights', default=None)
    parser.add_argument('--yolo-conf', type=float, default=None)
    parser.add_argument('--yolo-imgsz', type=int, default=640, help='YOLO 推理尺寸，如 640 或 1280；不改变采集及保存分辨率')
    parser.add_argument('--yolo-device', default=None)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--exposure-3d', type=float, default=100.0, help='手动 3D 曝光强度，0.1–100，默认 100')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    config_path = args.config or 'config/recon/shenxiang.yaml'
    controller = LiveReconstruction(config_path, args.device_index, args.exposure_3d,
                                    args.camera_type, args.checkpoint, args.valid_iters, stereo_method=args.stereo_method)
    if args.camera_type == 'realsense_d455' and args.config is None:
        controller.camera_settings['aruco_board_yaml_path'] = 'config/aruco_board/main.yaml'
    foreground = controller.camera_settings['foreground']
    for key, value in [('weights', args.yolo_weights), ('conf', args.yolo_conf), ('imgsz', args.yolo_imgsz), ('device', args.yolo_device)]:
        if value is not None:
            foreground[key] = value
    try:
        build_ui(controller).queue().launch(server_name='127.0.0.1', server_port=args.port)
    finally:
        controller.stop()


if __name__ == '__main__':
    main()
