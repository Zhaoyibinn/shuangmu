"""Live Shengxiang camera capture and one-frame-at-a-time reconstruction UI."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import sys
import threading
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import open3d as o3d
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Recon.Reconstruction import Reconstruction
from Recon.config_loader import load_config


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
            frame = camera.get_latest_zmap_frame(timeout_ms=1000)
            if frame is None:
                continue
            sequence += 1
            item = (sequence, frame[0], frame[1], frame[2])
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


def _prepare_frame(raw_frame, image_size, min_depth, max_depth):
    sequence, depth_mm, gray, sdk_mask = raw_frame
    if depth_mm.shape != gray.shape or gray.shape != sdk_mask.shape:
        raise ValueError('相机的深度、灰度和 mask 尺寸不一致')
    valid = ((sdk_mask & 0x07) == 0) & np.isfinite(depth_mm)
    valid &= (depth_mm >= min_depth) & (depth_mm <= max_depth)
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


def _aruco_summary(frame, pose, estimator):
    _, detected_ids, _ = estimator.detector.detectMarkers(frame)
    marker_ids = set() if detected_ids is None else {int(value) for value in detected_ids.ravel()}
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
        '检测到的板子数': len(boards),
        '板子': boards,
        '锚点板子 ID': None if estimator.anchor_marker_id is None else int(estimator.anchor_marker_id),
        '本帧参考板子 ID': None if pose is None else int(pose['reference_board_id']),
    }
    if pose is not None:
        summary['相机到世界平移_mm'] = np.asarray(pose['translation_cam_to_world']).reshape(3).tolist()
        summary['相机到世界旋转矩阵'] = np.asarray(pose['rotation_cam_to_world']).tolist()
    return summary


def _depth_preview(depth):
    valid = depth[depth > 0]
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if valid.size:
        low, high = np.percentile(valid, (2, 98))
        high = max(high, low + 1)
        scaled = np.clip((depth.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    color[depth == 0] = 0
    return color


class LiveReconstruction:
    def __init__(self, config_path, device_index=0, exposure_3d=100.0):
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
        self.global_display_path = None
        self.pose_summaries = {}
        self.message = '未启动'

    def _drain_frames(self):
        if self.frame_queue is None:
            return
        while True:
            try:
                raw_frame = self.frame_queue.get_nowait()
            except Empty:
                break
            self.latest_raw = raw_frame
            self.latest_stats = _raw_depth_stats(
                raw_frame[1], raw_frame[3],
                float(self.config['reconstruction']['depth_range_mm']['min']),
                float(self.config['reconstruction']['depth_range_mm']['max']),
            )
            self.latest = _prepare_frame(
                raw_frame,
                self.image_size,
                float(self.config['reconstruction']['depth_range_mm']['min']),
                float(self.config['reconstruction']['depth_range_mm']['max']),
            )

    def _status(self):
        diagnostics = ''
        if self.latest_stats is not None:
            stats = self.latest_stats
            raw_min = '无' if stats['raw_min_mm'] is None else f'{stats["raw_min_mm"]:.0f} mm'
            sdk_min = '无' if stats['sdk_valid_min_mm'] is None else f'{stats["sdk_valid_min_mm"]:.0f} mm'
            diagnostics = (
                f'｜原始最近深度：{raw_min}；SDK 有效最近深度：{sdk_min}'
                f'；低于软件下限：{stats["raw_below_min_count"]} 像素'
            )
        return f'{self.message}｜已拍摄 {self.captured_count} 帧{diagnostics}｜输出：{self.output_dir or "未创建"}'

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
            target=_camera_worker,
            args=(self.frame_queue, child_pipe, self.stop_event, settings),
            daemon=True,
        )
        self.process.start()
        child_pipe.close()
        if not parent_pipe.poll(20):
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
            self.camera_settings = settings or self.camera_settings
            self.software_settings = software_settings or self.software_settings
            detail = self._launch_camera(self.camera_settings)

            stamp = datetime.now().strftime('realtime_%Y%m%d_%H%M%S')
            self.raw_dir = _project_path(self.config['paths']['data_root_path']) / stamp
            self.output_dir = _project_path(self.config['paths']['save_dir']) / stamp
            for name in ('gray', 'depth', 'mask', 'sdk_depth_mm', 'sdk_mask'):
                (self.raw_dir / name).mkdir(parents=True, exist_ok=True)
            recon = self.config['reconstruction']
            seg = self.config['segmentation']
            reconstruction = Reconstruction(
                data_root_path=str(self.raw_dir),
                color_ext_yaml_path=str(_project_path(self.config['paths']['int_yaml_path'])),
                aruco_board_yaml_path=str(_project_path(self.config['paths']['aruco_board_yaml_path'])),
                aruco_length=recon['aruco_length'],
                world_voxel_size_mm=recon['world_voxel_size_mm'],
                method=recon['stereo_method'],
                use_aruco=True,
                brightness_mask_enabled=recon['brightness_mask']['enabled'],
                brightness_threshold=recon['brightness_mask']['threshold'],
                use_sem=seg['enabled'],
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
            self.pose_summaries = {}
            self.latest = self.latest_raw = self.latest_stats = None
            self.message = f'相机运行中：{detail.get("name", "设备已连接")}'
            return self._status()

    def poll(self):
        with self.lock:
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
            if self.latest is None:
                return None, None, None, self._status()
            _, gray, depth, mask = self.latest
            return gray, _depth_preview(depth), mask, self._status()

    def capture(self):
        with self.lock:
            if self.reconstruction is None or self.process is None or not self.process.is_alive():
                raise RuntimeError('请先点击开始并等待相机画面')
            self._drain_frames()
            if self.latest is None:
                raise RuntimeError('尚未收到相机帧')
            sequence, gray, depth, mask = self.latest
            if sequence == self.last_captured_sequence:
                raise RuntimeError('尚未收到新帧，请稍后再拍摄')
            if not np.any(depth):
                diagnostic_dir = self.raw_dir / 'diagnostics'
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
                np.save(diagnostic_dir / f'sdk_depth_{sequence:06d}.npy', self.latest_raw[1])
                cv2.imwrite(str(diagnostic_dir / f'sdk_mask_{sequence:06d}.png'), self.latest_raw[3])
                raise RuntimeError(f'当前帧没有有效深度；SDK 原始数据已保存到 {diagnostic_dir}')
            raw_depth = self.latest_raw[1]
            raw_mask = self.latest_raw[3]
            raw_stats = self.latest_stats.copy()
            idx = self.captured_count
            sample = {
                'idx': idx,
                'gray_left': gray.copy(),
                'depth': depth.copy(),
                'mask': mask.copy(),
                'rgb': None,
                'camera_params': {
                    'K1': self.camera_matrix,
                    'min_dis': float(self.config['reconstruction']['depth_range_mm']['min']),
                    'max_dis': float(self.config['reconstruction']['depth_range_mm']['max']),
                },
            }
            result = self.reconstruction.process_frame(sample)
            pose_summary = _aruco_summary(
                result['aruco_est_frame'], result['frame_pose'], self.reconstruction.aruco_estimater
            )
            # Persist raw inputs only after successful reconstruction; repeat clicks
            # cannot overwrite a previous captured frame.
            for folder, image in (('gray', gray), ('depth', depth), ('mask', mask)):
                path = self.raw_dir / folder / f'frame_{idx:04d}.png'
                if not cv2.imwrite(str(path), image):
                    raise OSError(f'无法保存 {path}')
            np.save(self.raw_dir / 'sdk_depth_mm' / f'frame_{idx:04d}.npy', raw_depth)
            sdk_mask_path = self.raw_dir / 'sdk_mask' / f'frame_{idx:04d}.png'
            if not cv2.imwrite(str(sdk_mask_path), raw_mask):
                raise OSError(f'无法保存 {sdk_mask_path}')
            self.reconstruction.save_frame_result(result)
            if len(self.reconstruction.global_cloud.points) > 0:
                fused = self.reconstruction.global_cloud.voxel_down_sample(
                    voxel_size=self.reconstruction.world_voxel_size_mm
                )
                if not o3d.io.write_point_cloud(self.reconstruction.global_cloud_path, fused):
                    raise OSError(f'无法保存全局点云 {self.reconstruction.global_cloud_path}')
                self.global_display_path = self.reconstruction.global_cloud_path
            # Save poses after each capture without closing the video writer.
            with open(self.reconstruction.pose_yaml_path, 'w', encoding='utf-8') as pose_file:
                yaml.safe_dump(
                    {'camera_poses': self.reconstruction.pose_records},
                    pose_file,
                    allow_unicode=True,
                    sort_keys=False,
                )
            with (self.output_dir / 'captures.jsonl').open('a', encoding='utf-8') as log_file:
                log_file.write(json.dumps({
                    'idx': idx,
                    'camera_sequence': sequence,
                    'aruco_pose_found': result['frame_pose'] is not None,
                    'aruco': pose_summary,
                    'raw_depth_stats': raw_stats,
                    'cloud': str(self.output_dir / 'cloud' / f'cloud_{idx:04d}.ply'),
                }, ensure_ascii=False) + '\n')
            self.captured_count += 1
            self.last_captured_sequence = sequence
            self.pose_summaries[idx] = pose_summary
            cloud_path = self.output_dir / 'cloud' / f'cloud_{idx:04d}.ply'
            self.message = f'已拍摄第 {idx + 1} 帧；{pose_summary["位姿状态"]}'
            choices = [(f'第 {i + 1} 帧 · {i:04d}', i) for i in range(self.captured_count)]
            pose_image = None if result['pose_vis'] is None else cv2.cvtColor(result['pose_vis'], cv2.COLOR_BGR2RGB)
            return (
                self._status(), gr.update(choices=choices, value=idx),
                str(cloud_path), str(cloud_path),
                self.global_display_path, self.global_display_path,
                pose_image, pose_summary,
            )

    def select_frame(self, idx):
        with self.lock:
            if idx is None or self.output_dir is None:
                return None, None, None, None
            frame_idx = int(idx)
            if not 0 <= frame_idx < self.captured_count:
                raise ValueError('所选帧不在当前会话中')
            cloud_path = self.output_dir / 'cloud' / f'cloud_{frame_idx:04d}.ply'
            if not cloud_path.is_file():
                raise FileNotFoundError(f'单帧点云不存在：{cloud_path}')
            pose_path = self.output_dir / 'aruco_fusion' / 'pose_vis' / f'frame_{frame_idx:04d}.png'
            pose_image = cv2.imread(str(pose_path)) if pose_path.is_file() else None
            if pose_image is not None:
                pose_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            return str(cloud_path), str(cloud_path), pose_image, self.pose_summaries.get(frame_idx)

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
                use_sem=self.config['segmentation']['enabled'],
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
    with gr.Blocks(title='盛相实时重建') as demo:
        gr.Markdown('# 盛相实时重建\n开始后预览相机；拍摄时记录一帧灰度、深度、mask、点云和 ArUco 位姿。')
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
            optimize = gr.Button('全局图优化')
            stop = gr.Button('停止')
        status = gr.Textbox(label='状态', value='未启动', interactive=False)
        with gr.Row():
            gray = gr.Image(label='灰度图', interactive=False)
            depth = gr.Image(label='深度图（伪彩）', interactive=False)
            mask = gr.Image(label='有效深度 mask', interactive=False)
        with gr.Accordion('ArUco 位姿估计', open=True):
            with gr.Row():
                pose_image = gr.Image(label='pose_vis（当前选中帧）', interactive=False)
                pose_summary = gr.JSON(label='检测到的板子与码数、世界位姿状态')
        frame_selector = gr.Dropdown(label='选择单帧点云', choices=[], value=None)
        with gr.Row():
            single_cloud = gr.Model3D(label='单帧点云', display_mode='point_cloud', interactive=False)
            global_cloud = gr.Model3D(label='全局点云', display_mode='point_cloud', interactive=False)
        with gr.Row():
            single_file = gr.File(label='下载单帧点云', interactive=False)
            global_file = gr.File(label='下载全局点云', interactive=False)
        timer = gr.Timer(0.7)

        def start_from_ui(*values):
            settings = _camera_settings(*values[:len(camera_inputs)])
            software_settings = _software_settings(*values[len(camera_inputs):])
            already_running = controller.process is not None and controller.process.is_alive()
            message = controller.start(settings, software_settings)
            if already_running:
                return message, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
            return message, gr.update(choices=[], value=None), None, None, None, None, None, None

        def stop_from_ui():
            message = controller.stop()
            return message, controller.global_display_path, controller.global_display_path

        start.click(
            start_from_ui, inputs=camera_inputs + software_inputs,
            outputs=[status, frame_selector, single_cloud, single_file, global_cloud, global_file, pose_image, pose_summary],
            concurrency_limit=1,
        )
        capture.click(
            controller.capture,
            outputs=[status, frame_selector, single_cloud, single_file, global_cloud, global_file, pose_image, pose_summary],
            concurrency_limit=1,
        )
        frame_selector.input(
            controller.select_frame, inputs=frame_selector,
            outputs=[single_cloud, single_file, pose_image, pose_summary], concurrency_limit=1,
        )
        optimize.click(controller.optimize, inputs=graph_inputs, outputs=[status, global_cloud, global_file], concurrency_limit=1)
        stop.click(stop_from_ui, outputs=[status, global_cloud, global_file], concurrency_limit=1)
        timer.tick(controller.poll, outputs=[gray, depth, mask, status], concurrency_limit=1, show_progress='hidden')
    return demo


def main():
    parser = argparse.ArgumentParser(description='Shengxiang live reconstruction UI')
    parser.add_argument('--config', default='config/recon/shenxiang.yaml')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--exposure-3d', type=float, default=100.0, help='手动 3D 曝光强度，0.1–100，默认 100')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    controller = LiveReconstruction(args.config, args.device_index, args.exposure_3d)
    try:
        build_ui(controller).queue().launch(server_name='127.0.0.1', server_port=args.port)
    finally:
        controller.stop()


if __name__ == '__main__':
    main()
