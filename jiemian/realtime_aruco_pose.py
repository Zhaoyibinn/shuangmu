"""Gray-only, fixed-camera tracking of independently moving ArUco boards."""
import argparse
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Recon.Aruco_pose import Aruco_pose_Estimater
from Recon.config_loader import load_config


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def pose_storage_root(value, camera_type):
    folders = {'shenxiang': 'shengxiang_pose', 'realsense_d455': 'realsense_pose'}
    if camera_type not in folders:
        raise ValueError('未知相机类型')
    root = project_path(value)
    parts = list(root.parts)
    aliases = {'shengxiang', 'shenxiang', 'realsense', 'realsense_d455',
               'shengxiang_pose', 'shenxiang_pose', 'realsense_pose', 'realsense_d455_pose'}
    for index in range(len(parts) - 1, 0, -1):
        if parts[index] in aliases:
            parts[index] = folders[camera_type]
            return Path(*parts)
    return root / folders[camera_type]


def display_image(image, max_width=1332):
    """Resize only UI outputs; detection and disk images stay at native size."""
    if image is None or image.shape[1] <= max_width:
        return image
    height = max(1, round(image.shape[0] * max_width / image.shape[1]))
    return cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)


def rotation_euler_xyz_deg(rotation):
    """Fixed-axis XYZ angles in degrees, with R = Rz(z) @ Ry(y) @ Rx(x).

    At gimbal lock (Y = +/-90 degrees), choose Z = 0 deterministically.
    """
    rotation = np.asarray(rotation, dtype=float)
    cos_y = np.hypot(rotation[0, 0], rotation[1, 0])
    y = np.arctan2(-rotation[2, 0], cos_y)
    if cos_y > 1e-8:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        z = 0.0
    return np.degrees([x, y, z]).tolist()


def camera_worker(connection, settings):
    """Capture grayscale frames; isolate camera SDKs and USB teardown."""
    camera = None
    try:
        if settings.get('camera_type', 'shenxiang') == 'realsense_d455':
            from jiemian.cameras.realsense_gray_camera import RealSenseGrayCamera
            camera = RealSenseGrayCamera(settings['device_index'])
            detail = camera.detail
        else:
            from jiemian.cameras._gray_camera import GrayCamera
            camera = GrayCamera(settings['device_index'], settings['exposure_gray'], settings['gain'])
            detail = dict(camera_type='shenxiang', device_index=settings['device_index'], capture='Image2D callback')
        connection.send(('ready', detail))
        while True:
            command = connection.recv()
            if command == 'stop':
                break
            gray = camera.capture(timeout_ms=5000)
            connection.send(('frame', (time.time(), gray)))
    except Exception as exc:
        try:
            connection.send(('error', str(exc)))
        except (OSError, EOFError):
            pass
    finally:
        if camera is not None and settings.get('camera_type') == 'realsense_d455':
            try:
                camera.close()
            except Exception:
                pass
        connection.close()
        os._exit(0)


class BoardTracker:
    def __init__(self, estimator):
        self.estimator = estimator
        self.initial = {}
        self.initial_frames = {}

    def detect(self, gray, index):
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError('需要 uint8 单通道灰度图')
        estimator = self.estimator
        width, height = estimator.image_size
        if width <= 0 or height <= 0:
            raise ValueError('标定文件必须提供有效的图像宽高')
        actual_height, actual_width = gray.shape
        if abs((actual_width / actual_height) / (width / height) - 1) > 0.02:
            raise ValueError('图像与标定宽高比不一致，请提供全分辨率相机标定')
        # Scale intrinsics to native pixels; never resize the detection image.
        calibration_matrix = estimator.camera_matrix.copy()
        estimator.camera_matrix = np.diag([actual_width / width, actual_height / height, 1.0]) @ calibration_matrix
        try:
            corners, ids, poses = estimator.estimate_marker_poses(gray)
            visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(visual, corners, ids)
            boards = {}
            for board_id, pose in sorted(poses.items()):
                transform = np.eye(4)
                transform[:3, :3] = cv2.Rodrigues(pose['rvec_obj_to_cam'])[0]
                transform[:3, 3] = pose['tvec_obj_to_cam'].reshape(3)
                if not np.all(np.isfinite(transform)) or transform[2, 3] <= 0:
                    continue
                if board_id not in self.initial:
                    self.initial[board_id] = transform.copy()
                    self.initial_frames[board_id] = index
                relative = np.linalg.inv(self.initial[board_id]) @ transform
                rotation_vector = cv2.Rodrigues(relative[:3, :3])[0].reshape(3)
                boards[str(board_id)] = {
                    'initial_frame': self.initial_frames[board_id],
                    'marker_ids': pose['visible_board_marker_ids'],
                    'reprojection_error_px': pose['reprojection_error_mean'],
                    'T_camera_board': transform.tolist(),
                    'T_initial_board': relative.tolist(),
                    'translation_mm': relative[:3, 3].tolist(),
                    'rotation_vector_deg': np.degrees(rotation_vector).tolist(),
                    'rotation_euler_xyz_deg': rotation_euler_xyz_deg(relative[:3, :3]),
                    'rotation_angle_deg': float(np.degrees(np.linalg.norm(rotation_vector))),
                }
                cv2.drawFrameAxes(visual, estimator.camera_matrix, estimator.dist_coeffs,
                                  pose['rvec_obj_to_cam'], pose['tvec_obj_to_cam'], estimator.aruco_length * 2)
                location = tuple(np.rint(pose['image_points'].mean(axis=0)).astype(int))
                cv2.putText(visual, f'Board {board_id}', location, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return cv2.cvtColor(visual, cv2.COLOR_BGR2RGB), boards
        finally:
            estimator.camera_matrix = calibration_matrix


def pose_plot(records, index, board_id=None, quantity='rotation'):
    """Plot one board's three relative components, leaving missed frames blank."""
    figure = go.Figure()
    board_ids = sorted({key for record in records for key in record['boards']})
    if board_id not in board_ids:
        board_id = board_ids[0] if board_ids else None
    frames = [record['index'] for record in records]
    values = []
    for record in records:
        pose = record['boards'].get(board_id)
        if pose is None:
            values.append([None] * 3)
        elif quantity == 'translation':
            values.append(pose['translation_mm'])
        else:
            angles = pose.get('rotation_euler_xyz_deg')
            if angles is None:
                # Older sessions have the relative matrix but no Euler angles.
                angles = rotation_euler_xyz_deg(np.asarray(pose['T_initial_board'])[:3, :3])
            values.append(angles)
    for axis, color in enumerate(('#e74c3c', '#27ae60', '#2980b9')):
        figure.add_trace(go.Scatter(
            x=frames, y=[value[axis] for value in values], mode='lines+markers',
            name=('绕 ' if quantity == 'rotation' else 'Δ') + 'XYZ'[axis],
            line=dict(color=color), marker=dict(size=4), connectgaps=False))
    if records:
        index = max(0, min(index, len(records) - 1))
        figure.add_vline(x=frames[index], line_dash='dash', line_color='#555555')
    figure.update_layout(
        xaxis_title='原始帧序号',
        yaxis_title='角度 / °' if quantity == 'rotation' else '平移 / mm',
        hovermode='x unified', uirevision=f'aruco-{quantity}-{board_id}',
        margin=dict(l=55, r=20, t=25, b=45))
    return figure


class PoseController:
    def __init__(self, config_path, output=None):
        self.config = load_config(project_path(config_path))
        self.output = project_path(output or self.config['paths']['save_dir'])
        self.context = mp.get_context('spawn')
        self.lock = threading.RLock()
        self.process = self.connection = None
        self.tracker = None
        self.records = []
        self.session = None
        self.preview = None
        self._display_key = None
        self._display_image = None
        self.continuous = self.playing = False
        self.replay_index = 0
        self.plot_board_id = None
        self.play_time = self.play_record_time = 0.0
        self.status = '请启动相机并拍摄起始图像。'

    def _create_session(self, camera_type, metadata):
        session = pose_storage_root(self.output, camera_type) / datetime.now().strftime('pose_%Y%m%d_%H%M%S_%f')
        (session / 'gray').mkdir(parents=True)
        (session / 'visual').mkdir()
        (session / 'session.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        return session

    def _receive(self, timeout, action):
        try:
            if not self.connection.poll(timeout):
                raise RuntimeError(f'{action}超时')
            return self.connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            self.process.join(timeout=0.5)
            code = self.process.exitcode
            raise RuntimeError(f'{action}时相机子进程异常退出（退出码 {code}）；请检查终端中的 SDK 错误。') from exc

    def _stop_camera(self):
        self.continuous = False
        if self.process is not None:
            try:
                self.connection.send('stop')
            except (OSError, EOFError):
                pass
            if self.process.pid is not None:
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
            self.connection.close()
        self.process = self.connection = None

    def stop(self):
        with self.lock:
            self._stop_camera()
            self.status = '相机已停止；记录已保存，可以回放。'
            return self.render()

    def start(self, device, exposure, gain, camera_type='shenxiang'):
        with self.lock:
            try:
                if camera_type not in ('shenxiang', 'realsense_d455'):
                    raise ValueError('未知相机类型')
                if int(device) != float(device) or int(device) < 0:
                    raise ValueError('设备序号须为 ≥ 0 的整数')
                settings = dict(camera_type=camera_type, device_index=int(device))
                if camera_type == 'shenxiang':
                    settings.update(exposure_gray=float(exposure), gain=int(gain))
                    if not 0.1 <= settings['exposure_gray'] <= 100 or not 0 <= settings['gain'] <= 5:
                        raise ValueError('灰度曝光须在 0.1–100，增益须在 0–5')
                self._stop_camera()
                paths = self.config['paths']
                if camera_type == 'shenxiang':
                    estimator = Aruco_pose_Estimater.from_yaml(str(project_path(paths['int_yaml_path'])),
                        self.config['reconstruction']['aruco_length'],
                        board_yaml_path=str(project_path(paths['aruco_board_yaml_path'])))
                    # Build a workspace-local adapter, leaving the installed SDK unchanged.
                    import importlib.util
                    if importlib.util.find_spec('jiemian.cameras._gray_camera') is None:
                        from jiemian.cameras.build_gray_camera import build
                        build()
                self.connection, child = self.context.Pipe()
                self.process = self.context.Process(target=camera_worker, args=(child, settings), daemon=True)
                self.process.start()
                child.close()
                kind, detail = self._receive(20, '启动')
                if kind != 'ready':
                    raise RuntimeError(str(detail))
                if camera_type == 'realsense_d455':
                    calibration = detail['calibration']
                    length, layouts = Aruco_pose_Estimater.load_board_config(str(project_path(paths['aruco_board_yaml_path'])))
                    estimator = Aruco_pose_Estimater(
                        calibration['camera_matrix'], calibration['dist_coeffs'],
                        length if length is not None else self.config['reconstruction']['aruco_length'],
                        image_size=tuple(calibration['image_size']), board_marker_layouts=layouts)
                self.tracker = BoardTracker(estimator)
                self.records = []
                self.preview = None
                self._display_key = self._display_image = None
                self.replay_index = 0
                self.playing = False
                metadata = dict(settings=settings, config=self.config, device=detail,
                    convention='T_camera_board: board to fixed camera; T_initial_board = inverse(T_camera_board_initial) @ T_camera_board; mm',
                    reference_policy='Each board uses its first valid observation, including boards absent in the starting image.')
                self.session = self._create_session(camera_type, metadata)
                self.status = '相机已启动，实时预览中，尚未保存起始图。请点击“拍摄起始图像”建立参考位姿。'
            except Exception as exc:
                self._stop_camera()
                self.status = f'启动失败：{str(exc) or type(exc).__name__}'
            return self.render()

    def _capture(self, mode):
        if mode != 'initial' and not self.records:
            raise RuntimeError('请先点击“拍摄起始图像”')
        timestamp, gray = self._read_gray()
        index = len(self.records)
        visual, boards = self.tracker.detect(gray, index)
        gray_name = f'gray/frame_{index:06d}.png'
        visual_name = f'visual/frame_{index:06d}.png'
        if not cv2.imwrite(str(self.session / gray_name), gray) or not cv2.imwrite(str(self.session / visual_name), cv2.cvtColor(visual, cv2.COLOR_RGB2BGR)):
            raise RuntimeError('图像保存失败，请检查磁盘空间')
        record = dict(index=index, timestamp=timestamp, elapsed_s=0.0 if not self.records else timestamp - self.records[0]['timestamp'],
                      mode=mode, gray=gray_name, visual=visual_name, boards=boards)
        with (self.session / 'poses.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.records.append(record)
        self.replay_index = index
        self._display_key = str(self.session / visual_name)
        self._display_image = display_image(visual)
        self.status = f'已保存第 {index} 帧，检测到 {len(boards)} 块板。' + ('未检测到的板子不更新位姿。' if not boards else '')

    def _read_gray(self):
        if self.process is None or not self.process.is_alive():
            raise RuntimeError('请先启动相机')
        try:
            self.connection.send('capture')
            kind, payload = self._receive(8, '拍摄')
        except (OSError, RuntimeError):
            self._stop_camera()
            raise
        if kind != 'frame':
            self._stop_camera()
            raise RuntimeError(str(payload))
        timestamp, gray = payload
        self.preview = display_image(gray)
        return timestamp, gray

    def capture_initial(self):
        with self.lock:
            self.continuous = self.playing = False
            try:
                if self.records:
                    raise RuntimeError('本会话已有起始图像；请重新启动相机创建新会话后再拍摄起始图像')
                self._capture('initial')
            except Exception as exc:
                self.status = f'起始图像拍摄失败：{str(exc) or type(exc).__name__}'
            return self.render()

    def capture(self):
        with self.lock:
            self.playing = False
            try:
                self._capture('single')
            except Exception as exc:
                self.status = f'检测失败：{str(exc) or type(exc).__name__}'
            return self.render()

    def continuous_mode(self, enabled):
        with self.lock:
            if enabled and (self.process is None or not self.records):
                self.continuous = False
                self.status = '请先启动相机并点击“拍摄起始图像”，再启用连续检测。'
                return self.render()
            self.continuous = bool(enabled) and self.process is not None
            self.playing = False
            self.status = '连续检测中。' if self.continuous else '连续检测已暂停，可以单帧检测或回放。'
            return self.render()

    def seek(self, index):
        with self.lock:
            self.continuous = self.playing = False
            self.replay_index = max(0, min(int(index), len(self.records) - 1))
            self.status = f'回放第 {self.replay_index} 帧。'
            return self.render()

    def playback(self, enabled):
        with self.lock:
            self.continuous = False
            self.playing = bool(enabled) and bool(self.records)
            if self.playing:
                if self.replay_index >= len(self.records) - 1:
                    self.replay_index = 0
                self.play_time = time.monotonic()
                self.play_record_time = self.records[self.replay_index]['elapsed_s']
            self.status = '按采集时间回放。' if self.playing else '回放已暂停。'
            return self.render()

    def load_session(self, value):
        with self.lock:
            try:
                folder = project_path(value).resolve()
                records = [json.loads(line) for line in (folder / 'poses.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                if not records:
                    raise ValueError('会话没有记录')
                for record in records:
                    for key in ('gray', 'visual'):
                        path = (folder / record[key]).resolve()
                        if not path.is_relative_to(folder) or not path.is_file():
                            raise ValueError('记录图像缺失或路径越界')
                    if not np.isfinite(record['elapsed_s']):
                        raise ValueError('记录时间无效')
                if any(a['elapsed_s'] > b['elapsed_s'] for a, b in zip(records, records[1:])):
                    raise ValueError('记录时间顺序无效')
                self._stop_camera()
                self.playing = False
                self.session, self.records, self.replay_index = folder, records, 0
                self.status = f'已载入 {len(records)} 帧。'
            except Exception as exc:
                self.status = f'载入失败：{exc}'
            return self.render()

    def clear_history(self):
        with self.lock:
            try:
                # Start a fresh recording directory; never overwrite old frames.
                session = None
                if self.process is not None:
                    metadata = json.loads((self.session / 'session.json').read_text(encoding='utf-8'))
                    session = self._create_session(metadata['settings']['camera_type'], metadata)
                self.continuous = self.playing = False
                self.records = []
                self.replay_index = 0
                self._display_key = self._display_image = None
                self.session = session
                if self.tracker is not None:
                    self.tracker.initial.clear()
                    self.tracker.initial_frames.clear()
                self.status = '历史记录与起始位姿已清空，已保存文件保留。' + (
                    '实时预览继续，请重新拍摄起始图像。' if self.process is not None else '请启动相机开始新会话。')
            except Exception as exc:
                self.status = f'清空历史失败：{str(exc) or type(exc).__name__}'
            return self.render()

    def tick(self):
        with self.lock:
            if not self.continuous and self.process is not None:
                try:
                    self._read_gray()
                except Exception as exc:
                    self._stop_camera()
                    self.status = f'实时预览已停止：{str(exc) or type(exc).__name__}'
                    return self.render()
            if self.continuous:
                try:
                    self._capture('continuous')
                except Exception as exc:
                    self.continuous = False
                    self.status = f'连续检测已暂停：{exc}'
            elif self.playing:
                elapsed = self.play_record_time + time.monotonic() - self.play_time
                while self.replay_index + 1 < len(self.records) and self.records[self.replay_index + 1]['elapsed_s'] <= elapsed:
                    self.replay_index += 1
                self.status = f'回放第 {self.replay_index} 帧。'
                if self.replay_index == len(self.records) - 1:
                    self.playing = False
            else:
                return (*tuple(gr.skip() for _ in range(8)), self.preview if self.process is not None else gr.skip(),
                        gr.skip(), gr.skip())
            return self.render()

    def set_plot_board(self, board_id):
        with self.lock:
            self.plot_board_id = board_id
            return self.render()

    def render(self):
        plot_board_ids = sorted({key for item in self.records for key in item['boards']})
        if self.plot_board_id not in plot_board_ids:
            self.plot_board_id = plot_board_ids[0] if plot_board_ids else None
        index = self.replay_index
        record = self.records[index] if self.records else None
        image = None
        if record:
            key = str(self.session / record['visual'])
            if key != self._display_key:
                stored = cv2.imread(key)
                self._display_image = (cv2.cvtColor(display_image(stored), cv2.COLOR_BGR2RGB)
                                       if stored is not None else None)
                self._display_key = key
            image = self._display_image
        rows = []
        board_ids = self.tracker.estimator.board_marker_layouts if self.tracker and self.process is not None else sorted({key for item in self.records for key in item['boards']})
        for board_id in board_ids:
            pose = record['boards'].get(str(board_id)) if record else None
            if pose:
                euler_angles = pose.get('rotation_euler_xyz_deg')
                if euler_angles is None:
                    # Older sessions store the relative matrix but no Euler angles.
                    euler_angles = rotation_euler_xyz_deg(np.asarray(pose['T_initial_board'])[:3, :3])
                rows.append([str(board_id), '已检测', pose['initial_frame'], len(pose['marker_ids']),
                             *pose['translation_mm'], *euler_angles, pose['rotation_angle_deg'], pose['reprojection_error_px']])
            else:
                rows.append([str(board_id), '本帧未检测', None, 0, *([None] * 8)])
        return (image, rows, record or {}, pose_plot(self.records, index, self.plot_board_id).to_json(),
                gr.update(value=index, maximum=max(1, len(self.records) - 1)),
                self.status + (f'\n会话：{self.session}' if self.session else ''), self.continuous, self.playing, self.preview,
                pose_plot(self.records, index, self.plot_board_id, 'translation').to_json(),
                gr.update(choices=plot_board_ids, value=self.plot_board_id))


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

    with gr.Blocks(title='ArUco 实时位姿估计') as ui:
        gr.Markdown('## ArUco 实时位姿估计\n固定相机、移动板子；纯灰度、全分辨率。每块板独立以首次有效观测为起点。平移单位 mm，各轴欧拉角和总体旋转角单位 °。')
        with gr.Row():
            camera_type = gr.Dropdown(label='相机类型', choices=[('盛相相机', 'shenxiang'), ('RealSense D455', 'realsense_d455')], value='shenxiang')
            device = gr.Number(label='设备序号', value=0, precision=0)
            exposure = gr.Number(label='灰度曝光（0.1–100）', value=100)
            gain = gr.Number(label='增益（0–5）', value=0, precision=0)
        with gr.Row():
            start = gr.Button('启动 / 切换相机', variant='primary')
            initial = gr.Button('拍摄起始图像', variant='primary')
            capture = gr.Button('单帧检测')
            continuous = gr.Checkbox(label='连续视频检测', value=False)
            stop = gr.Button('停止相机')
        status = gr.Textbox(label='状态 / 保存目录', interactive=False)
        gr.Markdown('选择相机类型及设备序号后，点击“启动 / 切换相机”。切换会创建新会话，需重新拍摄起始图像，已保存文件保留。D455 使用 1280×720 彩色流转灰度、设备自带标定和自动曝光；设备序号按 D455 序列号排序。')
        preview = gr.Image(label='实时灰度预览（不保存、不建立参考位姿）', type='numpy', image_mode='L')
        image = gr.Image(label='灰度检测与各板坐标轴', type='numpy')
        plot_board = gr.Dropdown(label='位姿曲线：板 ID', choices=[], value=None)
        with gr.Row():
            plot = persistent_plot(pose_plot([], 0), '历史旋转角：X / Y / Z（°）', 'aruco-rotation-history')
            translation_plot = persistent_plot(pose_plot([], 0, quantity='translation'),
                                               '历史平移：X / Y / Z（mm）', 'aruco-translation-history')
        table = gr.Dataframe(headers=['板 ID', '状态', '起始帧', '码数', 'ΔX', 'ΔY', 'ΔZ', '绕 X 角度 (°)', '绕 Y 角度 (°)', '绕 Z 角度 (°)', '总体旋转角 (°)', '重投影误差 px'], interactive=False)
        gr.Markdown('平移和旋转均表示当前板子相对自身起始坐标系的变化。三轴角度采用固定轴 XYZ 欧拉角：依次绕起始 X、Y、Z 轴旋转，R = Rz · Ry · Rx；X/Z 范围 ±180°，Y 范围 ±90°，Y 为 ±90° 时约定 Z = 0°。总体旋转角是相对起始姿态的最小等效旋转角（0–180°），不是三轴角度之和或累计转角。两张折线图分别显示所选板的三轴欧拉角和平移，虚线标记当前回放帧；漏检帧留空。')
        with gr.Row():
            slider = gr.Slider(label='回放帧（0 为起始图像）', minimum=0, maximum=1, step=1, value=0)
            playing = gr.Checkbox(label='自动回放（按原始时间）', value=False)
        with gr.Row():
            folder = gr.Textbox(label='历史会话目录', placeholder='depth_outputs/shengxiang_pose/continuous/pose_...')
            load = gr.Button('载入历史会话')
        with gr.Accordion('本帧完整位姿与变换矩阵', open=False):
            detail = gr.JSON()
        clear = gr.Button('清空历史')
        outputs = [image, table, detail, plot, slider, status, continuous, playing, preview, translation_plot, plot_board]
        events = dict(outputs=outputs, concurrency_id='pose-controller', concurrency_limit=1)
        camera_type.change(lambda value: (gr.update(visible=value == 'shenxiang'), gr.update(visible=value == 'shenxiang')),
                           inputs=[camera_type], outputs=[exposure, gain])
        plot_board.input(controller.set_plot_board, inputs=[plot_board], **events)
        start.click(controller.start, inputs=[device, exposure, gain, camera_type], **events)
        initial.click(controller.capture_initial, **events)
        capture.click(controller.capture, **events)
        stop.click(controller.stop, **events)
        continuous.input(controller.continuous_mode, inputs=[continuous], **events)
        slider.release(controller.seek, inputs=[slider], **events)
        playing.input(controller.playback, inputs=[playing], **events)
        load.click(controller.load_session, inputs=[folder], **events)
        clear.click(controller.clear_history, **events)
        gr.Timer(0.2).tick(controller.tick, **events)
    return ui


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/recon/shenxiang.yaml')
    parser.add_argument('--output', help='保存根目录，自动按 shengxiang_pose / realsense_pose 分目录')
    parser.add_argument('--port', type=int, default=7861)
    args = parser.parse_args()
    controller = PoseController(args.config, args.output)
    try:
        build_ui(controller).queue().launch(server_name='127.0.0.1', server_port=args.port,
                                           allowed_paths=[str(pose_storage_root(controller.output, camera).resolve())
                                                          for camera in ('shenxiang', 'realsense_d455')])
    finally:
        controller.stop()


if __name__ == '__main__':
    main()
