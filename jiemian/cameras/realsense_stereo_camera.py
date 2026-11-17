"""D455 infrared stereo capture, rectification and FoundationStereo depth."""
from pathlib import Path
import gc
import sys
from time import perf_counter

from Recon.timing import StageTimer

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SIZE = (1280, 720)
INTRINSICS_PATH = ROOT / 'biaoding/intrinsics_d455_720p_20260526.yml'
EXTRINSICS_PATH = ROOT / 'biaoding/extrinsics_d455_720p_20260526.yml'
DEFAULT_CHECKPOINT = 'jiegouguang/weights/FoundationStereo/23-51-11/model_best_bp2.pth'
DEFAULT_FAST_CHECKPOINT = 'jiegouguang/weights/fast_FoundationStereo/23-36-37/model_best_bp2_serialize.pth'
STEREO_METHODS = ('foundation_stereo', 'fast_foundation_stereo')


def default_checkpoint(method):
    if method not in STEREO_METHODS:
        raise ValueError(f'不支持的双目模式：{method}')
    return DEFAULT_FAST_CHECKPOINT if method == 'fast_foundation_stereo' else DEFAULT_CHECKPOINT



def load_stereo_calibration():
    """Load the 1280x720 IR pair calibration; file T is in millimetres."""
    values = {}
    for path, names in ((INTRINSICS_PATH, ('M1', 'D1', 'M2', 'D2')),
                        (EXTRINSICS_PATH, ('R', 'T'))):
        if not path.is_file():
            raise FileNotFoundError(f'D455 标定文件不存在：{path}')
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            for name in names:
                value = storage.getNode(name).mat()
                if value is None or not np.isfinite(value).all():
                    raise ValueError(f'{path} 缺少有效的 {name}')
                values[name] = np.asarray(value, dtype=np.float64)
        finally:
            storage.release()
    for name in ('M1', 'M2', 'R'):
        if values[name].shape != (3, 3):
            raise ValueError(f'D455 标定 {name} 必须为 3×3 矩阵')
    for name in ('D1', 'D2'):
        if values[name].size not in (4, 5, 8, 12, 14):
            raise ValueError(f'D455 标定 {name} 畸变系数数量无效')
    if values['T'].size != 3:
        raise ValueError('D455 标定 T 必须有 3 个分量')
    return [(values['M1'], values['D1']), (values['M2'], values['D2'])], values['R'], values['T'].reshape(3) / 1000.


def stereo_geometry(left, right, rotation, translation_m, image_size, return_rotation=False):
    """Build maps and a common pinhole frame; translations must be metres."""
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        left[0], left[1], right[0], right[1], image_size, rotation,
        np.asarray(translation_m, dtype=np.float64).reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    baseline = -float(p2[0, 3] / p2[0, 0])
    if not np.isfinite(baseline) or baseline <= 0 or abs(p2[1, 3]) > 1e-6:
        raise ValueError('D455 左右红外流必须形成正基线的水平双目')
    maps = [cv2.initUndistortRectifyMap(k, d, r, p[:3, :3], image_size, cv2.CV_32FC1)
            for (k, d), r, p in ((left, r1, p1), (right, r2, p2))]
    result = (maps, p1[:3, :3], baseline)
    return (*result, r1) if return_rotation else result


def depth_in_mm(result):
    depth = np.asarray(result['depth'], dtype=np.float32) * 1000.0
    valid = np.asarray(result['valid_mask'], dtype=bool) & np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0).astype(np.float32)
    # Match the reconstruction worker's convention: zero means valid.
    return depth, np.where(valid, 0, 1).astype(np.uint8)


def infer_scaled(model, left, right, matrix, baseline_m, scale, **kwargs):
    """Reduce network input while returning depth in the full rectified geometry."""
    if scale == 1.:
        return model.infer(left, right, matrix, baseline_m, **kwargs)
    height, width = left.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    sx, sy = size[0] / width, size[1] / height
    scaled_matrix = matrix.copy()
    scaled_matrix[0] *= sx
    scaled_matrix[1] *= sy
    scaled_matrix[0, 2] += (sx - 1.) * .5
    scaled_matrix[1, 2] += (sy - 1.) * .5
    result = model.infer(cv2.resize(left, size, interpolation=cv2.INTER_AREA),
                         cv2.resize(right, size, interpolation=cv2.INTER_AREA),
                         scaled_matrix, baseline_m, **kwargs)
    disparity = np.asarray(result['disparity'], dtype=np.float32)
    valid = np.asarray(result['valid_mask'], dtype=bool) & np.isfinite(disparity) & (disparity > 0)
    # Keep invalid samples invalid rather than filling depth holes by interpolation.
    disparity = cv2.resize(np.where(valid, disparity, 0), (width, height),
                           interpolation=cv2.INTER_NEAREST) / sx
    valid = cv2.resize(valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    depth = np.zeros_like(disparity)
    depth[valid] = float(matrix[0, 0]) * baseline_m / disparity[valid]
    return dict(disparity=disparity, depth=depth, valid_mask=valid)


def configure_emitter(sensor, option, enabled):
    """Set the projector explicitly and record the state reported by the device."""
    if not sensor.supports(option):
        if enabled:
            raise RuntimeError('当前设备不支持控制红外散斑投射器')
        return None
    requested = int(bool(enabled))
    sensor.set_option(option, requested)
    actual = sensor.get_option(option)
    if not np.isclose(actual, requested):
        raise RuntimeError(f'散斑投射器状态设置失败：请求 {requested}，设备返回 {actual}')
    return bool(requested)


class RealSenseStereoCamera:
    def __init__(self, device_index=0, checkpoint=None, valid_iters=None, foreground_settings=None, stereo_method='foundation_stereo', inference_scale=1., emitter_enabled=False):
        default_checkpoint(stereo_method)  # Validate before opening the device.
        self.inference_scale = float(inference_scale)
        if not 0 < self.inference_scale <= 1:
            raise ValueError('双目推理缩放必须在 (0, 1] 内')
        self.stereo_method = stereo_method
        width, height = CAPTURE_SIZE
        self.pipeline = None
        calibration, rotation, translation_m = load_stereo_calibration()
        from jiemian.processing.foreground import ForegroundSegmenter
        foreground_settings = foreground_settings or {}
        self.segmenter = ForegroundSegmenter(**{
            key: foreground_settings[key] for key in ('weights', 'conf', 'imgsz', 'device')
            if key in foreground_settings})
        self.valid_iters = int(valid_iters if valid_iters is not None else (8 if stereo_method == 'fast_foundation_stereo' else 32))
        if self.valid_iters < 1:
            raise ValueError('FoundationStereo 迭代次数必须 ≥ 1')
        checkpoint = Path(checkpoint or default_checkpoint(stereo_method))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f'{stereo_method} 权重不存在：{checkpoint}')
        if stereo_method == 'foundation_stereo' and not checkpoint.with_name('cfg.yaml').is_file():
            raise FileNotFoundError(f'FoundationStereo 权重同目录缺少 cfg.yaml：{checkpoint}')
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError('D455 需要在当前环境安装 pyrealsense2') from exc
        # Load only in the child process, leaving the Shengxiang backend independent.
        sys.path.insert(0, str(ROOT / 'jiegouguang'))
        if stereo_method == 'fast_foundation_stereo':
            from fast_FoundationStereo.fast_stereo_inference import FastStereoInference
            self.model = FastStereoInference(str(checkpoint), valid_iters=self.valid_iters)
        else:
            from FoundationStereo.stereo_inference import StereoInference
            self.model = StereoInference(str(checkpoint))
            self.model.args.low_memory = True
        devices = [d for d in rs.context().query_devices()
                   if 'D455' in d.get_info(rs.camera_info.name)]
        devices.sort(key=lambda d: d.get_info(rs.camera_info.serial_number))
        if not 0 <= device_index < len(devices):
            raise RuntimeError(f'未找到序号 {device_index} 的 D455（已连接 {len(devices)} 台）')
        device = devices[device_index]
        serial = device.get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(serial)
        for index in (1, 2):
            config.enable_stream(rs.stream.infrared, index, width, height, rs.format.y8, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
            self.pipeline = pipeline
            sensor = profile.get_device().first_depth_sensor()
            emitter_state = configure_emitter(sensor, rs.option.emitter_enabled, emitter_enabled)
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1)
            streams = [profile.get_stream(rs.stream.infrared, i).as_video_stream_profile() for i in (1, 2)]
            for stream in streams:
                intr = stream.get_intrinsics()
                if (intr.width, intr.height) != CAPTURE_SIZE:
                    raise ValueError('D455 标定文件要求 1280×720 红外流')
            self.maps, self.matrix, self.baseline_m, self.rectification_rotation = stereo_geometry(
                *calibration, rotation, translation_m, CAPTURE_SIZE, return_rotation=True)
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intr = color_stream.get_intrinsics()
            self.rgb_k = np.array([[color_intr.fx, 0., color_intr.ppx],
                                   [0., color_intr.fy, color_intr.ppy], [0., 0., 1.]])
            self.rgb_size = (color_intr.width, color_intr.height)
            coeffs = np.asarray(color_intr.coeffs)
            self.rgb_maps = None
            if np.any(coeffs):
                if color_intr.model in (rs.distortion.modified_brown_conrady, rs.distortion.inverse_brown_conrady):
                    from jiemian.cameras.realsense_gray_camera import modified_brown_maps
                    self.rgb_maps = modified_brown_maps(self.rgb_k, coeffs, *self.rgb_size)
                elif color_intr.model == rs.distortion.brown_conrady:
                    self.rgb_maps = cv2.initUndistortRectifyMap(self.rgb_k, coeffs, None,
                        self.rgb_k, self.rgb_size, cv2.CV_32FC1)
                elif color_intr.model != rs.distortion.none:
                    raise RuntimeError(f'不支持此 RGB 畸变模型：{color_intr.model}')
            rgb_extr = streams[0].get_extrinsics_to(color_stream)
            self.ir_to_rgb_rotation = np.asarray(rgb_extr.rotation).reshape(3, 3, order='F')
            self.ir_to_rgb_translation = np.asarray(rgb_extr.translation)
            self.detail = dict(
                camera_type='realsense_d455', name=device.get_info(rs.camera_info.name),
                serial_number=serial, depth_source=stereo_method, checkpoint=str(checkpoint),
                valid_iters=self.valid_iters, baseline_m=self.baseline_m,
                inference_scale=self.inference_scale,
                inference_size=[round(width * self.inference_scale), round(height * self.inference_scale)],
                emitter_requested=bool(emitter_enabled), emitter_enabled=emitter_state,
                capture=('infrared 1/2 Y8 + color BGR8, 1280x720 @ 30 fps, emitter '
                         + ('unsupported' if emitter_state is None else 'on' if emitter_state else 'off')),
                requested_resolution='1280x720',
                foreground=foreground_settings, depth_coordinate_system='undistorted_rgb',
                stereo_camera_matrix=self.matrix.tolist(), rectification_rotation=self.rectification_rotation.tolist(),
                ir_to_rgb_rotation=self.ir_to_rgb_rotation.tolist(),
                ir_to_rgb_translation_m=self.ir_to_rgb_translation.tolist(),
                rgb_source_distortion_coefficients=list(color_intr.coeffs),
                rgb_source_distortion_model=str(color_intr.model),
                source_intrinsics=[dict(matrix=k.tolist(), coefficients=d.tolist()) for k, d in calibration],
                source_rotation=rotation.tolist(), source_translation_m=translation_m.tolist(),
                stereo_calibration_source=dict(intrinsics=str(INTRINSICS_PATH), extrinsics=str(EXTRINSICS_PATH),
                                               translation_file_unit='mm', rectification='recomputed from M1/D1/M2/D2/R/T'),
                rgb_calibration_source='device',
                calibration=dict(camera_matrix=self.rgb_k.tolist(), dist_coeffs=[0.] * 5, image_size=list(self.rgb_size)))
        except Exception:
            self.close()
            raise

    def capture(self):
        import torch
        synchronize = (lambda: torch.cuda.synchronize(self.model.device)) if self.model.device.type == 'cuda' else None
        timer = StageTimer(synchronize)
        frames = self.pipeline.wait_for_frames(5000)
        timer.mark('wait_for_frames')
        images = [frames.get_infrared_frame(i) for i in (1, 2)]
        if not all(images):
            raise RuntimeError('D455 未返回完整左右红外帧')
        left, right = [cv2.remap(np.asanyarray(f.get_data()), *maps, cv2.INTER_LINEAR)
                       for f, maps in zip(images, self.maps)]
        timer.mark('ir_rectify')
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError('D455 未返回 RGB 帧')
        rgb = np.asanyarray(color.get_data()).copy()  # OpenCV BGR convention
        if self.rgb_maps is not None:
            rgb = cv2.remap(rgb, *self.rgb_maps, cv2.INTER_LINEAR)
        timer.mark('rgb_copy_undistort')
        foreground = self.segmenter.segment(rgb) if self.segmenter is not None else np.ones(rgb.shape[:2], dtype=bool)
        timer.mark('foreground_segment')
        stereo_timings = {}
        result = infer_scaled(self.model, left, right, self.matrix, self.baseline_m,
                              self.inference_scale, valid_iters=self.valid_iters, timings=stereo_timings)
        timer.mark(self.stereo_method)
        stereo_depth, _ = depth_in_mm(result)
        timer.mark('depth_to_mm')
        from jiegouguang.color import project_depth_to_rgb
        from jiemian.processing.foreground import mask_depth
        # Rectified left -> original left -> RGB. Depth is mm; SDK translation is m.
        aligned = project_depth_to_rgb(stereo_depth, self.matrix,
            self.ir_to_rgb_rotation @ self.rectification_rotation.T,
            self.ir_to_rgb_translation * 1000., self.rgb_k, self.rgb_size)
        timer.mark('project_depth_to_rgb')
        depth = mask_depth(aligned, foreground)
        mask = np.where(depth > 0, 0, 1).astype(np.uint8)
        extra = dict(rgb=rgb, foreground=foreground.astype(np.uint8) * 255,
                     aligned_depth_mm=aligned, stereo_depth_mm=stereo_depth, left=left)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        timer.mark('mask_pack')
        extra['timings_ms'] = timer.finish()
        extra['stereo_timings_ms'] = stereo_timings
        extra['produced_at'] = perf_counter()
        return aligned, gray, mask, right, extra

    def close(self):
        pipeline = self.pipeline
        try:
            if pipeline is not None:
                pipeline.stop()
        finally:
            self.pipeline = None
            self.model = self.segmenter = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
