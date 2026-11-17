"""RealSense D455 color capture with grayscale output and stream calibration."""
import cv2
import numpy as np


def modified_brown_maps(matrix, coefficients, width, height):
    """Ideal-to-source pixel map matching librealsense's Brown projection.

    The SDK uses this projection for modified/inverse Brown profiles; their
    tangential terms use radially distorted coordinates, unlike OpenCV Brown.
    """
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    x = (u - matrix[0, 2]) / matrix[0, 0]
    y = (v - matrix[1, 2]) / matrix[1, 1]
    k1, k2, p1, p2, k3 = coefficients
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x, y = x * radial, y * radial
    xd = x + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    return ((xd * matrix[0, 0] + matrix[0, 2]).astype(np.float32),
            (yd * matrix[1, 1] + matrix[1, 2]).astype(np.float32))


class RealSenseGrayCamera:
    def __init__(self, device_index=0):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError('D455 需要 pyrealsense2，请在当前 Python 环境安装：python -m pip install pyrealsense2') from exc
        self.pipeline = None
        devices = [device for device in rs.context().query_devices()
                   if 'D455' in device.get_info(rs.camera_info.name)]
        devices.sort(key=lambda device: device.get_info(rs.camera_info.serial_number))
        if not 0 <= device_index < len(devices):
            raise RuntimeError(f'未找到序号 {device_index} 的 D455（已连接 {len(devices)} 台，按序列号排序）')
        device = devices[device_index]
        serial = device.get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
            self.pipeline = pipeline
            sensor = profile.get_device().first_color_sensor()
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1)
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            matrix = np.array([[intrinsics.fx, 0., intrinsics.ppx],
                               [0., intrinsics.fy, intrinsics.ppy], [0., 0., 1.]])
            coefficients = np.asarray(intrinsics.coeffs, dtype=float)
            self.maps = None
            if intrinsics.model == rs.distortion.none or not np.any(coefficients):
                coefficients = np.zeros(5)
            elif intrinsics.model in (rs.distortion.inverse_brown_conrady,
                                       rs.distortion.modified_brown_conrady):
                self.maps = modified_brown_maps(matrix, coefficients, intrinsics.width, intrinsics.height)
                coefficients = np.zeros(5)
            elif intrinsics.model != rs.distortion.brown_conrady:
                raise RuntimeError(f'不支持此 D455 彩色流的畸变模型：{intrinsics.model}')
            self.detail = dict(
                camera_type='realsense_d455', device_index=device_index,
                name=device.get_info(rs.camera_info.name), serial_number=serial,
                capture='color BGR8 -> gray, 1280x720 @ 30 fps, auto exposure',
                source_distortion_model=str(intrinsics.model),
                source_distortion_coefficients=list(intrinsics.coeffs),
                rectified=self.maps is not None,
                calibration=dict(camera_matrix=matrix.tolist(), dist_coeffs=coefficients.tolist(),
                                 image_size=[intrinsics.width, intrinsics.height]),
            )
        except Exception:
            self.close()
            raise

    def capture(self, timeout_ms=5000):
        frame = self.pipeline.wait_for_frames(timeout_ms).get_color_frame()
        if not frame:
            raise RuntimeError('D455 未返回彩色帧')
        gray = cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_BGR2GRAY)
        if self.maps is not None:
            gray = cv2.remap(gray, *self.maps, interpolation=cv2.INTER_LINEAR)
        return gray

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
