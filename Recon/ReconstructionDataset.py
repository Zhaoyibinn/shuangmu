import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        data_root_path,
        ext_yaml_path,
        int_yaml_path,
        color_ext_yaml_path=None,
        min_depth_mm=100,
        max_depth_mm=4000,
        input_mode='stereo',
    ):
        self.data_root_path = data_root_path
        self.ext_yaml_path = ext_yaml_path
        self.int_yaml_path = int_yaml_path
        self.color_ext_yaml_path = color_ext_yaml_path
        self.min_depth_mm = float(min_depth_mm)
        self.max_depth_mm = float(max_depth_mm)
        self.input_mode = str(input_mode).lower()

        if self.input_mode == 'direct_depth':
            self._initialize_direct_depth(self.int_yaml_path)
            return
        if self.input_mode != 'stereo':
            raise ValueError('unsupported input mode: {}'.format(input_mode))

        self.left_images = self._load_media_sequence(os.path.join(data_root_path, 'left'))
        self.right_images = self._load_media_sequence(os.path.join(data_root_path, 'right'))

        rgb_path = os.path.join(data_root_path, 'color')
        self.rgb_images = None
        if os.path.exists(rgb_path):
            self.rgb_images = self._load_media_sequence(rgb_path, color=True)

    @staticmethod
    def _indexed_images(path):
        path = Path(path)
        if not path.is_dir():
            raise ValueError('missing image directory: {}'.format(path))
        supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        images = {
            item.stem: item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in supported_exts
        }
        if not images:
            raise ValueError('directory contains no supported images: {}'.format(path))
        return images

    def _initialize_direct_depth(self, int_yaml_path):
        if not int_yaml_path:
            raise ValueError('int_yaml_path is required for direct_depth input')

        depth_images = self._indexed_images(Path(self.data_root_path) / 'depth')
        gray_images = self._indexed_images(Path(self.data_root_path) / 'gray')
        mask_images = self._indexed_images(Path(self.data_root_path) / 'mask')
        depth_stems = set(depth_images)
        for name, images in [('gray', gray_images), ('mask', mask_images)]:
            missing = sorted(depth_stems - set(images))
            extra = sorted(set(images) - depth_stems)
            if missing or extra:
                raise ValueError(
                    '{} frames do not match depth frames (missing={}, extra={})'.format(
                        name, missing, extra
                    )
                )

        self.frame_stems = sorted(depth_stems)
        self.depth_paths = [depth_images[stem] for stem in self.frame_stems]
        self.gray_paths = [gray_images[stem] for stem in self.frame_stems]
        self.mask_paths = [mask_images[stem] for stem in self.frame_stems]

        intrinsics = cv2.FileStorage(int_yaml_path, cv2.FILE_STORAGE_READ)
        try:
            camera_matrix = intrinsics.getNode('M1').mat()
        finally:
            intrinsics.release()
        if camera_matrix is None:
            raise ValueError('M1 is missing from {}'.format(int_yaml_path))
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if camera_matrix.shape != (3, 3):
            raise ValueError('M1 in {} must be 3x3'.format(int_yaml_path))
        self.camera_matrix = camera_matrix

    @staticmethod
    def _read_image(path, flags):
        image = cv2.imread(str(path), flags)
        if image is None:
            raise ValueError('failed to read image {}'.format(path))
        return image

    @staticmethod
    def _load_media_sequence(path, color=False):
        path = Path(path)
        frames = []
        if path.is_dir():
            supported_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
            images = sorted(
                item for item in path.iterdir() if item.is_file() and item.suffix.lower() in supported_exts
            )
            if not images:
                raise ValueError('目录不包含支持的图像：{}'.format(path))
            for image_path in images:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError('failed to read image {}'.format(image_path))
                if not color and image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif color and image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                frames.append(image)
            return frames

        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
            while ok:
                if not color:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                else:
                    frames.append(frame)
                ok, frame = cap.read()

            if frames:
                return frames

            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('failed to read media from {}'.format(path))
            if not color and image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif color and image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return [image]
        finally:
            cap.release()

    def _rectify_frame_pair(self, idx):
        current_left = self.left_images[idx]
        current_right = self.right_images[idx]

        extri = cv2.FileStorage(self.ext_yaml_path, cv2.FILE_STORAGE_READ)
        intri = cv2.FileStorage(self.int_yaml_path, cv2.FILE_STORAGE_READ)
        try:
            M1 = intri.getNode('M1').mat()
            M2 = intri.getNode('M2').mat()
            D1 = intri.getNode('D1').mat()
            D2 = intri.getNode('D2').mat()

            R = extri.getNode('R').mat()
            t = extri.getNode('T').mat()

            height, width = current_left.shape[:2]
            new_M1, _ = cv2.getOptimalNewCameraMatrix(M1, D1, (width, height), 1, (width, height))
            new_M2, _ = cv2.getOptimalNewCameraMatrix(M2, D2, (width, height), 1, (width, height))

            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                new_M1,
                D1,
                new_M2,
                D2,
                (width, height),
                R,
                t,
                flags=cv2.CALIB_ZERO_TANGENT_DIST,
            )

            left_map_x, left_map_y = cv2.initUndistortRectifyMap(new_M1, D1, R1, P1, (width, height), cv2.CV_32FC1)
            right_map_x, right_map_y = cv2.initUndistortRectifyMap(new_M2, D2, R2, P2, (width, height), cv2.CV_32FC1)

            left_rectified = cv2.remap(current_left, left_map_x, left_map_y, cv2.INTER_LINEAR)
            right_rectified = cv2.remap(current_right, right_map_x, right_map_y, cv2.INTER_LINEAR)

            focal_length = float(new_M1[0, 0])
            baseline = abs(float(t[0][0]))
            min_dis = self.min_depth_mm
            max_dis = self.max_depth_mm

            return {
                'left_rectified': left_rectified,
                'right_rectified': right_rectified,
                'camera_params': {
                    'K1': new_M1,
                    'K2': new_M2,
                    'cam_R': R,
                    'cam_t': t,
                    'P1': P1,
                    'P2': P2,
                    'Q': Q,
                    'min_dis': min_dis,
                    'max_dis': max_dis,
                    'min_disp': focal_length * baseline / max_dis,
                    'max_disp': focal_length * baseline / min_dis,
                },
            }
        finally:
            extri.release()
            intri.release()

    def __len__(self):
        if self.input_mode == 'direct_depth':
            return len(self.depth_paths)
        return len(self.left_images)

    def __getitem__(self, idx):
        if self.input_mode == 'direct_depth':
            depth = self._read_image(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
            gray = self._read_image(self.gray_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = self._read_image(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if depth.ndim != 2:
                raise ValueError('depth image must be single-channel: {}'.format(self.depth_paths[idx]))
            if gray.shape != depth.shape or mask.shape != depth.shape:
                raise ValueError(
                    'depth/gray/mask shape mismatch for frame {}'.format(
                        self.frame_stems[idx]
                    )
                )
            return {
                'idx': idx,
                'frame_name': self.frame_stems[idx],
                'depth': depth,
                'mask': mask,
                'gray_left': gray,
                'rgb': None,
                'camera_params': {
                    'K1': self.camera_matrix,
                    'min_dis': self.min_depth_mm,
                    'max_dis': self.max_depth_mm,
                },
            }

        rectified = self._rectify_frame_pair(idx)

        rgb = None
        if self.rgb_images is not None:
            rgb = self.rgb_images[idx]

        return {
            'idx': idx,
            'left_rectified': rectified['left_rectified'],
            'right_rectified': rectified['right_rectified'],
            'gray_left': self.left_images[idx],
            'rgb': rgb,
            'camera_params': rectified['camera_params'],
        }
