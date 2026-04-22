import os
from pathlib import Path

import cv2
from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        data_root_path,
        ext_yaml_path,
        int_yaml_path,
        color_ext_yaml_path=None,
    ):
        self.data_root_path = data_root_path
        self.ext_yaml_path = ext_yaml_path
        self.int_yaml_path = int_yaml_path
        self.color_ext_yaml_path = color_ext_yaml_path

        self.left_images = self._load_media_sequence(os.path.join(data_root_path, 'left'))
        self.right_images = self._load_media_sequence(os.path.join(data_root_path, 'right'))

        rgb_path = os.path.join(data_root_path, 'color')
        self.rgb_images = None
        if os.path.exists(rgb_path):
            self.rgb_images = self._load_media_sequence(rgb_path, color=True)

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
            min_dis = 100
            max_dis = 4000

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
        return len(self.left_images)

    def __getitem__(self, idx):
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