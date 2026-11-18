"""Foreground masks from local Ultralytics segmentation checkpoints."""
from pathlib import Path
import numpy as np
from Recon.Reconstruction import Reconstruction, import_yolo

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO = 'jiegouguang/weights/yolo/suobipengguan/last.pt'


class ForegroundSegmenter:
    def __init__(self, weights=DEFAULT_YOLO, conf=.25, imgsz=640, device=None):
        path = Path(weights)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f'请复制 YOLO 分割权重到：{path}')
        self.model = import_yolo()(str(path))
        if self.model.task not in ('segment', 'semantic'):
            raise ValueError('需要 YOLO 分割权重（segment 或 semantic），不能使用检测权重')
        self.conf, self.imgsz, self.device = float(conf), int(imgsz), device
        if not 0 <= self.conf <= 1 or self.imgsz <= 0:
            raise ValueError('YOLO conf 必须在 0–1，imgsz 必须 > 0')

    def segment(self, bgr):
        kwargs = dict(source=bgr, conf=self.conf, imgsz=self.imgsz,
                      retina_masks=True, verbose=False, save=False)
        if self.device is not None and self.device != '':
            kwargs['device'] = self.device
        results = self.model.predict(**kwargs)
        foreground = np.zeros(bgr.shape[:2], dtype=bool)
        if self.model.task == 'semantic':
            if not results or results[0].semantic_mask is None:
                raise ValueError('YOLO semantic 模型未返回 semantic_mask')
            result = results[0]
            class_map = result.semantic_mask.data.cpu().numpy()
            # Ultralytics binary semantic models use 0=background, 1=foreground,
            # even though names contains only the target class at index 0.
            if len(result.names) == 1:
                foreground = class_map == 1
            else:
                foreground_ids = [
                    class_id for class_id, name in result.names.items()
                    if str(name).strip().lower() not in ('background', '_background_', '__background__', 'bg', '背景')
                ]
                foreground = np.isin(class_map, foreground_ids)
            # Semantic predictions are already scaled to the original image.
            if foreground.shape != bgr.shape[:2]:
                raise ValueError('YOLO semantic_mask 与输入 RGB 尺寸不一致')
            return foreground
        if results and results[0].masks is not None:
            for mask in results[0].masks.data:
                foreground |= Reconstruction.align_letterbox_mask(mask.cpu().numpy() > .5, foreground.shape)
        return foreground


def mask_depth(depth_mm, foreground):
    if depth_mm.shape != foreground.shape:
        raise ValueError('前景 mask 与 RGB 对齐深度尺寸不一致')
    return np.where(foreground & np.isfinite(depth_mm) & (depth_mm > 0), depth_mm, 0).astype(np.float32)
