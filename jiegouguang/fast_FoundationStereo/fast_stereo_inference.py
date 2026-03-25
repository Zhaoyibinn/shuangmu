"""
Fast-FoundationStereo inference wrapper for disparity estimation.

使用序列化模型（model_best_bp2_serialize.pth）进行推理，与 FoundationStereo/stereo_inference.py 接口保持一致。
"""

import os
import sys
import importlib.util
import torch
import numpy as np
import cv2
from typing import Optional, Dict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

AMP_DTYPE = torch.float32


def _load_input_padder():
    """从 fast_FoundationStereo/core/utils/utils.py 直接加载 InputPadder，避免与 FoundationStereo 的 core 冲突。"""
    utils_path = os.path.join(_THIS_DIR, 'core', 'utils', 'utils.py')
    spec = importlib.util.spec_from_file_location('fast_fs_core_utils', utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.InputPadder


class FastStereoInference:
    """
    Fast-FoundationStereo inference wrapper.

    使用 torch.load 加载序列化完整模型（.pth），提供与 StereoInference 一致的 infer() 接口。
    """

    DEFAULT_MODEL_PATH = "jiegouguang/weights/fast_FoundationStereo/23-36-37/model_best_bp2_serialize.pth"

    @staticmethod
    def _configure_safe_precision(model):
        model.args.mixed_precision = False
        utils_mod = sys.modules.get('Utils')
        if utils_mod is not None and hasattr(utils_mod, 'AMP_DTYPE'):
            utils_mod.AMP_DTYPE = torch.float32
        return model

    def __init__(self, ckpt_path: Optional[str] = None, valid_iters: int = 8, max_disp: int = 192):
        """
        初始化并加载 Fast-FoundationStereo 模型。

        Args:
            ckpt_path: 序列化模型权重路径（.pth）。若为 None，使用默认路径。
            valid_iters: 推理时的迭代次数，默认 8。
            max_disp: 最大视差，默认 192。
        """
        if ckpt_path is None:
            ckpt_path = self.DEFAULT_MODEL_PATH

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Fast-FoundationStereo 模型文件不存在: {ckpt_path}")

        print(f"Loading Fast-FoundationStereo model: {ckpt_path}")

        # torch.load 反序列化时，pickle 会顺着 sys.path 查找 core.* 和 Utils 模块。
        # FoundationStereo/stereo_inference.py 会把 jiegouguang/FoundationStereo/ 加入 sys.path，
        # 导致 fast_FoundationStereo/core/submodule.py 里的 `from Utils import AMP_DTYPE`
        # 找到了 FoundationStereo/Utils.py（其中没有 AMP_DTYPE），从而报错。
        # 修复：加载前
        #   1. 保存并清除 sys.modules 中所有 core* 和 Utils 条目
        #   2. 临时把 FoundationStereo/ 目录从 sys.path 剔除，把 fast_FoundationStereo/ 置首
        # 加载后恢复原状。
        _fs_dir = os.path.join(os.path.dirname(_THIS_DIR), 'FoundationStereo')
        _conflict_keys = [k for k in sys.modules
                          if k == 'core' or k.startswith('core.') or k == 'Utils']
        _saved_modules = {k: sys.modules.pop(k) for k in _conflict_keys}
        _saved_path = sys.path[:]
        sys.path[:] = [_THIS_DIR] + [p for p in sys.path
                                     if p != _THIS_DIR and p != _fs_dir]
        try:
            model = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        finally:
            sys.path[:] = _saved_path
            for k in list(sys.modules.keys()):
                if k == 'core' or k.startswith('core.') or k in ('Utils', 'fast_fs_core_utils'):
                    del sys.modules[k]
            sys.modules.update(_saved_modules)

        model.args.valid_iters = valid_iters
        model.args.max_disp = max_disp
        model = self._configure_safe_precision(model)
        model.cuda().eval()
        torch.autograd.set_grad_enabled(False)

        self.model = model
        self.device = torch.device('cuda')
        print("Fast-FoundationStereo model loaded successfully!")

    def infer(self,
              left_img_bgr: np.ndarray,
              right_img_bgr: np.ndarray,
              K_rect: np.ndarray,
              baseline_m: float) -> Dict[str, np.ndarray]:
        """
        对已极线校正的双目图像对进行立体匹配推理。

        Args:
            left_img_bgr: 左图（BGR 或灰度），uint8，(H, W, 3) 或 (H, W)
            right_img_bgr: 右图（BGR 或灰度），uint8，(H, W, 3) 或 (H, W)
            K_rect: 校正后的相机内参矩阵 (3, 3)
            baseline_m: 基线距离（米，正数）

        Returns:
            字典，包含:
                - 'disparity': 视差图 (H, W)，float32，单位像素
                - 'depth'    : 深度图 (H, W)，float32，单位米
        """
        assert left_img_bgr.shape == right_img_bgr.shape, "左右图尺寸必须一致"
        assert K_rect.shape == (3, 3), "K_rect 必须为 (3, 3)"
        assert baseline_m > 0, "基线距离必须为正数"

        # 灰度图转三通道 RGB
        if left_img_bgr.ndim == 2:
            img0 = np.tile(left_img_bgr[..., None], (1, 1, 3)).astype(np.float32)
            img1 = np.tile(right_img_bgr[..., None], (1, 1, 3)).astype(np.float32)
        else:
            img0 = left_img_bgr[:, :, ::-1].astype(np.float32)   # BGR -> RGB
            img1 = right_img_bgr[:, :, ::-1].astype(np.float32)

        H, W = img0.shape[:2]

        img0_t = torch.as_tensor(img0).to(self.device).float()[None].permute(0, 3, 1, 2)
        img1_t = torch.as_tensor(img1).to(self.device).float()[None].permute(0, 3, 1, 2)

        padder = _load_input_padder()(img0_t.shape, divis_by=32, force_square=False)
        img0_t, img1_t = padder.pad(img0_t, img1_t)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False, dtype=AMP_DTYPE):
            disp = self.model.forward(img0_t, img1_t,
                                      iters=self.model.args.valid_iters,
                                      test_mode=True,
                                      optimize_build_volume='pytorch1')

        disp = padder.unpad(disp.float())
        disp_np = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        # 由视差计算深度
        fx = float(K_rect[0, 0])
        valid_mask = disp_np > 0
        depth = np.zeros_like(disp_np)
        depth[valid_mask] = fx * baseline_m / disp_np[valid_mask]

        return {
            'disparity': disp_np,
            'depth': depth,
        }
