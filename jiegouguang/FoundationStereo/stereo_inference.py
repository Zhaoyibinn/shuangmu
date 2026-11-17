"""
FoundationStereo inference wrapper for point cloud, depth and disparity estimation.

This module provides a clean interface to FoundationStereo model for stereo vision tasks,
outputting results in the first camera coordinate system.
"""

import os
import sys
import torch
import numpy as np
import cv2
import contextlib
from time import perf_counter
from typing import Tuple, Optional, Dict, Any
from omegaconf import OmegaConf

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder


class StereoInference:
    """
    FoundationStereo inference wrapper for stereo vision tasks.

    Provides point cloud, depth and disparity estimation from rectified stereo pairs.
    Supports coordinate transformation back to original left camera frame.

    Supports large-scale pretrained models (e.g., 3.3GB ViT-Large models).
    """

    # Default model locations (in priority order)
    DEFAULT_MODEL_PATHS = [
        "jiegouguang/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",  # External large model
        "jiegouguang/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",  # Project local model
    ]

    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None, vit_size: str = 'vitl'):
        """
        Initialize FoundationStereo model and load checkpoint.

        Args:
            ckpt_path: Path to model checkpoint (.pth file). If None, auto-detect from default locations.
            device: Device to run inference on. If None, auto-detect CUDA availability.
            vit_size: ViT backbone size ('vitl', 'vits', 'vitb', 'vitg'). Default 'vitl' for large models.
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.args = None
        self.vit_size = vit_size

        # Auto-detect model path if not provided
        if ckpt_path is None:
            ckpt_path = self._find_model()

        # Load model
        self._load_model(ckpt_path)

    def _find_model(self) -> str:
        """Auto-detect model checkpoint from default paths."""
        for path in self.DEFAULT_MODEL_PATHS:
            if os.path.exists(path):
                print(f"Auto-detected model: {path}")
                return path

        raise FileNotFoundError(
            f"No model checkpoint found. Tried:\n" +
            "\n".join(f"  - {p}" for p in self.DEFAULT_MODEL_PATHS) +
            "\n\nPlease provide ckpt_path explicitly."
        )

    def _load_model(self, ckpt_path: str):
        """Load FoundationStereo model from checkpoint."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load configuration
        cfg_path = os.path.join(os.path.dirname(ckpt_path), 'cfg.yaml')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        cfg = OmegaConf.load(cfg_path)

        # Handle missing vit_size in config (common issue)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = self.vit_size
            print(f"Added missing vit_size: {self.vit_size}")

        self.args = OmegaConf.create(cfg)

        # Print model configuration
        model_size_gb = os.path.getsize(ckpt_path) / (1024**3)
        print(f"Loading FoundationStereo model:")
        print(f"  Model file: {ckpt_path}")
        print(f"  Model size: {model_size_gb:.1f} GB")
        print(f"  ViT backbone: {cfg.vit_size}")
        print(f"  Max disparity: {cfg.max_disp}")
        print(f"  Mixed precision: {cfg.mixed_precision}")

        # Create model
        self.model = FoundationStereo(self.args)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

        print(f"  Training steps: {ckpt.get('global_step', 'unknown')}")
        print(f"  Training epochs: {ckpt.get('epoch', 'unknown')}")
        print(f"  Device: {self.device}")
        print(f"Model loaded successfully!")

    def infer(self,
              left_img_bgr: np.ndarray,
              right_img_bgr: np.ndarray,
              K_rect: np.ndarray,
              baseline_m: float,
              R1: Optional[np.ndarray] = None,
              to_original_left_cam: bool = False,
              valid_iters: int = 32,
              timings: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Perform stereo inference on rectified image pair.

        Args:
            left_img_bgr: Left rectified image (H, W, 3) in BGR uint8 format
            right_img_bgr: Right rectified image (H, W, 3) in BGR uint8 format
            K_rect: Rectified camera intrinsic matrix (3, 3)
            baseline_m: Baseline distance in meters
            R1: Optional rotation matrix (3, 3) from rectified to original left camera
            to_original_left_cam: If True and R1 provided, transform points to original frame
            valid_iters: Number of refinement iterations
            timings: Optional output dict of stage wall times in ms (CUDA synchronized).

        Returns:
            Dictionary containing:
            - disparity: (H, W) disparity map in pixels
            - depth: (H, W) depth map in meters
            - points_cam1: (H, W, 3) point cloud in left camera coordinates (meters)
            - valid_mask: (H, W) boolean mask for valid depth values
        """
        def boundary():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            return perf_counter()

        started = previous = boundary() if timings is not None else 0.

        def mark(name):
            nonlocal previous
            if timings is not None:
                now = boundary()
                timings[name] = (now - previous) * 1000.
                previous = now

        # Input validation
        assert left_img_bgr.shape == right_img_bgr.shape, "Image shapes must match"
        # assert len(left_img_bgr.shape) == 3 and left_img_bgr.shape[2] == 3, "Images must be (H,W,3)"
        if len(left_img_bgr.shape) == 2:
            left_img_bgr = cv2.cvtColor(left_img_bgr, cv2.COLOR_GRAY2BGR)
            right_img_bgr = cv2.cvtColor(right_img_bgr, cv2.COLOR_GRAY2BGR)

        assert K_rect.shape == (3, 3), "K_rect must be (3,3)"
        assert baseline_m > 0, "Baseline must be positive"

        H, W = left_img_bgr.shape[:2]

        # Convert BGR to RGB (keep 0-255 range, model will normalize internally)
        left_rgb = cv2.cvtColor(left_img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        right_rgb = cv2.cvtColor(right_img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Convert to torch tensors
        left_tensor = torch.from_numpy(left_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        right_tensor = torch.from_numpy(right_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Pad images for network processing
        padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_padded, right_padded = padder.pad(left_tensor, right_tensor)

        mark('input_convert_upload_pad')

        # Run inference
        autocast_ctx = torch.cuda.amp.autocast if self.device.type == 'cuda' else contextlib.nullcontext
        autocast_kwargs = {'enabled': True} if self.device.type == 'cuda' else {}

        # Gradient mode is thread-local: Gradio may infer on a different worker
        # from the one that constructed the model.
        with torch.inference_mode(), autocast_ctx(**autocast_kwargs):
            disp_padded = self.model.forward(left_padded, right_padded, iters=valid_iters, test_mode=True)

        mark('network_forward')

        # Unpad and convert to numpy
        disp_tensor = padder.unpad(disp_padded.float())
        disp_pred = disp_tensor.squeeze().cpu().numpy()

        mark('unpad_download')

        # Handle resolution mismatch - resize disparity and scale values
        if disp_pred.shape != (H, W):
            H_pred, W_pred = disp_pred.shape
            scale_factor = W / W_pred
            disp_pred = cv2.resize(disp_pred, (W, H), interpolation=cv2.INTER_LINEAR)
            disp_pred *= scale_factor  # Scale disparity values

        # Compute depth from disparity
        # depth = fx * baseline / disparity
        fx = K_rect[0, 0]
        valid_mask = disp_pred > 0
        depth = np.zeros_like(disp_pred)
        depth[valid_mask] = fx * baseline_m / disp_pred[valid_mask]

        mark('disparity_to_depth')

        # Generate 3D points in camera coordinates
        points_cam1 = self._disparity_to_points(disp_pred, K_rect, baseline_m, valid_mask)

        # Transform to original left camera frame if requested
        if to_original_left_cam and R1 is not None:
            assert R1.shape == (3, 3), "R1 must be (3,3)"
            points_cam1 = self._transform_points(points_cam1, R1.T, valid_mask)

        mark('generate_points_transform')
        if timings is not None:
            timings['total_ms'] = (perf_counter() - started) * 1000.

        return {
            'disparity': disp_pred,
            'depth': depth,
            'points_cam1': points_cam1,
            'valid_mask': valid_mask
        }

    def _disparity_to_points(self, disparity: np.ndarray, K_rect: np.ndarray,
                           baseline_m: float, valid_mask: np.ndarray) -> np.ndarray:
        """Convert disparity map to 3D points in left camera coordinates."""
        H, W = disparity.shape
        fx, fy = K_rect[0, 0], K_rect[1, 1]
        cx, cy = K_rect[0, 2], K_rect[1, 2]

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Initialize points array
        points = np.zeros((H, W, 3), dtype=np.float32)

        # Compute 3D coordinates where valid
        points[valid_mask, 2] = fx * baseline_m / disparity[valid_mask]  # Z (depth)
        points[valid_mask, 0] = (u[valid_mask] - cx) * points[valid_mask, 2] / fx  # X
        points[valid_mask, 1] = (v[valid_mask] - cy) * points[valid_mask, 2] / fy  # Y

        return points

    def _transform_points(self, points: np.ndarray, R: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Transform 3D points using rotation matrix R."""
        H, W = points.shape[:2]
        transformed_points = np.zeros_like(points)

        # Reshape valid points for matrix multiplication
        valid_points = points[valid_mask]  # (N, 3)
        if len(valid_points) > 0:
            transformed_valid = (R @ valid_points.T).T  # (3, 3) @ (3, N) -> (3, N) -> (N, 3)
            transformed_points[valid_mask] = transformed_valid

        return transformed_points


def create_example_images(height: int = 480, width: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic stereo pair for testing."""
    # Create textured left image
    left_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Create right image with horizontal shift to simulate disparity
    right_img = np.zeros_like(left_img)
    shift = 10  # pixels
    right_img[:, shift:] = left_img[:, :-shift]
    right_img[:, :shift] = left_img[:, :shift]  # Fill border

    return left_img, right_img


def create_example_calibration(width: int = 640, height: int = 480) -> Tuple[np.ndarray, float]:
    """Create example camera calibration parameters."""
    # Typical camera intrinsics
    fx = fy = 400.0
    cx = width / 2.0
    cy = height / 2.0

    K_rect = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    baseline_m = 0.1  # 10cm baseline

    return K_rect, baseline_m


if __name__ == "__main__":
    # Quick test script
    print("Testing StereoInference class...")

    # Create test data
    left_img, right_img = create_example_images()
    K_rect, baseline_m = create_example_calibration()

    # Test model path (adjust to your setup)
    ckpt_path = "./pretrained_models/23-51-11/model_best_bp2.pth"

    if os.path.exists(ckpt_path):
        try:
            # Initialize inference
            stereo_infer = StereoInference(ckpt_path)

            # Run inference
            results = stereo_infer.infer(left_img, right_img, K_rect, baseline_m)

            print(f"Disparity shape: {results['disparity'].shape}")
            print(f"Depth shape: {results['depth'].shape}")
            print(f"Points shape: {results['points_cam1'].shape}")
            print(f"Valid pixels: {results['valid_mask'].sum()}/{results['valid_mask'].size}")
            print(f"Depth range: {results['depth'][results['valid_mask']].min():.3f} - {results['depth'][results['valid_mask']].max():.3f}m")

            print("Basic test passed!")

        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print(f"Checkpoint not found: {ckpt_path}")
        print("Please ensure the model checkpoint exists to run tests.")
