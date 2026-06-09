import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


WORKSPACE_DIR = Path(__file__).resolve().parents[1]


def import_yolo():
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError:
        ultralytics_dir = WORKSPACE_DIR / "submodule" / "ultralytics"
        if ultralytics_dir.exists():
            sys.path.insert(0, str(ultralytics_dir))
            from ultralytics import YOLO

            return YOLO
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO26x-seg weights on one image and export segmentation masks."
    )
    parser.add_argument(
        "--weights",
        default="huojian/d455_penguan_20260525/video/exp.pt",
        help="Path to YOLO26x-seg segmentation weights.",
    )
    parser.add_argument(
        "--image",
        default="huojian/d455_penguan_20260525/video/color/0000.png",
        help="Input image path.",
    )
    parser.add_argument(
        "--output-dir",
        default="depth_outputs/d455_penguan_20260525/video/yolo26_seg_0000",
        help="Directory for mask outputs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0 or cpu. Default lets Ultralytics choose.",
    )
    return parser.parse_args()


def mask_from_result(result, mask_id, image_shape):
    height, width = image_shape[:2]
    mask = result.masks.data[mask_id].cpu().numpy() > 0.5
    if mask.shape == (height, width):
        return mask

    mask_height, mask_width = mask.shape[:2]
    gain = min(mask_width / float(width), mask_height / float(height))
    if gain > 0:
        unpad_width = int(round(width * gain))
        unpad_height = int(round(height * gain))
        pad_x = int(round((mask_width - unpad_width) / 2.0))
        pad_y = int(round((mask_height - unpad_height) / 2.0))
        x0 = max(pad_x, 0)
        y0 = max(pad_y, 0)
        x1 = min(x0 + unpad_width, mask_width)
        y1 = min(y0 + unpad_height, mask_height)
        cropped = mask[y0:y1, x0:x1]
        if cropped.size > 0:
            return cv2.resize(
                cropped.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    # result.masks.data can be in the letterboxed inference shape. The polygon
    # coordinates in masks.xy are already scaled back to the original image.
    if result.masks.xy is not None and mask_id < len(result.masks.xy):
        polygon = result.masks.xy[mask_id]
        if polygon is not None and len(polygon) >= 3:
            polygon = np.round(polygon).astype(np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            aligned_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(aligned_mask, [polygon], 1)
            return aligned_mask.astype(bool)

    raise ValueError(
        "mask shape {} does not match image shape {}; no original-image polygon is available".format(
            mask.shape,
            (height, width),
        )
    )


def save_masks(result, image, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "0000_yolo26_seg_overlay.png"
    combined_mask_path = output_dir / "0000_yolo26_seg_mask.png"
    masked_image_path = output_dir / "0000_yolo26_seg_masked.png"

    if result.masks is None or len(result.masks) == 0:
        empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.imwrite(str(combined_mask_path), empty_mask)
        cv2.imwrite(str(masked_image_path), image)
        cv2.imwrite(str(annotated_path), image)
        print("No segmentation mask was detected.")
        print("Saved empty mask: {}".format(combined_mask_path))
        return

    masks = result.masks.data.cpu().numpy()
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    overlay = image.copy()

    boxes = result.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int) if boxes is not None and boxes.cls is not None else []
    confidences = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else []

    for mask_id, mask in enumerate(masks):
        mask_bool = mask_from_result(result, mask_id, image.shape)
        combined_mask |= mask_bool
        instance_mask_path = output_dir / "0000_yolo26_seg_mask_{:02d}.png".format(mask_id)
        cv2.imwrite(str(instance_mask_path), mask_bool.astype(np.uint8) * 255)

        color = np.array(
            [
                (37 * mask_id + 80) % 255,
                (97 * mask_id + 160) % 255,
                (151 * mask_id + 220) % 255,
            ],
            dtype=np.uint8,
        )
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * color).astype(np.uint8)

        class_id = int(class_ids[mask_id]) if mask_id < len(class_ids) else -1
        confidence = float(confidences[mask_id]) if mask_id < len(confidences) else 0.0
        print(
            "mask {:02d}: class={}, conf={:.4f}, area={} px, path={}".format(
                mask_id,
                class_id,
                confidence,
                int(mask_bool.sum()),
                instance_mask_path,
            )
        )

    combined_mask_u8 = combined_mask.astype(np.uint8) * 255
    masked_image = image.copy()
    masked_image[~combined_mask] = 0

    cv2.imwrite(str(combined_mask_path), combined_mask_u8)
    cv2.imwrite(str(masked_image_path), masked_image)
    cv2.imwrite(str(annotated_path), overlay)
    print("Saved combined mask: {}".format(combined_mask_path))
    print("Saved masked image: {}".format(masked_image_path))
    print("Saved overlay: {}".format(annotated_path))


def main():
    args = parse_args()
    weights_path = WORKSPACE_DIR / args.weights
    image_path = WORKSPACE_DIR / args.image
    output_dir = WORKSPACE_DIR / args.output_dir

    if not weights_path.exists():
        raise FileNotFoundError("weights not found: {}".format(weights_path))
    if not image_path.exists():
        raise FileNotFoundError("image not found: {}".format(image_path))

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to read image: {}".format(image_path))

    YOLO = import_yolo()
    model = YOLO(str(weights_path))
    predict_kwargs = {
        "source": str(image_path),
        "task": "segment",
        "imgsz": args.imgsz,
        "conf": args.conf,
        "retina_masks": True,
        "save": False,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    if not results:
        raise RuntimeError("YOLO returned no results")

    save_masks(results[0], image, output_dir)


if __name__ == "__main__":
    main()
