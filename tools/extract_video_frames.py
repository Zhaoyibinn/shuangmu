from argparse import ArgumentParser
from pathlib import Path

import cv2


def build_output_dir(video_path, output_dir):
    if output_dir is not None:
        return Path(output_dir)
    return video_path.parent / f"{video_path.stem}_frames"


def parse_args():
    parser = ArgumentParser(description="Extract frames from an mp4 video into sequential images.")
    parser.add_argument("video", help="Path to the input mp4 file")
    parser.add_argument("-o", "--output-dir", help="Directory to save extracted frames")
    parser.add_argument("--prefix", default="frame", help="Output image filename prefix")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="Output image format")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index for output filenames")
    parser.add_argument("--step", type=int, default=1, help="Save every Nth frame")
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"expected an mp4 file, got: {video_path}")
    if args.step <= 0:
        raise ValueError("step must be a positive integer")

    output_dir = build_output_dir(video_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    frame_idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.step == 0:
            output_name = f"{args.prefix}_{args.start_index + saved_count:06d}.{args.ext}"
            output_path = output_dir / output_name
            ok = cv2.imwrite(str(output_path), frame)
            if not ok:
                raise RuntimeError(f"failed to save frame: {output_path}")
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"video: {video_path}")
    print(f"saved {saved_count} frames to: {output_dir}")


if __name__ == "__main__":
    main()