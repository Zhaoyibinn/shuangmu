#!/usr/bin/env python3
"""Reorganize and rename sequence images inside the suipian subfolder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re


def _extract_numeric_suffix(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (0, path.name)


def build_rename_map(directory: Path, start_index: int = 1) -> dict[Path, Path]:
    """Create a mapping from existing files to their new paths inside prefix folders."""

    candidates: dict[str, list[Path]] = {prefix: [] for prefix in ("left", "right", "depth", "color")}
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        for prefix in candidates:
            if entry.name.startswith(f"{prefix}_") and entry.suffix.lower() == ".png":
                candidates[prefix].append(entry)
                break

    rename_map: dict[Path, Path] = {}
    for prefix, files in candidates.items():
        sorted_files = sorted(files, key=_extract_numeric_suffix)
        if not sorted_files:
            continue
        prefix_dir = directory / prefix
        for offset, source in enumerate(sorted_files, start=start_index):
            target = prefix_dir / f"{prefix}{offset:04d}.png"
            rename_map[source] = target

    return rename_map


def confirm_no_conflicts(rename_map: dict[Path, Path]) -> None:
    """Ensure no target already exists unless it is one of the planned sources."""

    existing_targets = [
        target
        for target in rename_map.values()
        if target.exists() and target not in rename_map
    ]
    if existing_targets:
        names = ", ".join(str(p.name) for p in existing_targets)
        raise SystemExit(f"停止：目标文件已存在 ({names})")


def run(directory: Path, dry_run: bool, start_index: int) -> None:
    rename_map = build_rename_map(directory, start_index)
    if not rename_map:
        print("没有匹配的图像需要重命名。", file=sys.stderr)
        return

    confirm_no_conflicts(rename_map)

    for src, dst in sorted(rename_map.items()):
        print(f"{src.name} -> {dst.name}")
    if dry_run:
        print("--dry-run 模式，不执行实际重命名。")
        return

    for target_path in {dst.parent for dst in rename_map.values()}:
        target_path.mkdir(parents=True, exist_ok=True)

    for src, dst in sorted(rename_map.items()):
        src.rename(dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 suipian 子目录里的 left/right/depth/color 图像移动并重命名成统一编号。"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="目标文件夹（默认为 d455_jiegouguang_save/20260116_suipian/d455_suipian_huojianpengguan）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不执行重命名")
    parser.add_argument("--start", type=int, default=1, help="编号起始值（默认 1）")

    args = parser.parse_args()
    if not args.path.is_dir():
        raise SystemExit(f"路径不是目录：{args.path}")

    run(args.path, args.dry_run, args.start)


if __name__ == "__main__":
    main()
