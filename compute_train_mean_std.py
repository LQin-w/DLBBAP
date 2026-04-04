from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:
    from _bootstrap import bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 RHPE_train 灰度图像的 mean/std")
    parser.add_argument(
        "--image-dir",
        default="dataset/RHPE_train",
        help="训练图像目录，默认使用 dataset/RHPE_train",
    )
    parser.add_argument(
        "--output",
        default="dataset/train_mean_std.json",
        help="统计结果保存路径，默认写入 dataset/train_mean_std.json",
    )
    return parser.parse_args()


def main() -> None:
    bootstrap()
    from rhpe_boneage.data.stats import compute_grayscale_mean_std, iter_image_paths

    args = parse_args()
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    image_paths = iter_image_paths(image_dir)
    result = compute_grayscale_mean_std(image_dir)

    for item in result.get("skipped_details", []):
        print(
            f"警告: 跳过损坏或不可读图像 -> {item['path']} | 原因: {item['error']}",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(f"训练图像目录: {image_dir.resolve()}")
    print(f"递归找到图像数量: {result['total_image_files']}")
    print(f"参与统计图像数量: {result['used_image_files']}")
    print(f"跳过图像数量: {result['skipped_image_files']}")
    print(f"总像素数量: {result['pixel_count']}")
    print(f"mean (0-1): {result['mean']:.8f}")
    print(f"std  (0-1): {result['std']:.8f}")
    print(f"mean (0-255): {result['mean_255']:.4f}")
    print(f"std  (0-255): {result['std_255']:.4f}")
    print(f"结果已保存到: {output_path.resolve()}")


if __name__ == "__main__":
    main()
