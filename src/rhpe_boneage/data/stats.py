from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_image_paths(image_dir: str | Path) -> list[Path]:
    directory = Path(image_dir)
    if not directory.exists():
        raise FileNotFoundError(f"图像目录不存在: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"给定路径不是目录: {directory}")
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_grayscale_array(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        array = np.asarray(grayscale, dtype=np.float64) / 255.0
    if array.ndim != 2:
        raise ValueError(f"图像不是二维灰度图: {image_path}")
    return array


def compute_grayscale_mean_std_from_paths(image_paths: list[Path]) -> dict[str, Any]:
    total_files = len(image_paths)
    valid_files = 0
    skipped_files = 0
    pixel_count = 0
    pixel_sum = 0.0
    pixel_sum_sq = 0.0
    skipped_details: list[dict[str, str]] = []

    for image_path in image_paths:
        try:
            array = load_grayscale_array(image_path)
        except Exception as exc:  # pragma: no cover - 真实坏图保护
            skipped_files += 1
            skipped_details.append({"path": str(image_path), "error": str(exc)})
            continue

        valid_files += 1
        pixel_count += int(array.size)
        pixel_sum += float(array.sum(dtype=np.float64))
        pixel_sum_sq += float(np.square(array, dtype=np.float64).sum(dtype=np.float64))

    if valid_files == 0 or pixel_count == 0:
        raise RuntimeError("没有可用于统计的有效图像。")

    mean = pixel_sum / pixel_count
    variance = max(pixel_sum_sq / pixel_count - mean * mean, 0.0)
    std = math.sqrt(variance)

    return {
        "total_image_files": total_files,
        "used_image_files": valid_files,
        "skipped_image_files": skipped_files,
        "pixel_count": pixel_count,
        "mean": mean,
        "std": std,
        "mean_255": mean * 255.0,
        "std_255": std * 255.0,
        "skipped_details": skipped_details,
    }


def compute_grayscale_mean_std(image_dir: str | Path) -> dict[str, Any]:
    directory = Path(image_dir)
    result = compute_grayscale_mean_std_from_paths(iter_image_paths(directory))
    result["image_dir"] = str(directory.resolve())
    return result


def load_mean_std_cache(path: str | Path) -> dict[str, Any] | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"归一化缓存格式错误: {cache_path}")
    mean = payload.get("mean")
    std = payload.get("std")
    if mean is None or std is None:
        raise ValueError(f"归一化缓存缺少 mean/std: {cache_path}")
    return payload


def save_mean_std_cache(payload: dict[str, Any], path: str | Path) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
