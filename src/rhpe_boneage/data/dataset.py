from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..training.normalization import ScalarNormalizer


@dataclass
class DatasetStats:
    target_normalizer: ScalarNormalizer
    chronological_normalizer: ScalarNormalizer


def generate_heatmap(height: int, width: int, keypoints: list[list[float]], sigma: float) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    if sigma <= 0:
        sigma = 8.0
    yy, xx = np.mgrid[0:height, 0:width]
    for x_coord, y_coord, visibility in keypoints:
        if visibility <= 0:
            continue
        if x_coord <= 0 and y_coord <= 0:
            continue
        heatmap += np.exp(-((xx - x_coord) ** 2 + (yy - y_coord) ** 2) / (2.0 * sigma**2))
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap.astype(np.float32)


def _safe_square_patch(image: np.ndarray, center_x: float, center_y: float, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    patch = np.zeros((patch_size, patch_size), dtype=image.dtype)
    x0 = int(round(center_x)) - half
    y0 = int(round(center_y)) - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image.shape[1], x1)
    src_y1 = min(image.shape[0], y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        patch[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return patch


def _sanitize_coco_bbox(image_shape: tuple[int, int], bbox: list[float]) -> list[float]:
    height, width = image_shape
    x_coord, y_coord, box_width, box_height = [float(value) for value in bbox]

    x0 = min(max(x_coord, 0.0), float(width))
    y0 = min(max(y_coord, 0.0), float(height))
    x1 = min(max(x_coord + box_width, 0.0), float(width))
    y1 = min(max(y_coord + box_height, 0.0), float(height))

    clipped_width = max(x1 - x0, 1.0)
    clipped_height = max(y1 - y0, 1.0)
    x0 = min(x0, max(float(width) - clipped_width, 0.0))
    y0 = min(y0, max(float(height) - clipped_height, 0.0))
    return [x0, y0, clipped_width, clipped_height]


def _crop_to_bbox_context(
    image: np.ndarray,
    heatmap: np.ndarray,
    keypoints: np.ndarray,
    bbox: list[float],
    margin_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    height, width = image.shape
    bbox = _sanitize_coco_bbox((height, width), bbox)
    x_coord, y_coord, box_width, box_height = bbox
    margin_x = box_width * margin_ratio
    margin_y = box_height * margin_ratio

    x0 = max(0, int(np.floor(x_coord - margin_x)))
    y0 = max(0, int(np.floor(y_coord - margin_y)))
    x1 = min(width, int(np.ceil(x_coord + box_width + margin_x)))
    y1 = min(height, int(np.ceil(y_coord + box_height + margin_y)))

    if x1 <= x0 or y1 <= y0:
        return image, heatmap, keypoints, bbox

    cropped_image = image[y0:y1, x0:x1]
    cropped_heatmap = heatmap[y0:y1, x0:x1]
    cropped_keypoints = keypoints.copy()
    for index in range(cropped_keypoints.shape[0]):
        if cropped_keypoints[index, 2] > 0:
            cropped_keypoints[index, 0] -= x0
            cropped_keypoints[index, 1] -= y0

    cropped_bbox = [
        float(bbox[0] - x0),
        float(bbox[1] - y0),
        float(bbox[2]),
        float(bbox[3]),
    ]
    return cropped_image, cropped_heatmap, cropped_keypoints, cropped_bbox


def _resize_with_geometry(
    image: np.ndarray,
    heatmap: np.ndarray,
    keypoints: np.ndarray,
    bbox: list[float],
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    source_height, source_width = image.shape
    if source_height == target_size and source_width == target_size:
        return image, heatmap, keypoints, bbox

    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    resized_heatmap = cv2.resize(heatmap, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scale_x = target_size / max(source_width, 1)
    scale_y = target_size / max(source_height, 1)

    resized_keypoints = keypoints.copy()
    resized_keypoints[:, 0] *= scale_x
    resized_keypoints[:, 1] *= scale_y
    resized_bbox = [
        float(bbox[0] * scale_x),
        float(bbox[1] * scale_y),
        float(bbox[2] * scale_x),
        float(bbox[3] * scale_y),
    ]
    return resized_image, resized_heatmap, resized_keypoints, resized_bbox


class RHPEBoneAgeDataset(Dataset):
    """基于严格匹配后的记录构建数据集。"""

    def __init__(
        self,
        records: list[dict[str, Any]],
        config: dict,
        stats: DatasetStats,
        geometric_transform,
        image_intensity_transform,
    ) -> None:
        self.records = records
        self.config = config
        self.stats = stats
        self.geometric_transform = geometric_transform
        self.image_intensity_transform = image_intensity_transform
        self.patch_size = int(config["data"]["local_patch_size"])
        self.max_keypoints = int(config["data"]["max_keypoints"])
        self.input_size = int(config["data"]["input_size"])
        self.global_crop_mode = config["data"].get("global_crop_mode", "bbox")
        self.global_crop_margin_ratio = float(config["data"].get("global_crop_margin_ratio", 0.05))

    def __len__(self) -> int:
        return len(self.records)

    def _transform_roi(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        bbox: list[float],
        keypoints: list[list[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        valid_indices = []
        valid_keypoints = []
        for index, (x_coord, y_coord, visibility) in enumerate(keypoints[: self.max_keypoints]):
            if visibility > 0:
                valid_indices.append(index)
                valid_keypoints.append((float(x_coord), float(y_coord)))

        bbox = _sanitize_coco_bbox(image.shape, bbox)
        transformed = self.geometric_transform(
            image=image,
            heatmap=heatmap,
            keypoints=valid_keypoints,
            bboxes=[bbox],
            bbox_labels=["hand"],
        )

        transformed_image = transformed["image"]
        transformed_heatmap = transformed["heatmap"]
        transformed_bbox = transformed["bboxes"][0] if transformed["bboxes"] else [0.0, 0.0, float(image.shape[1]), float(image.shape[0])]
        transformed_bbox = _sanitize_coco_bbox(transformed_image.shape, list(transformed_bbox))

        reconstructed = np.zeros((self.max_keypoints, 3), dtype=np.float32)
        for index, point in enumerate(keypoints[: self.max_keypoints]):
            reconstructed[index, 2] = float(point[2])
        for original_index, (x_coord, y_coord) in zip(valid_indices, transformed["keypoints"]):
            reconstructed[original_index, 0] = float(x_coord)
            reconstructed[original_index, 1] = float(y_coord)
        return transformed_image, transformed_heatmap, reconstructed, list(transformed_bbox)

    def _extract_local_tensors(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        keypoints: np.ndarray,
        bbox: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_patches = []
        heatmap_patches = []
        patch_mask = []

        height, width = image.shape
        for x_coord, y_coord, visibility in keypoints[: self.max_keypoints]:
            in_bounds = 0 <= x_coord < width and 0 <= y_coord < height
            is_valid = float(visibility > 0 and in_bounds)
            if is_valid:
                image_patch = _safe_square_patch(image, x_coord, y_coord, self.patch_size)
                heatmap_patch = _safe_square_patch(heatmap, x_coord, y_coord, self.patch_size)
            else:
                image_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                heatmap_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            image_patches.append(image_patch)
            heatmap_patches.append(heatmap_patch)
            patch_mask.append(is_valid)

        if not any(patch_mask):
            center_x = bbox[0] + bbox[2] / 2.0
            center_y = bbox[1] + bbox[3] / 2.0
            image_patches[0] = _safe_square_patch(image, center_x, center_y, self.patch_size)
            heatmap_patches[0] = _safe_square_patch(heatmap, center_x, center_y, self.patch_size)
            patch_mask[0] = 1.0

        image_tensor = torch.from_numpy(np.stack(image_patches)).unsqueeze(1).float()
        heatmap_tensor = torch.from_numpy(np.stack(heatmap_patches)).unsqueeze(1).float()
        mask_tensor = torch.tensor(patch_mask, dtype=torch.float32)
        return image_tensor, heatmap_tensor, mask_tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = np.array(Image.open(record["image_path"]).convert("L"), dtype=np.uint8)
        bbox = _sanitize_coco_bbox(image.shape, list(record["bbox"]))
        keypoints = record["keypoints"][: self.max_keypoints]
        sigma = max(bbox[2], bbox[3]) * float(self.config["data"]["heatmap_sigma_ratio"])
        heatmap = generate_heatmap(image.shape[0], image.shape[1], keypoints, sigma=max(sigma, 6.0))
        try:
            image, heatmap, keypoints_arr, bbox = self._transform_roi(image, heatmap, bbox, keypoints)
        except Exception as exc:
            raise ValueError(f"样本 {record['id']} 在 ROI 几何变换阶段失败: {exc}") from exc
        if self.global_crop_mode == "bbox":
            image, heatmap, keypoints_arr, bbox = _crop_to_bbox_context(
                image=image,
                heatmap=heatmap,
                keypoints=keypoints_arr,
                bbox=bbox,
                margin_ratio=self.global_crop_margin_ratio,
            )
            image, heatmap, keypoints_arr, bbox = _resize_with_geometry(
                image=image,
                heatmap=heatmap,
                keypoints=keypoints_arr,
                bbox=bbox,
                target_size=self.input_size,
            )

        image = self.image_intensity_transform(image=image)["image"].astype(np.float32)
        heatmap = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
        local_images, local_heatmaps, local_mask = self._extract_local_tensors(image, heatmap, keypoints_arr, bbox)

        height, width = image.shape
        bbox_vector = np.array(
            [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height,
            ],
            dtype=np.float32,
        )
        normalized_keypoints = np.zeros((self.max_keypoints, 3), dtype=np.float32)
        normalized_keypoints[:, 2] = keypoints_arr[:, 2]
        normalized_keypoints[:, 0] = keypoints_arr[:, 0] / width
        normalized_keypoints[:, 1] = keypoints_arr[:, 1] / height
        roi_vector = np.concatenate([bbox_vector, normalized_keypoints.reshape(-1)], axis=0)

        chronological_raw = float(record["chronological"])
        chronological_input = float(self.stats.chronological_normalizer.transform(chronological_raw))
        boneage = float(record["boneage"]) if record["has_boneage"] else np.nan

        return {
            "id": record["id"],
            "split": record["split"],
            "global_image": torch.from_numpy(image).unsqueeze(0).float(),
            "global_heatmap": torch.from_numpy(heatmap).unsqueeze(0).float(),
            "local_images": local_images,
            "local_heatmaps": local_heatmaps,
            "local_mask": local_mask,
            "roi_vector": torch.from_numpy(roi_vector).float(),
            "male": torch.tensor([float(record["male"])], dtype=torch.float32),
            "male_index": torch.tensor(int(record["male"]), dtype=torch.long),
            "chronological": torch.tensor([chronological_raw], dtype=torch.float32),
            "chronological_input": torch.tensor([chronological_input], dtype=torch.float32),
            "boneage": torch.tensor([boneage], dtype=torch.float32),
            "has_target": torch.tensor(record["has_boneage"], dtype=torch.bool),
        }
