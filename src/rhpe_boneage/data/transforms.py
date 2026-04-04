from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2


def build_geometric_transform(config: dict, is_train: bool) -> A.Compose:
    aug_cfg = config["augmentation"]
    input_size = int(config["data"]["input_size"])
    horizontal_flip_p = min(max(float(aug_cfg.get("horizontal_flip_p", 0.5) or 0.5), 0.0), 1.0)
    transforms = []
    if is_train:
        transforms.append(
            A.Affine(
                scale=(1.0 - aug_cfg["scale_limit"], 1.0 + aug_cfg["scale_limit"]),
                translate_percent={
                    "x": (-aug_cfg["translate_limit"], aug_cfg["translate_limit"]),
                    "y": (-aug_cfg["translate_limit"], aug_cfg["translate_limit"]),
                },
                rotate=(-aug_cfg["rotation_limit"], aug_cfg["rotation_limit"]),
                shear=(-aug_cfg["shear_limit"], aug_cfg["shear_limit"]),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=aug_cfg["affine_p"],
            )
        )
        if aug_cfg.get("horizontal_flip", False):
            transforms.append(A.HorizontalFlip(p=horizontal_flip_p))

    transforms.append(A.Resize(height=input_size, width=input_size, interpolation=cv2.INTER_LINEAR))
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["bbox_labels"],
            min_visibility=0.0,
            clip=True,
            filter_invalid_bboxes=True,
        ),
        additional_targets={"heatmap": "image"},
    )


def build_image_intensity_transform(config: dict, is_train: bool) -> A.Compose:
    aug_cfg = config["augmentation"]
    normalization_cfg = (config.get("data") or {}).get("normalization") or {}
    mean = float(normalization_cfg.get("mean", 0.5))
    std = float(normalization_cfg.get("std", 0.5))
    transforms = []
    if is_train and aug_cfg.get("use_noise", False):
        transforms.append(
            A.GaussNoise(
                std_range=(aug_cfg["noise_std_min"], aug_cfg["noise_std_max"]),
                mean_range=(0.0, 0.0),
                p=aug_cfg["noise_p"],
            )
        )
    if is_train and aug_cfg.get("use_blur", False):
        transforms.append(
            A.GaussianBlur(
                blur_limit=(3, aug_cfg["blur_limit"]),
                p=aug_cfg["blur_p"],
            )
        )
    transforms.append(A.Normalize(mean=(mean,), std=(std,), max_pixel_value=255.0))
    return A.Compose(transforms)
