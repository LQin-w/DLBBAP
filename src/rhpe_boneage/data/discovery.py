from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}


@dataclass
class SplitSources:
    split: str
    image_dir: str
    csv_path: str
    roi_json_path: str
    id_width: int
    has_boneage: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_id(value: Any, width: int) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.zfill(width)
    return text


def _find_split_match(paths: list[Path], split: str) -> Path | None:
    aliases = SPLIT_ALIASES[split]
    for path in sorted(paths):
        name = path.name.lower()
        if any(alias in name for alias in aliases):
            return path
    return None


def _discover_annotation_dir(dataset_root: Path) -> Path:
    candidates = [dataset_root]
    candidates.extend([path for path in dataset_root.iterdir() if path.is_dir()])
    scored = []
    for path in candidates:
        csv_count = len(list(path.glob("*.csv")))
        json_count = len(list(path.glob("*.json")))
        if csv_count or json_count:
            scored.append((csv_count + json_count, path))
    if not scored:
        raise FileNotFoundError(f"在 {dataset_root} 下没有找到包含 csv/json 的标注目录。")
    return max(scored, key=lambda item: item[0])[1]


def _read_roi_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _index_roi_annotations(json_path: Path, id_width: int) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    data = _read_roi_json(json_path)
    image_meta = {entry["id"]: entry for entry in data["images"]}
    roi_map: dict[str, dict[str, Any]] = {}
    roi_counts = Counter()
    for ann in data["annotations"]:
        meta = image_meta[ann["image_id"]]
        file_id = _normalize_id(Path(meta["file_name"]).stem, id_width)
        roi_counts[file_id] += 1
        keypoints = ann.get("keypoints", [])
        if keypoints:
            keypoints = [
                [float(keypoints[index]), float(keypoints[index + 1]), float(keypoints[index + 2])]
                for index in range(0, len(keypoints), 3)
            ]
        roi_map[file_id] = {
            "bbox": [float(value) for value in ann.get("bbox", [0.0, 0.0, 0.0, 0.0])],
            "keypoints": keypoints,
            "num_keypoints": int(ann.get("num_keypoints", len(keypoints))),
            "image_width": int(meta.get("width", 0)),
            "image_height": int(meta.get("height", 0)),
            "image_id": int(meta["id"]),
            "file_name": meta["file_name"],
        }
    duplicate_map = {key: count for key, count in roi_counts.items() if count > 1}
    return roi_map, duplicate_map


def _check_image_readable(path: Path) -> str | None:
    try:
        with Image.open(path) as handle:
            handle.verify()
        return None
    except Exception as exc:  # pragma: no cover - 数据异常保护
        return str(exc)


def _rows_from_csv(csv_path: Path, id_width: int) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    dataframe = pd.read_csv(csv_path)
    rows = []
    has_boneage = "Boneage" in dataframe.columns
    duplicates = Counter(_normalize_id(value, id_width) for value in dataframe["ID"].tolist())
    duplicate_map = {key: count for key, count in duplicates.items() if count > 1}

    for row in dataframe.to_dict(orient="records"):
        item = dict(row)
        item["ID"] = _normalize_id(item["ID"], id_width)
        if "Male" in item:
            item["Male"] = int(str(item["Male"]).strip().lower() in {"1", "true", "t", "yes"})
        if "Chronological" in item and not pd.isna(item["Chronological"]):
            item["Chronological"] = float(item["Chronological"])
        if "Boneage" in item and not pd.isna(item["Boneage"]):
            item["Boneage"] = float(item["Boneage"])
        rows.append(item)
    return rows, duplicate_map, has_boneage


def _resolve_id_width(image_dir: Path, csv_path: Path, json_path: Path) -> int:
    widths: list[int] = []
    widths.extend(len(path.stem) for path in image_dir.glob("*") if path.is_file())
    try:
        dataframe = pd.read_csv(csv_path)
        widths.extend(len(str(value).strip()) for value in dataframe["ID"].tolist())
    except Exception:
        pass
    try:
        roi_data = _read_roi_json(json_path)
        widths.extend(len(Path(item["file_name"]).stem) for item in roi_data["images"])
    except Exception:
        pass
    widths = [width for width in widths if width > 0]
    return max(widths) if widths else 5


def build_split_records(
    split: str,
    image_dir: str | Path,
    csv_path: str | Path,
    roi_json_path: str | Path,
    verify_images: bool = True,
) -> tuple[SplitSources, list[dict[str, Any]], dict[str, Any]]:
    image_dir = Path(image_dir)
    csv_path = Path(csv_path)
    roi_json_path = Path(roi_json_path)

    id_width = _resolve_id_width(image_dir, csv_path, roi_json_path)
    rows, csv_duplicates, has_boneage = _rows_from_csv(csv_path, id_width)
    image_files = {}
    image_duplicates = Counter()
    for path in sorted(image_dir.glob("*")):
        if not path.is_file():
            continue
        stem = _normalize_id(path.stem, id_width)
        image_duplicates[stem] += 1
        image_files[stem] = path
    image_duplicates = {key: count for key, count in image_duplicates.items() if count > 1}

    roi_map, roi_duplicates = _index_roi_annotations(roi_json_path, id_width)

    csv_map = {row["ID"]: row for row in rows}
    all_ids = sorted(set(csv_map) | set(image_files) | set(roi_map))

    issues = {
        "missing_images": [],
        "missing_csv_records": [],
        "missing_roi_json": [],
        "duplicate_csv_ids": csv_duplicates,
        "duplicate_image_ids": image_duplicates,
        "duplicate_roi_ids": roi_duplicates,
        "unreadable_images": [],
        "image_name_mismatch": [],
    }
    records: list[dict[str, Any]] = []

    for sample_id in all_ids:
        image_path = image_files.get(sample_id)
        csv_row = csv_map.get(sample_id)
        roi_entry = roi_map.get(sample_id)

        if image_path is None:
            issues["missing_images"].append(sample_id)
            continue
        if csv_row is None:
            issues["missing_csv_records"].append(sample_id)
            continue
        if roi_entry is None:
            issues["missing_roi_json"].append(sample_id)
            continue
        if _normalize_id(image_path.stem, id_width) != sample_id:
            issues["image_name_mismatch"].append(image_path.name)
            continue
        if verify_images:
            error_text = _check_image_readable(image_path)
            if error_text is not None:
                issues["unreadable_images"].append(
                    {"id": sample_id, "path": str(image_path), "error": error_text}
                )
                continue

        record = {
            "id": sample_id,
            "split": split,
            "image_path": str(image_path),
            "male": int(csv_row.get("Male", 0)),
            "chronological": float(csv_row.get("Chronological", math.nan)),
            "boneage": float(csv_row["Boneage"]) if has_boneage else None,
            "has_boneage": bool(has_boneage),
            "bbox": roi_entry["bbox"],
            "keypoints": roi_entry["keypoints"],
            "num_keypoints": roi_entry["num_keypoints"],
            "image_width": roi_entry["image_width"],
            "image_height": roi_entry["image_height"],
            "roi_image_id": roi_entry["image_id"],
            "roi_file_name": roi_entry["file_name"],
            "csv_columns": list(csv_row.keys()),
        }
        records.append(record)

    report = {
        "split": split,
        "image_dir": str(image_dir),
        "csv_path": str(csv_path),
        "roi_json_path": str(roi_json_path),
        "id_width": id_width,
        "has_boneage": has_boneage,
        "matched_records": len(records),
        "issues": issues,
    }
    if roi_duplicates:
        duplicate_text = ", ".join(f"{sample_id}x{count}" for sample_id, count in sorted(roi_duplicates.items()))
        raise ValueError(f"ROI JSON 存在重复样本标注: split={split} | {duplicate_text}")
    sources = SplitSources(
        split=split,
        image_dir=str(image_dir),
        csv_path=str(csv_path),
        roi_json_path=str(roi_json_path),
        id_width=id_width,
        has_boneage=has_boneage,
    )
    return sources, records, report


def build_manual_split_records(
    split: str,
    image_dir: str | Path,
    csv_path: str | Path,
    roi_json_path: str | Path,
    verify_images: bool = True,
) -> tuple[SplitSources, list[dict[str, Any]], dict[str, Any]]:
    return build_split_records(
        split=split,
        image_dir=image_dir,
        csv_path=csv_path,
        roi_json_path=roi_json_path,
        verify_images=verify_images,
    )


def build_dataset_index(dataset_root: str | Path, verify_images: bool = True) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    annotation_dir = _discover_annotation_dir(dataset_root)
    csv_paths = list(annotation_dir.glob("*.csv"))
    json_paths = list(annotation_dir.glob("*.json"))
    image_dirs = [path for path in dataset_root.iterdir() if path.is_dir() and path != annotation_dir]

    split_payloads: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        image_dir = _find_split_match(image_dirs, split)
        csv_path = _find_split_match(csv_paths, split)
        roi_json_path = _find_split_match(json_paths, split)
        if not all([image_dir, csv_path, roi_json_path]):
            raise FileNotFoundError(
                f"自动发现失败: split={split}, image_dir={image_dir}, csv={csv_path}, json={roi_json_path}"
            )
        sources, records, report = build_split_records(
            split=split,
            image_dir=image_dir,
            csv_path=csv_path,
            roi_json_path=roi_json_path,
            verify_images=verify_images,
        )
        split_payloads[split] = {
            "sources": sources.to_dict(),
            "records": records,
        }
        reports[split] = report

    notes = {}
    readme_path = annotation_dir / "Readme.txt"
    if readme_path.exists():
        notes["annotation_readme"] = readme_path.read_text(encoding="utf-8", errors="ignore")

    return {
        "dataset_root": str(dataset_root),
        "annotation_dir": str(annotation_dir),
        "splits": split_payloads,
        "reports": reports,
        "notes": notes,
    }
