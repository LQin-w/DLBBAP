from __future__ import annotations

import copy
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import load_config, save_config
from ..data import (
    RHPEBoneAgeDataset,
    build_dataset_index,
    build_manual_split_records,
    compute_grayscale_mean_std,
    load_mean_std_cache,
    save_mean_std_cache,
)
from ..data.dataset import DatasetStats
from ..data.transforms import build_geometric_transform, build_image_intensity_transform
from ..models import build_model
from ..utils import detect_runtime, ensure_dir, seed_everything, setup_logger, suggest_dataloader_kwargs, write_json
from ..utils.device import log_device_probe, maybe_compile_model
from ..utils.io import timestamp
from ..utils.plots import generate_training_report
from .control import TrainingCancelledError, TrainingControl, raise_if_stop_requested
from .engine import run_epoch, unwrap_model
from .losses import build_loss
from .normalization import ScalarNormalizer


def _safe_metric_value(value: float | None) -> float:
    return value if value is not None else math.nan


def _phase_extra(phase: str) -> dict[str, str]:
    return {"phase": phase.upper()}


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.2f}s"


def _format_duration_clock(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_scalar(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        if math.isnan(float(value)):
            return "nan"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{precision}f}"


def _format_lr(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _format_memory(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}MB"


def _resolve_log_interval(config: dict[str, Any]) -> int:
    raw_value = (config.get("training") or {}).get("log_interval", 20)
    if raw_value is None:
        return 20
    return max(0, int(raw_value))


def _resolve_positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return max(1, int(value))


def _resolve_non_negative_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return max(0, int(value))


def _resolve_non_negative_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return max(0.0, float(value))


def _resolve_gradient_accumulation_steps(config: dict[str, Any]) -> int:
    return _resolve_positive_int((config.get("training") or {}).get("gradient_accumulation_steps"), default=1)


def _resolve_eval_interval(config: dict[str, Any], scheduler_name: str, logger) -> int:
    interval = _resolve_positive_int((config.get("training") or {}).get("eval_interval"), default=1)
    if scheduler_name == "plateau" and interval != 1:
        logger.warning("ReduceLROnPlateau 需要每次验证指标，已将 training.eval_interval 从 %s 自动改为 1。", interval)
        return 1
    return interval


def _resolve_save_interval(config: dict[str, Any]) -> int:
    return _resolve_positive_int((config.get("training") or {}).get("save_interval"), default=1)


def _resolve_warmup_settings(config: dict[str, Any], total_epochs: int) -> tuple[int, float]:
    training_cfg = config.get("training") or {}
    warmup_epochs = min(
        max(total_epochs - 1, 0),
        _resolve_non_negative_int(training_cfg.get("warmup_epochs"), default=0),
    )
    warmup_start_factor = float(training_cfg.get("warmup_start_factor", 0.2) or 0.2)
    warmup_start_factor = min(max(warmup_start_factor, 1e-4), 1.0)
    return warmup_epochs, warmup_start_factor


def _resolve_early_stopping(config: dict[str, Any]) -> tuple[int, float]:
    training_cfg = config.get("training") or {}
    patience = _resolve_non_negative_int(training_cfg.get("early_stopping_patience"), default=0)
    min_delta = _resolve_non_negative_float(training_cfg.get("early_stopping_min_delta"), default=0.0)
    return patience, min_delta


def _should_run_validation(epoch: int, total_epochs: int, eval_interval: int) -> bool:
    return epoch == total_epochs or (epoch % eval_interval == 0)


def _should_save_checkpoint(epoch: int, total_epochs: int, save_interval: int) -> bool:
    return epoch == total_epochs or (epoch % save_interval == 0)


def _metric_improved(
    current: float | None,
    best: float | None,
    min_delta: float,
) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    return current < (best - min_delta)


def _set_optimizer_lr(optimizer, lr_value: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_value


def _warmup_lr(base_lr: float, next_epoch: int, warmup_epochs: int, start_factor: float) -> float:
    progress = min(max(next_epoch / max(warmup_epochs, 1), 0.0), 1.0)
    scale = start_factor + (1.0 - start_factor) * progress
    return base_lr * scale


def _should_use_channels_last(config: dict[str, Any], device: torch.device) -> bool:
    runtime_cfg = config.get("runtime") or {}
    return device.type == "cuda" and bool(runtime_cfg.get("channels_last", True))


def _resolve_runtime_settings(config: dict[str, Any]) -> tuple[str, bool, bool]:
    runtime_cfg = config.get("runtime") or {}
    requested_device = str(runtime_cfg.get("device") or "cuda:0")
    allow_cpu_fallback = bool(runtime_cfg.get("allow_cpu_fallback", False))
    deterministic = bool(runtime_cfg.get("deterministic", False))
    return requested_device, allow_cpu_fallback, deterministic


def _resolve_training_started_at(control: TrainingControl | None) -> tuple[float, str]:
    if control is not None:
        started_at = control.get_run_started_at()
        if started_at is not None:
            return started_at, "ui"
        started_at = time.perf_counter()
        control.set_run_started_at(started_at)
        return started_at, "runner"
    return time.perf_counter(), "runner"


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    cv2.setNumThreads(0)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)


def _describe_loss(config: dict[str, Any]) -> str:
    loss_name = str(config["training"]["loss"]).lower()
    if loss_name == "smoothl1":
        return f"smoothl1(beta={config['training']['smooth_l1_beta']})"
    return loss_name


def _resolve_experiment_mode(config: dict[str, Any]) -> str:
    experiment_cfg = config.setdefault("experiment", {})
    mode = str(experiment_cfg.get("mode") or "enhanced").strip().lower()
    allowed = {"enhanced", "simba", "bonet_like"}
    if mode not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"experiment.mode 仅支持: {allowed_text}")
    experiment_cfg["mode"] = mode
    return mode


def _mode_profile(mode: str) -> dict[str, str]:
    profiles = {
        "enhanced": {
            "label": "enhanced (default)",
            "simba": "partial",
            "bonet": "partial",
            "description": "Engineering-enhanced multimodal framework.",
        },
        "simba": {
            "label": "simba",
            "simba": "partial (emphasized)",
            "bonet": "partial (supporting)",
            "description": "SIMBA-oriented declaration mode within the current engineering framework.",
        },
        "bonet_like": {
            "label": "bonet_like",
            "simba": "partial (supporting)",
            "bonet": "partial (emphasized)",
            "description": "BoNet-oriented declaration mode within the current engineering framework.",
        },
    }
    return profiles[mode]


def _log_running_mode(logger, config: dict[str, Any]) -> str:
    mode = _resolve_experiment_mode(config)
    profile = _mode_profile(mode)
    logger.info("Running mode: %s", profile["label"], extra=_phase_extra("SYSTEM"))
    logger.info("- SIMBA: %s", profile["simba"], extra=_phase_extra("SYSTEM"))
    logger.info("- BoNet: %s", profile["bonet"], extra=_phase_extra("SYSTEM"))
    logger.info("Mode note: %s", profile["description"], extra=_phase_extra("SYSTEM"))

    metadata_mode = str((config.get("model") or {}).get("metadata", {}).get("mode") or "mlp").lower()
    target_mode = str((config.get("model") or {}).get("target_mode") or "relative").lower()
    branch_mode = str((config.get("model") or {}).get("branch_mode") or "global_local").lower()
    metadata_cfg = (config.get("model") or {}).get("metadata") or {}
    metadata_enabled = bool(metadata_cfg.get("enabled", True))
    use_gender = bool(metadata_cfg.get("use_gender", True))
    use_chronological = bool(metadata_cfg.get("use_chronological", True))
    if metadata_enabled and not (use_gender or use_chronological):
        raise ValueError("model.metadata.enabled=true 时至少启用一种元信息输入。")
    if mode == "simba":
        if target_mode != "relative":
            logger.warning("experiment.mode=simba 但 model.target_mode=%s；这与 SIMBA 风格相对骨龄设定不一致。", target_mode)
        if not metadata_enabled:
            logger.warning("experiment.mode=simba 但 metadata 已关闭；这会弱化 SIMBA 的 identity marker 设计。")
        if metadata_mode == "mlp":
            logger.warning("experiment.mode=simba 但 model.metadata.mode=mlp；若要更贴近 SIMBA，请改为 simba_multiplier 或 simba_hybrid。")
        if not use_gender:
            logger.warning("experiment.mode=simba 但未启用 gender metadata；这与 SIMBA 的 identity markers 不一致。")
        if not use_chronological:
            logger.warning("experiment.mode=simba 但未启用 chronological metadata；这会削弱 SIMBA 的相对骨龄设定。")
    if mode == "bonet_like":
        if branch_mode == "global_only":
            logger.warning("experiment.mode=bonet_like 但 model.branch_mode=global_only；局部分支未启用，无法体现 ROI/keypoints/local patches。")
        if target_mode != "direct":
            logger.warning("experiment.mode=bonet_like 但 model.target_mode=%s；原始 BoNet 更接近直接骨龄回归。", target_mode)
        if not metadata_enabled or not use_gender:
            logger.warning("experiment.mode=bonet_like 但未保留 gender metadata；原始 BoNet 使用 gender 输入。")
        if metadata_enabled and use_chronological:
            logger.warning("experiment.mode=bonet_like 仍启用了 chronological metadata；这会引入更强的 SIMBA 风格 identity marker 偏向。")
    return mode


def _resolve_normalization_cache_path(
    config: dict[str, Any],
    payload: dict[str, Any],
    train_image_dir: str | Path,
) -> Path:
    normalization_cfg = (config.get("data") or {}).setdefault("normalization", {})
    raw_path = normalization_cfg.get("stats_path") or "train_mean_std.json"
    cache_path = Path(raw_path)
    if cache_path.is_absolute():
        return cache_path
    dataset_root = payload.get("dataset_root")
    if dataset_root:
        return Path(dataset_root) / cache_path
    return Path(train_image_dir).resolve().parent / cache_path


def _resolve_image_normalization(
    config: dict[str, Any],
    payload: dict[str, Any],
    checkpoint_state: dict[str, Any] | None,
    run_dir: Path,
    logger,
) -> dict[str, Any]:
    data_cfg = config.setdefault("data", {})
    normalization_cfg = data_cfg.setdefault("normalization", {})
    normalization_source = str(normalization_cfg.get("source") or "auto_train_stats").lower()
    if normalization_source not in {"auto_train_stats", "manual"}:
        raise ValueError("data.normalization.source 仅支持 auto_train_stats 或 manual")

    checkpoint_config = (checkpoint_state or {}).get("config") or {}
    checkpoint_normalization = ((checkpoint_config.get("data") or {}).get("normalization") or {})
    checkpoint_snapshot = (checkpoint_state or {}).get("image_normalization") or {}

    resolved_source = normalization_source
    mean = checkpoint_normalization.get("mean")
    std = checkpoint_normalization.get("std")
    if mean is None or std is None:
        mean = checkpoint_snapshot.get("mean")
        std = checkpoint_snapshot.get("std")
    if mean is not None and std is not None:
        resolved_source = "checkpoint"
    else:
        mean = normalization_cfg.get("mean")
        std = normalization_cfg.get("std")

    train_sources = ((payload.get("splits") or {}).get("train") or {}).get("sources") or {}
    train_image_dir = train_sources.get("image_dir")
    cache_path = _resolve_normalization_cache_path(config, payload, train_image_dir or data_cfg.get("dataset_root", "dataset"))
    stats_payload = None

    if mean is None or std is None:
        if normalization_source == "manual":
            raise ValueError("data.normalization.source=manual 时必须同时提供 data.normalization.mean 和 data.normalization.std")
        cached = load_mean_std_cache(cache_path) if cache_path.exists() else None
        current_train_dir = str(Path(train_image_dir).resolve()) if train_image_dir else None
        if cached is not None and (current_train_dir is None or cached.get("image_dir") == current_train_dir):
            stats_payload = cached
            resolved_source = "auto_train_stats(cache)"
        else:
            if not train_image_dir:
                raise ValueError("自动统计 mean/std 需要 train split 图像目录，但当前 payload 中未找到。")
            logger.info("正在统计 train 集灰度图 mean/std | image_dir=%s", train_image_dir, extra=_phase_extra("SYSTEM"))
            stats_payload = compute_grayscale_mean_std(train_image_dir)
            save_mean_std_cache(stats_payload, cache_path)
            resolved_source = "auto_train_stats(computed)"
        mean = stats_payload["mean"]
        std = stats_payload["std"]
    else:
        mean = float(mean)
        std = float(std)
        resolved_source = "manual" if normalization_source == "manual" else resolved_source
        stats_payload = {
            "image_dir": str(Path(train_image_dir).resolve()) if train_image_dir else None,
            "mean": mean,
            "std": std,
            "mean_255": mean * 255.0,
            "std_255": std * 255.0,
            "source": resolved_source,
        }

    if std <= 0:
        raise ValueError(f"图像归一化标准差必须大于 0，当前为 {std}")

    normalization_cfg["mean"] = float(mean)
    normalization_cfg["std"] = float(std)
    normalization_cfg["resolved_source"] = resolved_source
    normalization_cfg["stats_path"] = str(cache_path)

    stats_payload = dict(stats_payload or {})
    stats_payload["source"] = resolved_source
    stats_payload["stats_path"] = str(cache_path)
    write_json(stats_payload, run_dir / "image_normalization.json")
    logger.info(
        "Image normalization | source=%s | mean=%.8f | std=%.8f | cache=%s",
        resolved_source,
        float(mean),
        float(std),
        cache_path,
        extra=_phase_extra("SYSTEM"),
    )
    return stats_payload


def _describe_target(config: dict[str, Any]) -> str:
    model_cfg = config.get("model") or {}
    target_mode = str(model_cfg.get("target_mode") or "relative").lower()
    if target_mode == "relative":
        direction = str(model_cfg.get("relative_target_direction") or "boneage_minus_chronological")
        if direction == "chronological_minus_boneage":
            return "relative_age = Chronological - Boneage -> final_boneage = Chronological - predicted_relative_age"
        return "relative_age = Boneage - Chronological -> final_boneage = Chronological + predicted_relative_age"
    return "direct boneage regression"


def _resolve_metadata_inputs(config: dict[str, Any]) -> tuple[bool, bool, bool]:
    metadata_cfg = ((config.get("model") or {}).get("metadata") or {})
    enabled = bool(metadata_cfg.get("enabled", True))
    use_gender = bool(metadata_cfg.get("use_gender", True))
    use_chronological = bool(metadata_cfg.get("use_chronological", True))
    return enabled, use_gender, use_chronological


def _describe_metadata_inputs(config: dict[str, Any]) -> str:
    metadata_enabled, use_gender, use_chronological = _resolve_metadata_inputs(config)
    if not metadata_enabled:
        return "disabled"
    selected = []
    if use_gender:
        selected.append("male")
    if use_chronological:
        selected.append("chronological")
    return ", ".join(selected) if selected else "none"


def _describe_input_modalities(config: dict[str, Any]) -> list[str]:
    model_cfg = config.get("model") or {}
    branch_mode = str(model_cfg.get("branch_mode") or "global_local").lower()
    metadata_enabled, use_gender, use_chronological = _resolve_metadata_inputs(config)
    modalities = ["grayscale_global_image"]
    if bool((model_cfg.get("heatmap_guidance") or {}).get("enabled", False)):
        modalities.append("global_roi_heatmap")
    if branch_mode in {"global_local", "local_only"}:
        local_mode = str(((model_cfg.get("local_branch") or {}).get("mode") or "patch_heatmap")).lower()
        if local_mode in {"patch", "patch_heatmap"}:
            modalities.append("local_patches")
        if local_mode in {"heatmap", "patch_heatmap"}:
            modalities.append("local_heatmaps")
        modalities.append("roi_geometry_vector")
    if metadata_enabled:
        if use_gender:
            modalities.append("male")
        if use_chronological:
            modalities.append("chronological")
    return modalities


def _describe_model_type(config: dict[str, Any]) -> str:
    model_cfg = config.get("model") or {}
    ensemble_mode = str(model_cfg.get("ensemble_mode") or "ensemble").lower()
    if ensemble_mode == "ensemble":
        return f"EnsembleBoneAgeModel({model_cfg.get('resnet_name')} + {model_cfg.get('efficientnet_name')})"
    if ensemble_mode == "resnet":
        return f"SingleBackboneModel({model_cfg.get('resnet_name')})"
    return f"SingleBackboneModel({model_cfg.get('efficientnet_name')})"


def _describe_augmentation_profile(config: dict[str, Any]) -> dict[str, Any]:
    aug_cfg = config.get("augmentation") or {}
    return {
        "affine_p": float(aug_cfg.get("affine_p", 0.0) or 0.0),
        "rotation_limit": float(aug_cfg.get("rotation_limit", 0.0) or 0.0),
        "translate_limit": float(aug_cfg.get("translate_limit", 0.0) or 0.0),
        "scale_limit": float(aug_cfg.get("scale_limit", 0.0) or 0.0),
        "shear_limit": float(aug_cfg.get("shear_limit", 0.0) or 0.0),
        "horizontal_flip": bool(aug_cfg.get("horizontal_flip", False)),
        "horizontal_flip_p": float(aug_cfg.get("horizontal_flip_p", 0.5) or 0.5),
        "use_noise": bool(aug_cfg.get("use_noise", False)),
        "noise_p": float(aug_cfg.get("noise_p", 0.0) or 0.0),
        "use_blur": bool(aug_cfg.get("use_blur", False)),
        "blur_p": float(aug_cfg.get("blur_p", 0.0) or 0.0),
    }


def _log_epoch_header(
    logger,
    config: dict[str, Any],
    optimizer,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    log_interval: int,
) -> None:
    training_cfg = config["training"]
    grad_accum_steps = int(training_cfg.get("gradient_accumulation_steps", 1) or 1)
    effective_batch_size = int(training_cfg["batch_size"]) * grad_accum_steps
    logger.info("%s", "=" * 108, extra=_phase_extra("SYSTEM"))
    logger.info("Epoch %d/%d started", epoch, total_epochs, extra=_phase_extra("SYSTEM"))
    logger.info(
        "Hyperparams | lr=%s | train_batch_size=%s | effective_train_batch_size=%s | val_batch_size=%s | test_batch_size=%s | optimizer=%s | scheduler=%s | weight_decay=%s | loss=%s | device=%s | amp=%s | grad_accum=%s | grad_clip=%s",
        _format_lr(optimizer.param_groups[0]["lr"]),
        training_cfg["batch_size"],
        effective_batch_size,
        training_cfg["val_batch_size"],
        training_cfg.get("test_batch_size"),
        str(training_cfg["optimizer"]).lower(),
        str(training_cfg.get("scheduler") or "none").lower(),
        training_cfg.get("weight_decay"),
        _describe_loss(config),
        device,
        use_amp,
        grad_accum_steps,
        training_cfg.get("gradient_clip"),
        extra=_phase_extra("SYSTEM"),
    )


def _log_learning_rate_update(
    logger,
    scheduler_name: str,
    epoch: int,
    total_epochs: int,
    previous_lr: float,
    current_lr: float,
) -> None:
    if math.isclose(previous_lr, current_lr, rel_tol=1e-12, abs_tol=1e-12):
        return
    logger.info(
        "Learning rate updated | epoch=%d/%d | scheduler=%s | from=%s | to=%s",
        epoch,
        total_epochs,
        scheduler_name,
        _format_lr(previous_lr),
        _format_lr(current_lr),
        extra=_phase_extra("SYSTEM"),
    )


def _log_epoch_timing(
    logger,
    epoch: int,
    total_epochs: int,
    train_stats: dict[str, Any],
    eval_stats: dict[str, Any],
    epoch_total_time: float,
) -> None:
    logger.info(
        "Epoch %d/%d finished | train_time=%s | eval_time=%s | epoch_total_time=%s | train_data_time=%s | eval_data_time=%s | train_transfer_time=%s | eval_transfer_time=%s | train_compute_time=%s | eval_compute_time=%s | train_samples_per_sec=%s | eval_samples_per_sec=%s | train_gpu_peak_alloc=%s | eval_gpu_peak_alloc=%s | train_gpu_peak_reserved=%s | eval_gpu_peak_reserved=%s | train_avg_batch_time=%s | eval_avg_batch_time=%s | train_fastest_batch=%s | train_slowest_batch=%s | eval_fastest_batch=%s | eval_slowest_batch=%s",
        epoch,
        total_epochs,
        _format_seconds(train_stats.get("total_time")),
        _format_seconds(eval_stats.get("total_time")),
        _format_seconds(epoch_total_time),
        _format_seconds(train_stats.get("data_time")),
        _format_seconds(eval_stats.get("data_time")),
        _format_seconds(train_stats.get("transfer_time")),
        _format_seconds(eval_stats.get("transfer_time")),
        _format_seconds(train_stats.get("compute_time")),
        _format_seconds(eval_stats.get("compute_time")),
        _format_scalar(train_stats.get("samples_per_second"), precision=2),
        _format_scalar(eval_stats.get("samples_per_second"), precision=2),
        _format_memory(train_stats.get("max_allocated_mb")),
        _format_memory(eval_stats.get("max_allocated_mb")),
        _format_memory(train_stats.get("max_reserved_mb")),
        _format_memory(eval_stats.get("max_reserved_mb")),
        _format_seconds(train_stats.get("avg_batch_time")),
        _format_seconds(eval_stats.get("avg_batch_time")),
        _format_seconds(train_stats.get("min_batch_time")),
        _format_seconds(train_stats.get("max_batch_time")),
        _format_seconds(eval_stats.get("min_batch_time")),
        _format_seconds(eval_stats.get("max_batch_time")),
        extra=_phase_extra("SYSTEM"),
    )


def _validate_best_metric(metric_name: str) -> str:
    allowed = {"loss", "mae", "mad"}
    normalized = str(metric_name).strip().lower()
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"training.best_metric 仅支持: {allowed_text}")
    return normalized


def _log_epoch_metrics(
    logger,
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any] | None,
    lr_start: float,
    lr_end: float,
    eval_ran: bool,
    epoch_time_seconds: float,
    total_elapsed_seconds: float,
    best_updated: bool,
) -> None:
    if not eval_ran or val_metrics is None:
        logger.info(
            "Epoch %d/%d metrics | train_loss=%s | train_final_mae=%s | train_relative_mae=%s | train_final_mad=%s | train_relative_mad=%s | lr_start=%s | lr_end=%s | epoch_time=%s | total_elapsed=%s | best_updated=%s | validation=skipped",
            epoch,
            total_epochs,
            _format_scalar(train_metrics.get("loss")),
            _format_scalar(train_metrics.get("final_mae")),
            _format_scalar(train_metrics.get("relative_mae")),
            _format_scalar(train_metrics.get("final_mad")),
            _format_scalar(train_metrics.get("relative_mad")),
            _format_lr(lr_start),
            _format_lr(lr_end),
            _format_duration_clock(epoch_time_seconds),
            _format_duration_clock(total_elapsed_seconds),
            best_updated,
            extra=_phase_extra("SYSTEM"),
        )
        return
    logger.info(
        "Epoch %d/%d metrics | train_loss=%s | val_loss=%s | train_final_mae=%s | val_final_mae=%s | train_relative_mae=%s | val_relative_mae=%s | train_final_mad=%s | val_final_mad=%s | lr_start=%s | lr_end=%s | epoch_time=%s | total_elapsed=%s | best_updated=%s",
        epoch,
        total_epochs,
        _format_scalar(train_metrics.get("loss")),
        _format_scalar(val_metrics.get("loss")),
        _format_scalar(train_metrics.get("final_mae")),
        _format_scalar(val_metrics.get("final_mae")),
        _format_scalar(train_metrics.get("relative_mae")),
        _format_scalar(val_metrics.get("relative_mae")),
        _format_scalar(train_metrics.get("final_mad")),
        _format_scalar(val_metrics.get("final_mad")),
        _format_lr(lr_start),
        _format_lr(lr_end),
        _format_duration_clock(epoch_time_seconds),
        _format_duration_clock(total_elapsed_seconds),
        best_updated,
        extra=_phase_extra("SYSTEM"),
    )


def _log_checkpoint_saved(
    logger,
    *,
    label: str,
    checkpoint_path: Path,
    epoch: int,
    total_epochs: int,
) -> None:
    logger.info(
        "Checkpoint saved | kind=%s | epoch=%d/%d | path=%s",
        label,
        epoch,
        total_epochs,
        checkpoint_path,
        extra=_phase_extra("SYSTEM"),
    )


def _log_dataloader_kwargs(logger, loader_kwargs: dict[str, Any]) -> None:
    logger.info(
        "DataLoader | num_workers=%s | pin_memory=%s | persistent_workers=%s | prefetch_factor=%s",
        loader_kwargs.get("num_workers"),
        loader_kwargs.get("pin_memory"),
        loader_kwargs.get("persistent_workers"),
        loader_kwargs.get("prefetch_factor"),
    )


def _log_runtime_info(logger, runtime) -> None:
    logger.info(
        "运行环境 | python=%s | executable=%s | platform=%s | is_wsl=%s | filesystem_encoding=%s | preferred_encoding=%s | stdout_encoding=%s",
        runtime.python,
        runtime.python_executable,
        runtime.platform,
        runtime.is_wsl,
        runtime.filesystem_encoding,
        runtime.preferred_encoding,
        runtime.stdout_encoding,
        extra=_phase_extra("SYSTEM"),
    )
    logger.info(
        "依赖版本 | torch=%s | torchvision=%s | torchaudio=%s | numpy=%s | pandas=%s | cuda_build=%s | torch_cuda_built=%s | amp_available=%s | compile_available=%s | triton_available=%s",
        runtime.torch_version,
        runtime.torchvision_version,
        runtime.torchaudio_version,
        runtime.numpy_version,
        runtime.pandas_version,
        runtime.cuda_build,
        runtime.torch_cuda_built,
        runtime.amp_available,
        runtime.compile_available,
        runtime.compile_triton_available,
        extra=_phase_extra("SYSTEM"),
    )
    logger.info(
        "设备环境 | requested_device=%s | selected_device=%s | cuda_available=%s | device_count=%s | gpus=%s | nvidia_smi=%s | device_nodes=%s | deterministic=%s | cudnn_benchmark=%s | tf32_matmul=%s | tf32_cudnn=%s | matmul_precision=%s",
        runtime.requested_device,
        runtime.selected_device,
        runtime.cuda_available,
        runtime.device_count,
        runtime.device_names,
        runtime.nvidia_smi_summary,
        runtime.device_nodes,
        runtime.deterministic,
        runtime.cudnn_benchmark,
        runtime.tf32_matmul,
        runtime.tf32_cudnn,
        runtime.float32_matmul_precision,
        extra=_phase_extra("SYSTEM"),
    )
    if runtime.cuda_diagnostic:
        logger.warning("CUDA 诊断 | %s", runtime.cuda_diagnostic, extra=_phase_extra("SYSTEM"))


def _numeric_range(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _build_dataset_summary(payload: dict[str, Any], datasets: dict[str, RHPEBoneAgeDataset]) -> dict[str, Any]:
    summary = {
        "dataset_root": payload.get("dataset_root"),
        "global_loader": "cv2.IMREAD_GRAYSCALE",
        "global_channels": 1,
        "splits": {},
    }

    for split, split_payload in payload["splits"].items():
        source_records = split_payload["records"]
        used_records = datasets[split].records if split in datasets else []
        sources = split_payload.get("sources") or {}
        first_record = source_records[0] if source_records else {}
        csv_columns = list(first_record.get("csv_columns", []))
        sex_values = Counter(int(record.get("male", 0)) for record in source_records)
        chronological_values = [
            float(record["chronological"])
            for record in source_records
            if record.get("chronological") is not None and not pd.isna(record.get("chronological"))
        ]
        boneage_values = [
            float(record["boneage"])
            for record in source_records
            if record.get("has_boneage") and record.get("boneage") is not None and not pd.isna(record.get("boneage"))
        ]
        summary["splits"][split] = {
            "matched_records": len(source_records),
            "used_records": len(used_records),
            "csv_columns": csv_columns,
            "sex_field": "Male" if "Male" in csv_columns else None,
            "chronological_field": "Chronological" if "Chronological" in csv_columns else None,
            "boneage_field": "Boneage" if "Boneage" in csv_columns else None,
            "has_boneage_targets": any(record.get("has_boneage") for record in source_records),
            "missing_target_count_in_used_records": sum(1 for record in used_records if not record.get("has_boneage")),
            "sex_distribution": {str(key): int(value) for key, value in sorted(sex_values.items())},
            "chronological_range": _numeric_range(chronological_values),
            "boneage_range": _numeric_range(boneage_values),
            "image_extensions": sorted({Path(record["image_path"]).suffix.lower() for record in source_records}),
            "sample_id_preview": [record["id"] for record in used_records[:5]],
            "image_dir": sources.get("image_dir"),
            "csv_path": sources.get("csv_path"),
            "roi_json_path": sources.get("roi_json_path"),
            "id_width": sources.get("id_width"),
        }
    return summary


def _log_dataset_summary(logger, summary: dict[str, Any]) -> None:
    logger.info(
        "图像输入 | loader=%s | channels=%s",
        summary.get("global_loader"),
        summary.get("global_channels"),
        extra=_phase_extra("SYSTEM"),
    )
    for split, split_summary in summary.get("splits", {}).items():
        logger.info(
            "数据摘要 | split=%s | matched=%s | used=%s | csv_columns=%s | sex_field=%s | chronological_field=%s | boneage_field=%s | image_ext=%s | missing_target_in_used=%s | chronological_range=%s | boneage_range=%s",
            split,
            split_summary.get("matched_records"),
            split_summary.get("used_records"),
            split_summary.get("csv_columns"),
            split_summary.get("sex_field"),
            split_summary.get("chronological_field"),
            split_summary.get("boneage_field"),
            split_summary.get("image_extensions"),
            split_summary.get("missing_target_count_in_used_records"),
            split_summary.get("chronological_range"),
            split_summary.get("boneage_range"),
            extra=_phase_extra("SYSTEM"),
        )


def _build_config_summary(
    config: dict[str, Any],
    runtime,
    datasets: dict[str, Any],
) -> dict[str, Any]:
    data_cfg = config.get("data") or {}
    model_cfg = config.get("model") or {}
    training_cfg = config.get("training") or {}
    metadata_cfg = model_cfg.get("metadata") or {}
    cbam_cfg = model_cfg.get("cbam") or {}
    heatmap_guidance_cfg = model_cfg.get("heatmap_guidance") or {}
    local_branch_cfg = model_cfg.get("local_branch") or {}
    normalization_cfg = (config.get("data") or {}).get("normalization") or {}
    return {
        "experiment_mode": str((config.get("experiment") or {}).get("mode") or "enhanced"),
        "model_type": _describe_model_type(config),
        "metadata_mode": str(metadata_cfg.get("mode") or "mlp"),
        "metadata_inputs": _describe_metadata_inputs(config),
        "input_modalities": _describe_input_modalities(config),
        "target_type": _describe_target(config),
        "dataset_size": {split: len(dataset) for split, dataset in datasets.items()},
        "device": runtime.selected_device,
        "structure": {
            "ensemble_mode": str(model_cfg.get("ensemble_mode") or "ensemble"),
            "resnet_name": str(model_cfg.get("resnet_name") or "resnet18"),
            "efficientnet_name": str(model_cfg.get("efficientnet_name") or "efficientnet_b0"),
            "pretrained": bool(model_cfg.get("pretrained", False)),
            "branch_mode": str(model_cfg.get("branch_mode") or "global_local"),
            "local_branch_mode": str(local_branch_cfg.get("mode") or "patch_heatmap"),
            "target_mode": str(model_cfg.get("target_mode") or "relative"),
            "relative_target_direction": str(model_cfg.get("relative_target_direction") or "boneage_minus_chronological"),
            "metadata_enabled": bool(metadata_cfg.get("enabled", True)),
            "metadata_mode": str(metadata_cfg.get("mode") or "mlp"),
            "metadata_use_gender": bool(metadata_cfg.get("use_gender", True)),
            "metadata_use_chronological": bool(metadata_cfg.get("use_chronological", True)),
            "heatmap_guidance": bool(heatmap_guidance_cfg.get("enabled", False)),
            "cbam_enabled": bool(cbam_cfg.get("enabled", False)),
            "cbam_global": bool(cbam_cfg.get("global_branch", False)),
            "cbam_local": bool(cbam_cfg.get("local_branch", False)),
        },
        "data_strategy": {
            "input_size": int(data_cfg.get("input_size", 0) or 0),
            "local_patch_size": int(data_cfg.get("local_patch_size", 0) or 0),
            "global_crop_mode": str(data_cfg.get("global_crop_mode") or "bbox"),
            "global_crop_margin_ratio": float(data_cfg.get("global_crop_margin_ratio", 0.0) or 0.0),
            "heatmap_sigma_ratio": float(data_cfg.get("heatmap_sigma_ratio", 0.0) or 0.0),
            "heatmap_sigma_min": float(data_cfg.get("heatmap_sigma_min", 0.0) or 0.0),
            "normalization_source": normalization_cfg.get("resolved_source") or normalization_cfg.get("source"),
        },
        "optimization": {
            "epochs": int(training_cfg.get("epochs", 0) or 0),
            "batch_size": int(training_cfg.get("batch_size", 0) or 0),
            "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1) or 1),
            "optimizer": str(training_cfg.get("optimizer") or "adamw").lower(),
            "lr": float(training_cfg.get("lr", 0.0) or 0.0),
            "weight_decay": float(training_cfg.get("weight_decay", 0.0) or 0.0),
            "scheduler": str(training_cfg.get("scheduler") or "none").lower(),
            "scheduler_factor": float(training_cfg.get("scheduler_factor", 0.5) or 0.5),
            "scheduler_patience": int(training_cfg.get("scheduler_patience", 2) or 2),
            "warmup_epochs": int(training_cfg.get("warmup_epochs", 0) or 0),
            "loss": _describe_loss(config),
            "best_metric": str(training_cfg.get("best_metric") or "mae").lower(),
            "amp_requested": bool(training_cfg.get("amp", False)),
            "compile_requested": bool(training_cfg.get("compile", False)),
            "compile_mode": str(training_cfg.get("compile_mode") or "default"),
        },
        "augmentation": _describe_augmentation_profile(config),
        "normalization": {
            "mean": normalization_cfg.get("mean"),
            "std": normalization_cfg.get("std"),
            "source": normalization_cfg.get("resolved_source") or normalization_cfg.get("source"),
        },
    }


def _log_config_summary(logger, summary: dict[str, Any]) -> None:
    structure = summary["structure"]
    data_strategy = summary["data_strategy"]
    optimization = summary["optimization"]
    augmentation = summary["augmentation"]
    logger.info("CONFIG SUMMARY:", extra=_phase_extra("SYSTEM"))
    logger.info("- model type: %s", summary["model_type"], extra=_phase_extra("SYSTEM"))
    logger.info(
        "- metadata mode: %s | inputs: %s",
        summary["metadata_mode"],
        summary["metadata_inputs"],
        extra=_phase_extra("SYSTEM"),
    )
    logger.info(
        "- structure switches: ensemble=%s | branch=%s | local_mode=%s | target=%s | metadata=%s(%s;g=%s,c=%s) | heatmap_guidance=%s | cbam=%s[g=%s,l=%s]",
        structure["ensemble_mode"],
        structure["branch_mode"],
        structure["local_branch_mode"],
        structure["target_mode"],
        structure["metadata_enabled"],
        structure["metadata_mode"],
        structure["metadata_use_gender"],
        structure["metadata_use_chronological"],
        structure["heatmap_guidance"],
        structure["cbam_enabled"],
        structure["cbam_global"],
        structure["cbam_local"],
        extra=_phase_extra("SYSTEM"),
    )
    logger.info(
        "- backbones: resnet=%s | efficientnet=%s | pretrained=%s",
        structure["resnet_name"],
        structure["efficientnet_name"],
        structure["pretrained"],
        extra=_phase_extra("SYSTEM"),
    )
    logger.info("- input modalities: %s", ", ".join(summary["input_modalities"]), extra=_phase_extra("SYSTEM"))
    logger.info("- target type: %s", summary["target_type"], extra=_phase_extra("SYSTEM"))
    logger.info(
        "- data strategy: input_size=%s | local_patch=%s | crop=%s | crop_margin=%s | sigma_ratio=%s | sigma_min=%s",
        data_strategy["input_size"],
        data_strategy["local_patch_size"],
        data_strategy["global_crop_mode"],
        _format_scalar(data_strategy["global_crop_margin_ratio"], precision=4),
        _format_scalar(data_strategy["heatmap_sigma_ratio"], precision=4),
        _format_scalar(data_strategy["heatmap_sigma_min"], precision=2),
        extra=_phase_extra("SYSTEM"),
    )
    logger.info(
        "- optimization: epochs=%s | batch=%s | grad_accum=%s | optimizer=%s | lr=%s | weight_decay=%s | scheduler=%s[factor=%s,patience=%s] | warmup_epochs=%s | amp=%s | compile=%s(mode=%s) | loss=%s | best_metric=%s",
        optimization["epochs"],
        optimization["batch_size"],
        optimization["gradient_accumulation_steps"],
        optimization["optimizer"],
        _format_lr(optimization["lr"]),
        optimization["weight_decay"],
        optimization["scheduler"],
        _format_scalar(optimization["scheduler_factor"], precision=3),
        optimization["scheduler_patience"],
        optimization["warmup_epochs"],
        optimization["amp_requested"],
        optimization["compile_requested"],
        optimization["compile_mode"],
        optimization["loss"],
        optimization["best_metric"],
        extra=_phase_extra("SYSTEM"),
    )
    logger.info(
        "- augmentation: affine_p=%s | flip=%s(p=%s) | noise=%s(p=%s) | blur=%s(p=%s)",
        _format_scalar(augmentation["affine_p"], precision=2),
        augmentation["horizontal_flip"],
        _format_scalar(augmentation["horizontal_flip_p"], precision=2),
        augmentation["use_noise"],
        _format_scalar(augmentation["noise_p"], precision=2),
        augmentation["use_blur"],
        _format_scalar(augmentation["blur_p"], precision=2),
        extra=_phase_extra("SYSTEM"),
    )
    logger.info("- dataset size: %s", summary["dataset_size"], extra=_phase_extra("SYSTEM"))
    logger.info("- device: %s", summary["device"], extra=_phase_extra("SYSTEM"))
    logger.info(
        "- image normalization: source=%s mean=%s std=%s",
        summary["normalization"]["source"],
        _format_scalar(summary["normalization"]["mean"], precision=8),
        _format_scalar(summary["normalization"]["std"], precision=8),
        extra=_phase_extra("SYSTEM"),
    )


def _prepare_artifact_dirs(run_dir: Path) -> dict[str, Path]:
    return {
        "model_dir": ensure_dir(run_dir / "model"),
        "plots_dir": ensure_dir(run_dir / "plots"),
    }


def _build_effective_params_payload(
    config: dict[str, Any],
    runtime,
    datasets: dict[str, Any],
    loader_kwargs: dict[str, Any],
    use_amp: bool,
    use_channels_last: bool,
    compile_info: Any | None = None,
) -> dict[str, Any]:
    training_cfg = config["training"]
    runtime_cfg = config.get("runtime") or {}
    model_cfg = config.get("model") or {}
    metadata_cfg = model_cfg.get("metadata") or {}
    cbam_cfg = model_cfg.get("cbam") or {}
    local_branch_cfg = model_cfg.get("local_branch") or {}
    augmentation_cfg = config.get("augmentation") or {}
    normalization_cfg = (config.get("data") or {}).get("normalization") or {}
    grad_accum_steps = _resolve_gradient_accumulation_steps(config)
    total_epochs = int(training_cfg["epochs"])
    warmup_epochs, warmup_start_factor = _resolve_warmup_settings(config, total_epochs)
    early_stop_patience, early_stop_min_delta = _resolve_early_stopping(config)
    compile_requested = bool(training_cfg.get("compile", False))
    compile_mode = str(training_cfg.get("compile_mode") or "default")
    compile_available = bool(runtime.compile_available)
    if runtime.selected_device.startswith("cuda"):
        compile_available = compile_available and bool(runtime.compile_triton_available)
    compile_actually_used = False
    compile_reason = "not attempted in this entry point" if compile_requested else "disabled by config"
    compile_cudagraphs = None
    if compile_info is not None:
        compile_requested = bool(getattr(compile_info, "requested", compile_requested))
        compile_available = bool(getattr(compile_info, "available", compile_available))
        compile_actually_used = bool(getattr(compile_info, "actually_used", False))
        compile_reason = getattr(compile_info, "reason", compile_reason)
        compile_cudagraphs = getattr(compile_info, "cudagraphs_enabled", None)
        compile_mode = str(getattr(compile_info, "mode", compile_mode) or compile_mode)

    return {
        "experiment_mode": str((config.get("experiment") or {}).get("mode") or "enhanced"),
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "device": runtime.selected_device,
        "requested_device": runtime.requested_device,
        "deterministic": runtime.deterministic,
        "channels_last": use_channels_last,
        "epochs": total_epochs,
        "batch_size": int(training_cfg["batch_size"]),
        "val_batch_size": int(training_cfg["val_batch_size"]),
        "test_batch_size": int(training_cfg["test_batch_size"]),
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_train_batch_size": int(training_cfg["batch_size"]) * grad_accum_steps,
        "optimizer": str(training_cfg["optimizer"]).lower(),
        "lr": float(training_cfg["lr"]),
        "weight_decay": float(training_cfg["weight_decay"]),
        "scheduler": str(training_cfg.get("scheduler") or "none").lower(),
        "scheduler_factor": float(training_cfg.get("scheduler_factor", 0.5) or 0.5),
        "scheduler_patience": int(training_cfg.get("scheduler_patience", 2) or 2),
        "warmup_epochs": warmup_epochs,
        "warmup_start_factor": warmup_start_factor,
        "min_lr": float(training_cfg["min_lr"]),
        "loss": _describe_loss(config),
        "best_metric": str(training_cfg.get("best_metric") or "mae").lower(),
        "amp_requested": bool(training_cfg.get("amp", False)),
        "amp_actually_used": use_amp,
        "compile_requested": compile_requested,
        "compile_available": compile_available,
        "compile_actually_used": compile_actually_used,
        "compile_reason": compile_reason,
        "compile_mode": compile_mode,
        "compile_cudagraphs": compile_cudagraphs,
        "eval_interval": _resolve_positive_int(training_cfg.get("eval_interval"), default=1),
        "save_interval": _resolve_positive_int(training_cfg.get("save_interval"), default=1),
        "early_stopping_patience": early_stop_patience,
        "early_stopping_min_delta": early_stop_min_delta,
        "num_workers": loader_kwargs.get("num_workers"),
        "pin_memory": loader_kwargs.get("pin_memory"),
        "persistent_workers": loader_kwargs.get("persistent_workers"),
        "prefetch_factor": loader_kwargs.get("prefetch_factor"),
        "verify_images": bool(config["data"]["verify_images"]),
        "input_size": int(config["data"]["input_size"]),
        "local_patch_size": int(config["data"]["local_patch_size"]),
        "global_crop_mode": str(config["data"].get("global_crop_mode")),
        "global_crop_margin_ratio": float(config["data"].get("global_crop_margin_ratio", 0.0) or 0.0),
        "heatmap_sigma_ratio": float(config["data"].get("heatmap_sigma_ratio", 0.0) or 0.0),
        "heatmap_sigma_min": float(config["data"].get("heatmap_sigma_min", 0.0) or 0.0),
        "model_type": _describe_model_type(config),
        "ensemble_mode": str(model_cfg.get("ensemble_mode") or "ensemble"),
        "resnet_name": str(model_cfg.get("resnet_name") or "resnet18"),
        "efficientnet_name": str(model_cfg.get("efficientnet_name") or "efficientnet_b0"),
        "pretrained": bool(model_cfg.get("pretrained", False)),
        "branch_mode": str(model_cfg.get("branch_mode") or "global_local"),
        "local_branch_mode": str(local_branch_cfg.get("mode") or "patch_heatmap"),
        "metadata_enabled": bool(metadata_cfg.get("enabled", True)),
        "input_modalities": _describe_input_modalities(config),
        "target_type": _describe_target(config),
        "metadata_mode": str(metadata_cfg.get("mode") or "mlp"),
        "metadata_inputs": _describe_metadata_inputs(config),
        "metadata_use_gender": bool(metadata_cfg.get("use_gender", True)),
        "metadata_use_chronological": bool(metadata_cfg.get("use_chronological", True)),
        "heatmap_guidance": bool((model_cfg.get("heatmap_guidance") or {}).get("enabled", False)),
        "cbam_enabled": bool(cbam_cfg.get("enabled", False)),
        "cbam_global": bool(cbam_cfg.get("global_branch", False)),
        "cbam_local": bool(cbam_cfg.get("local_branch", False)),
        "horizontal_flip": bool(augmentation_cfg.get("horizontal_flip", False)),
        "horizontal_flip_p": float(augmentation_cfg.get("horizontal_flip_p", 0.5) or 0.5),
        "use_noise": bool(augmentation_cfg.get("use_noise", False)),
        "noise_p": float(augmentation_cfg.get("noise_p", 0.0) or 0.0),
        "use_blur": bool(augmentation_cfg.get("use_blur", False)),
        "blur_p": float(augmentation_cfg.get("blur_p", 0.0) or 0.0),
        "normalization_mean": normalization_cfg.get("mean"),
        "normalization_std": normalization_cfg.get("std"),
        "normalization_source": normalization_cfg.get("resolved_source") or normalization_cfg.get("source"),
        "allow_cpu_fallback": bool(runtime_cfg.get("allow_cpu_fallback", False)),
    }


def _log_effective_params(logger, payload: dict[str, Any]) -> None:
    logger.info(
        "Effective params | mode=%s | device=%s | epochs=%s | batch_size=%s | effective_batch_size=%s | optimizer=%s | lr=%s | weight_decay=%s | scheduler=%s[factor=%s,patience=%s] | warmup_epochs=%s | amp=%s->%s | compile=%s/%s/%s(mode=%s) | channels_last=%s | eval_interval=%s | early_stopping_patience=%s | num_workers=%s | pin_memory=%s | persistent_workers=%s | prefetch_factor=%s | verify_images=%s | ensemble=%s | branch=%s | local_mode=%s | metadata=%s(%s;g=%s,c=%s;inputs=%s) | heatmap_guidance=%s | cbam=%s[g=%s,l=%s] | input_size=%s | local_patch_size=%s | crop=%s(margin=%s) | sigma=(%s,%s) | target_type=%s | normalization=(%s,%s,%s) | dataset_sizes=%s",
        payload["experiment_mode"],
        payload["device"],
        payload["epochs"],
        payload["batch_size"],
        payload["effective_train_batch_size"],
        payload["optimizer"],
        _format_lr(payload["lr"]),
        payload["weight_decay"],
        payload["scheduler"],
        _format_scalar(payload["scheduler_factor"], precision=3),
        payload["scheduler_patience"],
        payload["warmup_epochs"],
        payload["amp_requested"],
        payload["amp_actually_used"],
        payload["compile_requested"],
        payload["compile_available"],
        payload["compile_actually_used"],
        payload["compile_mode"],
        payload["channels_last"],
        payload["eval_interval"],
        payload["early_stopping_patience"],
        payload["num_workers"],
        payload["pin_memory"],
        payload["persistent_workers"],
        payload["prefetch_factor"],
        payload["verify_images"],
        payload["ensemble_mode"],
        payload["branch_mode"],
        payload["local_branch_mode"],
        payload["metadata_enabled"],
        payload["metadata_mode"],
        payload["metadata_use_gender"],
        payload["metadata_use_chronological"],
        payload["metadata_inputs"],
        payload["heatmap_guidance"],
        payload["cbam_enabled"],
        payload["cbam_global"],
        payload["cbam_local"],
        payload["input_size"],
        payload["local_patch_size"],
        payload["global_crop_mode"],
        _format_scalar(payload["global_crop_margin_ratio"], precision=3),
        _format_scalar(payload["heatmap_sigma_ratio"], precision=4),
        _format_scalar(payload["heatmap_sigma_min"], precision=2),
        payload["target_type"],
        _format_scalar(payload["normalization_mean"], precision=8),
        _format_scalar(payload["normalization_std"], precision=8),
        payload["normalization_source"],
        payload["dataset_sizes"],
        extra=_phase_extra("SYSTEM"),
    )
    if payload.get("compile_reason"):
        logger.info(
            "Compile status | requested=%s | available=%s | used=%s | reason=%s | cudagraphs=%s",
            payload["compile_requested"],
            payload["compile_available"],
            payload["compile_actually_used"],
            payload["compile_reason"],
            payload.get("compile_cudagraphs"),
            extra=_phase_extra("SYSTEM"),
        )


def _load_checkpoint_state(checkpoint_path: str | Path | None) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    return torch.load(path, map_location="cpu")


def _fit_or_restore_normalizers(
    train_records: list[dict[str, Any]],
    target_mode: str,
    relative_direction: str,
    checkpoint_state: dict[str, Any] | None,
) -> tuple[ScalarNormalizer, ScalarNormalizer]:
    if checkpoint_state is not None and "normalizers" in checkpoint_state:
        state = checkpoint_state["normalizers"]
        return (
            ScalarNormalizer.from_state_dict(state.get("target")),
            ScalarNormalizer.from_state_dict(state.get("chronological")),
        )

    chronological_values = [record["chronological"] for record in train_records]
    chronological_normalizer = ScalarNormalizer.fit(chronological_values)

    target_values = []
    for record in train_records:
        if not record["has_boneage"]:
            continue
        target_value = record["boneage"]
        if target_mode == "relative":
            if relative_direction == "chronological_minus_boneage":
                target_value = record["chronological"] - target_value
            else:
                target_value = target_value - record["chronological"]
        target_values.append(target_value)
    target_normalizer = ScalarNormalizer.fit(target_values)
    return target_normalizer, chronological_normalizer


def _limit_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit and limit > 0:
        return records[:limit]
    return records


def _coerce_optional_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _empty_phase_stats(phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "total_time": None,
        "data_time": None,
        "transfer_time": None,
        "compute_time": None,
        "avg_batch_time": None,
        "min_batch_time": None,
        "max_batch_time": None,
        "samples_per_second": None,
        "max_allocated_mb": None,
        "max_reserved_mb": None,
    }


def _build_data_payload(
    config: dict[str, Any],
    run_dir: Path,
    checkpoint_state: dict[str, Any] | None = None,
    manual_split: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_images = bool(config["data"]["verify_images"])
    if manual_split is not None:
        split_name = manual_split["split"]
        sources, records, report = build_manual_split_records(
            split=split_name,
            image_dir=manual_split["image_dir"],
            csv_path=manual_split["csv_path"],
            roi_json_path=manual_split["roi_json_path"],
            verify_images=verify_images,
        )
        payload = {
            "dataset_root": str(Path(manual_split["image_dir"]).parent),
            "splits": {split_name: {"sources": sources.to_dict(), "records": records}},
            "reports": {split_name: report},
        }
    else:
        payload = build_dataset_index(config["data"]["dataset_root"], verify_images=verify_images)

    write_json(payload["reports"], run_dir / "dataset_report.json")
    return payload, payload["reports"]


def _build_datasets(
    payload: dict[str, Any],
    config: dict[str, Any],
    checkpoint_state: dict[str, Any] | None,
) -> tuple[dict[str, RHPEBoneAgeDataset], dict[str, ScalarNormalizer]]:
    split_records = {split: item["records"] for split, item in payload["splits"].items()}
    train_records = split_records.get("train", [])
    debug_cfg = config["debug"]
    for split_name, records in list(split_records.items()):
        if split_name == "train":
            split_records[split_name] = _limit_records(records, int(debug_cfg["limit_train_samples"]))
        elif split_name == "val":
            split_records[split_name] = _limit_records(records, int(debug_cfg["limit_val_samples"]))
        elif split_name == "test":
            split_records[split_name] = _limit_records(records, int(debug_cfg["limit_test_samples"]))
        else:
            split_records[split_name] = records
    train_records = split_records.get("train", [])

    target_normalizer, chronological_normalizer = _fit_or_restore_normalizers(
        train_records=train_records,
        target_mode=config["model"]["target_mode"],
        relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
        checkpoint_state=checkpoint_state,
    )

    dataset_stats = {
        "target": target_normalizer,
        "chronological": chronological_normalizer,
    }

    datasets = {}
    for split, records in split_records.items():
        if not records:
            continue
        datasets[split] = RHPEBoneAgeDataset(
            records=records,
            config=config,
            stats=DatasetStats(
                target_normalizer=target_normalizer,
                chronological_normalizer=chronological_normalizer,
            ),
            geometric_transform=build_geometric_transform(config, is_train=split == "train"),
            image_intensity_transform=build_image_intensity_transform(config, is_train=split == "train"),
        )
    return datasets, dataset_stats


def _build_dataloaders(datasets: dict[str, Any], config: dict[str, Any], device: torch.device) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    training_cfg = config["training"]
    if training_cfg.get("workers_override") is not None:
        workers = int(training_cfg["workers_override"])
        loader_kwargs = {
            "num_workers": workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": workers > 0,
        }
        if workers > 0:
            default_prefetch = 2
            loader_kwargs["prefetch_factor"] = _coerce_optional_int(
                training_cfg.get("prefetch_factor"),
                default=default_prefetch,
            )
    else:
        loader_kwargs = suggest_dataloader_kwargs(
            batch_size=int(training_cfg["batch_size"]),
            use_cuda=device.type == "cuda",
        )

    if training_cfg.get("pin_memory") is not None:
        loader_kwargs["pin_memory"] = bool(training_cfg["pin_memory"])
    if device.type != "cuda":
        loader_kwargs["pin_memory"] = False

    if loader_kwargs["num_workers"] > 0:
        if training_cfg.get("persistent_workers") is not None:
            loader_kwargs["persistent_workers"] = bool(training_cfg["persistent_workers"])
        if training_cfg.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(training_cfg["prefetch_factor"])
        else:
            loader_kwargs["prefetch_factor"] = int(loader_kwargs.get("prefetch_factor", 2))
    else:
        loader_kwargs.pop("prefetch_factor", None)
        loader_kwargs["persistent_workers"] = False

    dataloaders = {}
    if "train" in datasets:
        dataloaders["train"] = DataLoader(
            datasets["train"],
            batch_size=int(training_cfg["batch_size"]),
            shuffle=True,
            worker_init_fn=_worker_init_fn if loader_kwargs["num_workers"] > 0 else None,
            **loader_kwargs,
        )
    if "val" in datasets:
        dataloaders["val"] = DataLoader(
            datasets["val"],
            batch_size=int(training_cfg["val_batch_size"]),
            shuffle=False,
            worker_init_fn=_worker_init_fn if loader_kwargs["num_workers"] > 0 else None,
            **loader_kwargs,
        )
    if "test" in datasets:
        dataloaders["test"] = DataLoader(
            datasets["test"],
            batch_size=int(training_cfg["test_batch_size"]),
            shuffle=False,
            worker_init_fn=_worker_init_fn if loader_kwargs["num_workers"] > 0 else None,
            **loader_kwargs,
        )
    return dataloaders, loader_kwargs


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    training_cfg = config["training"]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    params = []
    if decay_params:
        params.append({"params": decay_params, "weight_decay": float(training_cfg["weight_decay"])})
    if no_decay_params:
        params.append({"params": no_decay_params, "weight_decay": 0.0})
    name = str(training_cfg["optimizer"]).lower()
    lr = float(training_cfg["lr"])
    momentum = float(training_cfg["momentum"])
    use_fused = bool(torch.cuda.is_available()) and all(
        param.device.type == "cuda"
        for group in params
        for param in group["params"]
    )

    def _try_build(optimizer_cls, **kwargs):
        if use_fused and name in {"adam", "adamw"}:
            try:
                return optimizer_cls(params, fused=True, **kwargs)
            except (TypeError, RuntimeError):
                pass
        return optimizer_cls(params, **kwargs)

    if name == "adam":
        return _try_build(torch.optim.Adam, lr=lr)
    if name == "adamw":
        return _try_build(torch.optim.AdamW, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=True,
        )
    raise ValueError(f"不支持的优化器: {training_cfg['optimizer']}")


def _build_scheduler(optimizer, config: dict[str, Any]):
    training_cfg = config["training"]
    name = str(training_cfg["scheduler"]).lower()
    min_lr = float(training_cfg["min_lr"])
    scheduler_factor = min(max(float(training_cfg.get("scheduler_factor", 0.5) or 0.5), 1e-4), 0.9999)
    scheduler_patience = _resolve_non_negative_int(training_cfg.get("scheduler_patience"), default=2)
    warmup_epochs, _ = _resolve_warmup_settings(config, int(training_cfg["epochs"]))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(training_cfg["epochs"]) - warmup_epochs),
            eta_min=min_lr,
        )
    if name == "none":
        return None
    raise ValueError(f"不支持的学习率调度器: {training_cfg['scheduler']}")


def _save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float | None,
    config: dict[str, Any],
    normalizers: dict[str, ScalarNormalizer],
) -> None:
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "normalizers": {key: normalizer.state_dict() for key, normalizer in normalizers.items()},
        "image_normalization": copy.deepcopy(((config.get("data") or {}).get("normalization") or {})),
    }
    torch.save(state, checkpoint_path)


def _move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=True)


def _restore_training_state(model, optimizer, scheduler, scaler, checkpoint_state: dict[str, Any]) -> int:
    unwrap_model(model).load_state_dict(checkpoint_state["model"])
    if optimizer is not None and checkpoint_state.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        _move_optimizer_state_to_device(optimizer, next(unwrap_model(model).parameters()).device)
    if scheduler is not None and checkpoint_state.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint_state["scheduler"])
    if scaler is not None and checkpoint_state.get("scaler") is not None:
        scaler.load_state_dict(checkpoint_state["scaler"])
    return int(checkpoint_state.get("epoch", 0)) + 1


def _log_reports(logger, reports: dict[str, Any]) -> None:
    for split, report in reports.items():
        issues = report["issues"]
        logger.info(
            "数据检查 | split=%s | matched=%d | missing_image=%d | missing_csv=%d | missing_roi=%d | duplicate_csv=%d | duplicate_image=%d | duplicate_roi=%d | unreadable=%d",
            split,
            report["matched_records"],
            len(issues["missing_images"]),
            len(issues["missing_csv_records"]),
            len(issues["missing_roi_json"]),
            len(issues["duplicate_csv_ids"]),
            len(issues["duplicate_image_ids"]),
            len(issues["duplicate_roi_ids"]),
            len(issues["unreadable_images"]),
        )


def _prepare_run_dir(config: dict[str, Any], purpose: str) -> Path:
    output_root = ensure_dir(config["experiment"]["output_root"])
    run_name = f"{config['experiment']['name']}_{purpose}_{timestamp()}"
    return ensure_dir(output_root / run_name)


def _resolve_config(
    config_path: str | Path | None,
    overrides: list[str] | None,
    checkpoint_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    checkpoint_state = _load_checkpoint_state(checkpoint_path)
    checkpoint_config = checkpoint_state.get("config") if checkpoint_state else None
    config = load_config(config_path, overrides=overrides, checkpoint_config=checkpoint_config)
    return config, checkpoint_state


def train_main(
    config_path: str | Path,
    overrides: list[str] | None = None,
    control: TrainingControl | None = None,
) -> dict[str, Any]:
    base_config, _ = _resolve_config(
        config_path=config_path,
        overrides=overrides,
        checkpoint_path=None,
    )
    resume_checkpoint = (base_config.get("training") or {}).get("resume_checkpoint")
    if resume_checkpoint:
        config, checkpoint_state = _resolve_config(
            config_path=config_path,
            overrides=overrides,
            checkpoint_path=resume_checkpoint,
        )
    else:
        config = base_config
        checkpoint_state = None
    run_dir = _prepare_run_dir(config, purpose="train")
    artifact_dirs = _prepare_artifact_dirs(run_dir)
    logger = setup_logger(run_dir)
    training_started_at, training_timer_source = _resolve_training_started_at(control)
    if control is not None:
        control.update_phase("system", "initializing")
        control.reset_stop_logged()
    requested_device, allow_cpu_fallback, deterministic = _resolve_runtime_settings(config)
    seed_everything(int(config["experiment"]["seed"]), deterministic=deterministic)

    try:
        logger.info(
            "训练任务开始 | run_dir=%s | timer_source=%s | elapsed=%s",
            run_dir,
            training_timer_source,
            _format_duration_clock(time.perf_counter() - training_started_at),
            extra=_phase_extra("SYSTEM"),
        )
        if resume_checkpoint:
            logger.info("训练模式 | mode=resume | checkpoint=%s", resume_checkpoint, extra=_phase_extra("SYSTEM"))
        else:
            logger.info("训练模式 | mode=fresh | checkpoint=n/a", extra=_phase_extra("SYSTEM"))
        _log_running_mode(logger, config)
        raise_if_stop_requested(control, logger, phase="system", scope="initializing", checkpoint="before_runtime_setup")

        device, runtime = detect_runtime(
            requested_device=requested_device,
            allow_cpu_fallback=allow_cpu_fallback,
            deterministic=deterministic,
        )
        write_json(runtime.to_dict(), run_dir / "runtime.json")
        _log_runtime_info(logger, runtime)
        if runtime.requested_device.startswith("cuda") and device.type != "cuda":
            logger.warning("请求设备 %s 不可用，训练已回退到 CPU。", runtime.requested_device)

        raise_if_stop_requested(control, logger, phase="system", scope="loading_data", checkpoint="before_dataset_build")
        payload, reports = _build_data_payload(config, run_dir, checkpoint_state=checkpoint_state)
        _log_reports(logger, reports)
        _resolve_image_normalization(config, payload, checkpoint_state, run_dir, logger)
        datasets, normalizers = _build_datasets(payload, config, checkpoint_state=checkpoint_state)
        dataset_summary = _build_dataset_summary(payload, datasets)
        write_json(dataset_summary, run_dir / "dataset_summary.json")
        _log_dataset_summary(logger, dataset_summary)
        config_summary = _build_config_summary(config, runtime, datasets)
        write_json(config_summary, run_dir / "config_summary.json")
        _log_config_summary(logger, config_summary)
        dataloaders, loader_kwargs = _build_dataloaders(datasets, config, device)
        write_json(loader_kwargs, run_dir / "dataloader.json")
        _log_dataloader_kwargs(logger, loader_kwargs)

        raise_if_stop_requested(control, logger, phase="system", scope="building_model", checkpoint="before_model_init")
        use_channels_last = _should_use_channels_last(config, device)
        model = build_model(config).to(device)
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        log_device_probe(model, device, logger)
        model, compile_info = maybe_compile_model(
            model,
            bool(config["training"]["compile"]),
            logger,
            mode=str(config["training"].get("compile_mode") or "default"),
        )
        criterion = build_loss(config["training"]["loss"], config["training"]["smooth_l1_beta"])
        optimizer = _build_optimizer(unwrap_model(model), config)
        scheduler = _build_scheduler(optimizer, config)
        scaler = None
        use_amp = device.type == "cuda" and bool(config["training"]["amp"])
        if use_amp:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        show_progress = bool(config["training"].get("progress_bar", True))
        log_interval = _resolve_log_interval(config)
        total_epochs = int(config["training"]["epochs"])
        base_lr = float(config["training"]["lr"])
        scheduler_name = str(config["training"].get("scheduler") or "none").lower()
        grad_accum_steps = _resolve_gradient_accumulation_steps(config)
        warmup_epochs, warmup_start_factor = _resolve_warmup_settings(config, total_epochs)
        eval_interval = _resolve_eval_interval(config, scheduler_name, logger)
        save_interval = _resolve_save_interval(config)
        early_stop_patience, early_stop_min_delta = _resolve_early_stopping(config)

        start_epoch = 1
        best_metric = None
        if checkpoint_state is not None:
            start_epoch = _restore_training_state(model, optimizer, scheduler, scaler, checkpoint_state)
            best_metric = checkpoint_state.get("best_metric")
            logger.info("已从 checkpoint 续训: %s | next_epoch=%d", resume_checkpoint, start_epoch)
        elif warmup_epochs > 0:
            _set_optimizer_lr(optimizer, _warmup_lr(base_lr, 1, warmup_epochs, warmup_start_factor))

        save_config(config, run_dir / "config.yaml")
        write_json(config, run_dir / "config.json")
        write_json({"config": config, "runtime": runtime.to_dict()}, run_dir / "run_config.json")
        effective_payload = _build_effective_params_payload(
            config=config,
            runtime=runtime,
            datasets=datasets,
            loader_kwargs=loader_kwargs,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
            compile_info=compile_info,
        )
        effective_payload["eval_interval"] = eval_interval
        effective_payload["save_interval"] = save_interval
        effective_payload["gradient_accumulation_steps"] = grad_accum_steps
        effective_payload["effective_train_batch_size"] = int(config["training"]["batch_size"]) * grad_accum_steps
        write_json(effective_payload, run_dir / "effective_params.json")
        _log_effective_params(logger, effective_payload)
        history_rows = []
        best_metric_name = _validate_best_metric(config["training"]["best_metric"])
        best_checkpoint_path = artifact_dirs["model_dir"] / "best_model.pt"
        last_checkpoint_path = artifact_dirs["model_dir"] / "last_checkpoint.pt"
        epochs_without_improvement = 0

        for epoch in range(start_epoch, total_epochs + 1):
            raise_if_stop_requested(control, logger, phase="system", scope=f"epoch-{epoch}", checkpoint="before_epoch")
            epoch_started = time.perf_counter()
            _log_epoch_header(logger, config, optimizer, device, use_amp, epoch, total_epochs, log_interval)
            lr_start = float(optimizer.param_groups[0]["lr"])

            train_metrics, _, train_stats = run_epoch(
                model=model,
                loader=dataloaders["train"],
                criterion=criterion,
                device=device,
                target_mode=config["model"]["target_mode"],
                target_normalizer=normalizers["target"],
                train=True,
                relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
                optimizer=optimizer,
                scaler=scaler,
                gradient_clip=float(config["training"]["gradient_clip"]),
                epoch=epoch,
                total_epochs=total_epochs,
                amp=use_amp,
                show_progress=show_progress,
                collect_predictions=False,
                logger=logger,
                log_interval=log_interval,
                control=control,
                grad_accum_steps=grad_accum_steps,
                channels_last=use_channels_last,
            )
            raise_if_stop_requested(control, logger, phase="train", scope=f"{epoch}/{total_epochs}", checkpoint="after_train_phase")
            run_validation = "val" in dataloaders and _should_run_validation(epoch, total_epochs, eval_interval)
            val_metrics = None
            val_predictions = pd.DataFrame()
            val_stats = _empty_phase_stats("eval")
            if run_validation:
                val_metrics, val_predictions, val_stats = run_epoch(
                    model=model,
                    loader=dataloaders["val"],
                    criterion=criterion,
                    device=device,
                    target_mode=config["model"]["target_mode"],
                    target_normalizer=normalizers["target"],
                    train=False,
                    relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
                    optimizer=None,
                    scaler=None,
                    gradient_clip=None,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    amp=use_amp,
                    show_progress=show_progress,
                    collect_predictions=True,
                    logger=logger,
                    log_interval=log_interval,
                    lr_override=lr_start,
                    control=control,
                    channels_last=use_channels_last,
                )
                raise_if_stop_requested(control, logger, phase="eval", scope=f"{epoch}/{total_epochs}", checkpoint="after_eval_phase")
            else:
                logger.info(
                    "Epoch %d/%d 跳过验证 | eval_interval=%d | 当前仅执行训练与 checkpoint 保存。",
                    epoch,
                    total_epochs,
                    eval_interval,
                    extra=_phase_extra("SYSTEM"),
                )

            previous_lr = float(optimizer.param_groups[0]["lr"])
            scheduler_label = scheduler_name
            if epoch < warmup_epochs:
                scheduler_label = "warmup"
                _set_optimizer_lr(
                    optimizer,
                    _warmup_lr(base_lr, epoch + 1, warmup_epochs, warmup_start_factor),
                )
            elif scheduler is not None:
                if scheduler_name == "plateau" and val_metrics is not None:
                    scheduler.step(val_metrics[best_metric_name] if val_metrics[best_metric_name] is not None else val_metrics["loss"])
                elif scheduler_name != "plateau":
                    scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])
            _log_learning_rate_update(logger, scheduler_label, epoch, total_epochs, previous_lr, current_lr)

            current_metric = val_metrics[best_metric_name] if val_metrics is not None else None
            best_updated = False
            if run_validation and _metric_improved(current_metric, best_metric, early_stop_min_delta):
                best_metric = current_metric
                best_updated = True
                epochs_without_improvement = 0
                _save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    config=config,
                    normalizers=normalizers,
                )
                val_predictions.to_csv(run_dir / "best_val_predictions.csv", index=False)
                logger.info(
                    "Best model updated | epoch=%d/%d | metric=%s | value=%s | path=%s",
                    epoch,
                    total_epochs,
                    best_metric_name,
                    _format_scalar(current_metric),
                    best_checkpoint_path,
                    extra=_phase_extra("SYSTEM"),
                )
            elif run_validation and current_metric is not None:
                epochs_without_improvement += 1

            if _should_save_checkpoint(epoch, total_epochs, save_interval):
                _save_checkpoint(
                    last_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    config=config,
                    normalizers=normalizers,
                )
                _log_checkpoint_saved(
                    logger,
                    label="last",
                    checkpoint_path=last_checkpoint_path,
                    epoch=epoch,
                    total_epochs=total_epochs,
                )

            epoch_total_time = time.perf_counter() - epoch_started
            total_elapsed_seconds = time.perf_counter() - training_started_at

            history_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"] if val_metrics is not None else None,
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"] if val_metrics is not None else None,
                "train_mad": train_metrics["mad"],
                "val_mad": val_metrics["mad"] if val_metrics is not None else None,
                "train_final_mae": train_metrics["final_mae"],
                "val_final_mae": val_metrics["final_mae"] if val_metrics is not None else None,
                "train_final_mad": train_metrics["final_mad"],
                "val_final_mad": val_metrics["final_mad"] if val_metrics is not None else None,
                "train_relative_mae": train_metrics["relative_mae"],
                "val_relative_mae": val_metrics["relative_mae"] if val_metrics is not None else None,
                "train_relative_mad": train_metrics["relative_mad"],
                "val_relative_mad": val_metrics["relative_mad"] if val_metrics is not None else None,
                "lr_start": lr_start,
                "lr_end": current_lr,
                "eval_ran": run_validation,
                "epoch_time_seconds": epoch_total_time,
                "total_elapsed_seconds": total_elapsed_seconds,
                "best_updated": best_updated,
            }
            history_rows.append(history_row)
            pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

            _log_epoch_timing(
                logger=logger,
                epoch=epoch,
                total_epochs=total_epochs,
                train_stats=train_stats,
                eval_stats=val_stats,
                epoch_total_time=epoch_total_time,
            )
            _log_epoch_metrics(
                logger=logger,
                epoch=epoch,
                total_epochs=total_epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr_start=lr_start,
                lr_end=current_lr,
                eval_ran=run_validation,
                epoch_time_seconds=epoch_total_time,
                total_elapsed_seconds=total_elapsed_seconds,
                best_updated=best_updated,
            )
            if run_validation and early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                _save_checkpoint(
                    last_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    config=config,
                    normalizers=normalizers,
                )
                _log_checkpoint_saved(
                    logger,
                    label="last(early-stop)",
                    checkpoint_path=last_checkpoint_path,
                    epoch=epoch,
                    total_epochs=total_epochs,
                )
                logger.info(
                    "Early stopping triggered | epoch=%d/%d | patience=%d | min_delta=%s | best_%s=%s",
                    epoch,
                    total_epochs,
                    early_stop_patience,
                    _format_scalar(early_stop_min_delta),
                    best_metric_name,
                    _format_scalar(best_metric),
                    extra=_phase_extra("SYSTEM"),
                )
                break

        history_df = pd.DataFrame(history_rows)

        if not best_checkpoint_path.exists():
            fallback_epoch = int(history_rows[-1]["epoch"]) if history_rows else 0
            logger.warning("训练过程中未产生 best checkpoint，已回退为最后一个 checkpoint。")
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=fallback_epoch,
                best_metric=best_metric,
                config=config,
                normalizers=normalizers,
            )
            _log_checkpoint_saved(
                logger,
                label="best(fallback)",
                checkpoint_path=best_checkpoint_path,
                epoch=fallback_epoch,
                total_epochs=total_epochs,
            )

        raise_if_stop_requested(control, logger, phase="system", scope="best-val", checkpoint="before_best_val")
        best_state = _load_checkpoint_state(best_checkpoint_path)
        unwrap_model(model).load_state_dict(best_state["model"])
        logger.info("开始加载最佳 checkpoint 并执行最终验证。", extra=_phase_extra("SYSTEM"))
        val_metrics, val_predictions, _ = run_epoch(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
            target_mode=config["model"]["target_mode"],
            target_normalizer=normalizers["target"],
            train=False,
            relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
            epoch=None,
            total_epochs=None,
            amp=use_amp,
            show_progress=show_progress,
            collect_predictions=True,
            logger=logger,
            log_interval=log_interval,
            lr_override=optimizer.param_groups[0]["lr"],
            progress_label="best-val",
            control=control,
            channels_last=use_channels_last,
        )
        val_predictions.to_csv(run_dir / "val_predictions.csv", index=False)
        write_json(val_metrics, run_dir / "val_metrics.json")

        output = {
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_checkpoint_path),
            "last_checkpoint": str(last_checkpoint_path),
            "val_metrics": val_metrics,
        }
        test_metrics = None
        test_predictions = None
        if "test" in dataloaders:
            raise_if_stop_requested(control, logger, phase="system", scope="test", checkpoint="before_test_eval")
            logger.info("开始执行 test 集评估。", extra=_phase_extra("SYSTEM"))
            test_metrics, test_predictions, _ = run_epoch(
                model=model,
                loader=dataloaders["test"],
                criterion=criterion,
                device=device,
                target_mode=config["model"]["target_mode"],
                target_normalizer=normalizers["target"],
                train=False,
                relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
                epoch=None,
                total_epochs=None,
                amp=use_amp,
                show_progress=show_progress,
                collect_predictions=True,
                logger=logger,
                log_interval=log_interval,
                lr_override=optimizer.param_groups[0]["lr"],
                progress_label="test",
                control=control,
                channels_last=use_channels_last,
            )
            test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)
            write_json(test_metrics, run_dir / "test_metrics.json")
            output["test_metrics"] = test_metrics

        try:
            raise_if_stop_requested(control, logger, phase="system", scope="report", checkpoint="before_report_generation")
            logger.info("开始生成训练报告文件。", extra=_phase_extra("SYSTEM"))
            report_summary = generate_training_report(
                output_dir=run_dir,
                history_df=history_df,
                val_predictions=val_predictions,
                test_predictions=test_predictions,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                config=config,
                runtime=runtime.to_dict(),
                best_metric_name=best_metric_name,
                best_checkpoint_path=best_checkpoint_path,
                last_checkpoint_path=last_checkpoint_path,
            )
            output["best_summary"] = report_summary
            logger.info("训练报告生成完成。", extra=_phase_extra("SYSTEM"))
        except Exception as exc:
            logger.exception("论文结果文件生成失败。")
            raise RuntimeError(f"训练已完成，但论文结果文件生成失败: {exc}") from exc
        total_elapsed_seconds = time.perf_counter() - training_started_at
        logger.info(
            "训练任务完成 | total_elapsed=%s | run_dir=%s | best_checkpoint=%s | last_checkpoint=%s",
            _format_duration_clock(total_elapsed_seconds),
            run_dir,
            best_checkpoint_path,
            last_checkpoint_path,
            extra=_phase_extra("SYSTEM"),
        )
        output["total_elapsed_seconds"] = total_elapsed_seconds
        output["total_elapsed_text"] = _format_duration_clock(total_elapsed_seconds)
        return output
    except TrainingCancelledError as exc:
        logger.warning(
            "训练已按请求停止 | phase=%s | scope=%s | checkpoint=%s | elapsed=%s",
            exc.phase,
            exc.scope or "n/a",
            exc.checkpoint or "n/a",
            _format_duration_clock(time.perf_counter() - training_started_at),
            extra=_phase_extra(exc.phase or "SYSTEM"),
        )
        if "device" in locals() and isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.empty_cache()
        raise
    except Exception:
        logger.exception(
            "训练异常退出 | elapsed=%s",
            _format_duration_clock(time.perf_counter() - training_started_at),
            extra=_phase_extra("SYSTEM"),
        )
        if "device" in locals() and isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.empty_cache()
        raise
    finally:
        if control is not None:
            control.update_phase("system", "idle")


def evaluate_main(
    checkpoint_path: str | Path,
    split: str,
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
    manual_split: dict[str, str] | None = None,
) -> dict[str, Any]:
    config, checkpoint_state = _resolve_config(
        config_path=config_path,
        overrides=overrides,
        checkpoint_path=checkpoint_path,
    )
    run_dir = _prepare_run_dir(config, purpose=split)
    logger = setup_logger(run_dir)
    requested_device, allow_cpu_fallback, deterministic = _resolve_runtime_settings(config)
    seed_everything(int(config["experiment"]["seed"]), deterministic=deterministic)
    _log_running_mode(logger, config)

    device, runtime = detect_runtime(
        requested_device=requested_device,
        allow_cpu_fallback=allow_cpu_fallback,
        deterministic=deterministic,
    )
    write_json(runtime.to_dict(), run_dir / "runtime.json")
    _log_runtime_info(logger, runtime)
    payload, reports = _build_data_payload(config, run_dir, checkpoint_state=checkpoint_state, manual_split=manual_split)
    _log_reports(logger, reports)
    _resolve_image_normalization(config, payload, checkpoint_state, run_dir, logger)
    datasets, normalizers = _build_datasets(payload, config, checkpoint_state=checkpoint_state)
    dataset_summary = _build_dataset_summary(payload, datasets)
    write_json(dataset_summary, run_dir / "dataset_summary.json")
    _log_dataset_summary(logger, dataset_summary)
    config_summary = _build_config_summary(config, runtime, datasets)
    write_json(config_summary, run_dir / "config_summary.json")
    _log_config_summary(logger, config_summary)
    dataloaders, loader_kwargs = _build_dataloaders(datasets, config, device)
    write_json(loader_kwargs, run_dir / "dataloader.json")
    _log_dataloader_kwargs(logger, loader_kwargs)

    if split not in dataloaders:
        raise ValueError(f"请求评估的 split 不存在: {split}")

    model = build_model(config).to(device)
    use_channels_last = _should_use_channels_last(config, device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    log_device_probe(model, device, logger)
    unwrap_model(model).load_state_dict(checkpoint_state["model"])
    criterion = build_loss(config["training"]["loss"], config["training"]["smooth_l1_beta"])
    use_amp = device.type == "cuda" and bool(config["training"]["amp"])
    show_progress = bool(config["training"].get("progress_bar", True))
    log_interval = _resolve_log_interval(config)

    metrics, predictions, _ = run_epoch(
        model=model,
        loader=dataloaders[split],
        criterion=criterion,
        device=device,
        target_mode=config["model"]["target_mode"],
        target_normalizer=normalizers["target"],
        train=False,
        relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
        epoch=None,
        total_epochs=None,
        amp=use_amp,
        show_progress=show_progress,
        collect_predictions=True,
        logger=logger,
        log_interval=log_interval,
        progress_label=split,
        channels_last=use_channels_last,
    )
    predictions.to_csv(run_dir / f"{split}_predictions.csv", index=False)
    write_json(metrics, run_dir / f"{split}_metrics.json")
    write_json(metrics, run_dir / "metrics.json")
    save_config(config, run_dir / "config.yaml")
    write_json(config, run_dir / "config.json")
    effective_payload = _build_effective_params_payload(
        config=config,
        runtime=runtime,
        datasets=datasets,
        loader_kwargs=loader_kwargs,
        use_amp=use_amp,
        use_channels_last=use_channels_last,
    )
    write_json(effective_payload, run_dir / "effective_params.json")
    write_json({"config": config, "runtime": runtime.to_dict()}, run_dir / "run_config.json")
    logger.info("%s 评估完成 | metrics=%s", split, metrics)
    return {"run_dir": str(run_dir), "metrics": metrics}


def tune_main(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    import optuna

    base_config, _ = _resolve_config(config_path=config_path, overrides=overrides, checkpoint_path=None)
    tune_dir = _prepare_run_dir(base_config, purpose="optuna")
    logger = setup_logger(tune_dir)
    _log_running_mode(logger, base_config)
    save_config(base_config, tune_dir / "base_config.yaml")
    write_json(base_config, tune_dir / "base_config.json")

    optuna_cfg = base_config["optuna"]

    def objective(trial: optuna.Trial) -> float:
        trial_config = copy.deepcopy(base_config)
        trial_config["experiment"]["name"] = f"{base_config['experiment']['name']}_trial_{trial.number:03d}"
        trial_config["training"]["epochs"] = int(optuna_cfg["epochs_per_trial"])
        trial_config["training"]["lr"] = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        trial_config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16])
        trial_config["training"]["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        trial_config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        trial_config["training"]["scheduler"] = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])
        trial_config["model"]["head"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        trial_config["model"]["cbam"]["enabled"] = trial.suggest_categorical("cbam_enabled", [True, False])
        trial_config["model"]["target_mode"] = trial.suggest_categorical("target_mode", ["relative", "direct"])
        trial_config["model"]["metadata"]["hidden_dim"] = trial.suggest_categorical("metadata_hidden_dim", [32, 64, 128])
        trial_config["model"]["local_branch"]["feature_dim"] = trial.suggest_categorical("local_feature_dim", [64, 96, 128])
        trial_config["augmentation"]["use_blur"] = trial.suggest_categorical("use_blur", [True, False])
        trial_config["augmentation"]["use_noise"] = trial.suggest_categorical("use_noise", [True, False])
        trial_config["augmentation"]["horizontal_flip"] = trial.suggest_categorical("horizontal_flip", [False, True])
        # 这里直接调用内部训练逻辑，避免重复解析命令行。
        run_dir = _prepare_run_dir(trial_config, purpose="trial")
        logger_trial = setup_logger(run_dir, name=f"trial_{trial.number}")
        requested_device, allow_cpu_fallback, deterministic = _resolve_runtime_settings(trial_config)
        seed_everything(int(trial_config["experiment"]["seed"]), deterministic=deterministic)
        device, runtime = detect_runtime(
            requested_device=requested_device,
            allow_cpu_fallback=allow_cpu_fallback,
            deterministic=deterministic,
        )
        write_json(runtime.to_dict(), run_dir / "runtime.json")
        _log_runtime_info(logger_trial, runtime)
        payload, reports = _build_data_payload(trial_config, run_dir)
        _log_reports(logger_trial, reports)
        _resolve_image_normalization(trial_config, payload, None, run_dir, logger_trial)
        datasets, normalizers = _build_datasets(payload, trial_config, checkpoint_state=None)
        dataset_summary = _build_dataset_summary(payload, datasets)
        write_json(dataset_summary, run_dir / "dataset_summary.json")
        _log_dataset_summary(logger_trial, dataset_summary)
        config_summary = _build_config_summary(trial_config, runtime, datasets)
        write_json(config_summary, run_dir / "config_summary.json")
        _log_config_summary(logger_trial, config_summary)
        dataloaders, loader_kwargs = _build_dataloaders(datasets, trial_config, device)
        _log_dataloader_kwargs(logger_trial, loader_kwargs)
        model = build_model(trial_config).to(device)
        use_channels_last = _should_use_channels_last(trial_config, device)
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        log_device_probe(model, device, logger_trial)
        model, compile_info = maybe_compile_model(
            model,
            bool(trial_config["training"]["compile"]),
            logger_trial,
            mode=str(trial_config["training"].get("compile_mode") or "default"),
        )
        criterion = build_loss(trial_config["training"]["loss"], trial_config["training"]["smooth_l1_beta"])
        optimizer = _build_optimizer(unwrap_model(model), trial_config)
        scheduler = _build_scheduler(optimizer, trial_config)
        scaler = None
        use_amp = device.type == "cuda" and bool(trial_config["training"]["amp"])
        if use_amp:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        show_progress = bool(trial_config["training"].get("progress_bar", True))
        log_interval = _resolve_log_interval(trial_config)
        total_epochs = int(trial_config["training"]["epochs"])
        base_lr = float(trial_config["training"]["lr"])
        scheduler_name = str(trial_config["training"].get("scheduler") or "none").lower()
        grad_accum_steps = _resolve_gradient_accumulation_steps(trial_config)
        warmup_epochs, warmup_start_factor = _resolve_warmup_settings(trial_config, total_epochs)
        if warmup_epochs > 0:
            _set_optimizer_lr(optimizer, _warmup_lr(base_lr, 1, warmup_epochs, warmup_start_factor))

        best_metric = None
        for epoch in range(1, total_epochs + 1):
            epoch_started = time.perf_counter()
            _log_epoch_header(logger_trial, trial_config, optimizer, device, use_amp, epoch, total_epochs, log_interval)
            lr_start = float(optimizer.param_groups[0]["lr"])
            train_metrics, _, train_stats = run_epoch(
                model=model,
                loader=dataloaders["train"],
                criterion=criterion,
                device=device,
                target_mode=trial_config["model"]["target_mode"],
                target_normalizer=normalizers["target"],
                train=True,
                relative_direction=trial_config["model"].get("relative_target_direction", "boneage_minus_chronological"),
                optimizer=optimizer,
                scaler=scaler,
                gradient_clip=float(trial_config["training"]["gradient_clip"]),
                epoch=epoch,
                total_epochs=total_epochs,
                amp=use_amp,
                show_progress=show_progress,
                collect_predictions=False,
                logger=logger_trial,
                log_interval=log_interval,
                grad_accum_steps=grad_accum_steps,
                channels_last=use_channels_last,
            )
            val_metrics, _, val_stats = run_epoch(
                model=model,
                loader=dataloaders["val"],
                criterion=criterion,
                device=device,
                target_mode=trial_config["model"]["target_mode"],
                target_normalizer=normalizers["target"],
                train=False,
                relative_direction=trial_config["model"].get("relative_target_direction", "boneage_minus_chronological"),
                epoch=epoch,
                total_epochs=total_epochs,
                amp=use_amp,
                show_progress=show_progress,
                collect_predictions=False,
                logger=logger_trial,
                log_interval=log_interval,
                lr_override=lr_start,
                channels_last=use_channels_last,
            )
            value = val_metrics["mae"] if val_metrics["mae"] is not None else 1e9
            trial.report(value, step=epoch)
            previous_lr = float(optimizer.param_groups[0]["lr"])
            scheduler_label = scheduler_name
            if epoch < warmup_epochs:
                scheduler_label = "warmup"
                _set_optimizer_lr(
                    optimizer,
                    _warmup_lr(base_lr, epoch + 1, warmup_epochs, warmup_start_factor),
                )
            elif scheduler is not None:
                if trial_config["training"]["scheduler"].lower() == "plateau":
                    scheduler.step(value)
                else:
                    scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])
            _log_learning_rate_update(logger_trial, scheduler_label, epoch, total_epochs, previous_lr, current_lr)
            _log_epoch_timing(
                logger=logger_trial,
                epoch=epoch,
                total_epochs=total_epochs,
                train_stats=train_stats,
                eval_stats=val_stats,
                epoch_total_time=time.perf_counter() - epoch_started,
            )
            _log_epoch_metrics(
                logger=logger_trial,
                epoch=epoch,
                total_epochs=total_epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr_start=lr_start,
                lr_end=current_lr,
                eval_ran=True,
            )
            best_metric = value if best_metric is None else min(best_metric, value)
            if trial.should_prune():
                raise optuna.TrialPruned()

        save_config(trial_config, run_dir / "config.yaml")
        return float(best_metric if best_metric is not None else 1e9)

    study = optuna.create_study(direction=optuna_cfg["direction"], study_name=base_config["experiment"]["name"])
    study.optimize(
        objective,
        n_trials=int(optuna_cfg["n_trials"]),
        timeout=optuna_cfg["timeout"],
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(tune_dir / "optuna_trials.csv", index=False)
    best_payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    write_json(best_payload, tune_dir / "optuna_best.json")
    logger.info("Optuna 搜索完成 | best=%s", best_payload)
    return {"run_dir": str(tune_dir), **best_payload}
