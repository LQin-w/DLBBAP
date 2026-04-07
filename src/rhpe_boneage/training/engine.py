from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .control import TrainingControl, raise_if_stop_requested
from .metrics import compute_regression_metrics


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    channels_last: bool = True,
) -> dict[str, Any]:
    def _move(value: Any):
        if torch.is_tensor(value):
            if value.ndim == 4 and device.type == "cuda" and channels_last:
                return value.to(device=device, non_blocking=True, memory_format=torch.channels_last)
            return value.to(device=device, non_blocking=True)
        if isinstance(value, dict):
            return {key: _move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_move(item) for item in value)
        return value

    return {key: _move(value) for key, value in batch.items()}


def _log_first_batch_device(
    batch: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    logger,
    phase: str,
    epoch: int | None,
) -> None:
    if logger is None:
        return
    logged_phases = getattr(logger, "_logged_first_batch_phases", set())
    if phase in logged_phases:
        return
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device
    logger.info(
        "首个 batch | phase=%s | epoch=%s | model_device=%s | global_image=%s | local_images=%s | global_heatmap=%s | roi_vector=%s",
        phase,
        epoch,
        model_device,
        batch["global_image"].device,
        batch["local_images"].device,
        batch["global_heatmap"].device,
        batch["roi_vector"].device,
        extra={"phase": phase.upper()},
    )
    logged_phases.add(phase)
    setattr(logger, "_logged_first_batch_phases", logged_phases)


def build_relative_age(
    boneage: torch.Tensor,
    chronological: torch.Tensor,
    relative_direction: str = "boneage_minus_chronological",
) -> torch.Tensor:
    if relative_direction == "chronological_minus_boneage":
        return chronological - boneage
    return boneage - chronological


def build_training_target(
    batch: dict[str, torch.Tensor],
    target_mode: str,
    target_normalizer,
    relative_direction: str = "boneage_minus_chronological",
) -> torch.Tensor:
    if target_mode == "relative":
        raw_target = build_relative_age(
            boneage=batch["boneage"],
            chronological=batch["chronological"],
            relative_direction=relative_direction,
        )
    else:
        raw_target = batch["boneage"]
    return target_normalizer.transform_tensor(raw_target)


def decode_relative_age_prediction(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target_mode: str,
    target_normalizer,
    relative_direction: str = "boneage_minus_chronological",
) -> torch.Tensor:
    if target_mode == "relative":
        return target_normalizer.inverse_transform_tensor(prediction)
    final_boneage = target_normalizer.inverse_transform_tensor(prediction)
    return build_relative_age(
        boneage=final_boneage,
        chronological=batch["chronological"],
        relative_direction=relative_direction,
    )


def decode_boneage_prediction(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target_mode: str,
    target_normalizer,
    relative_direction: str = "boneage_minus_chronological",
) -> torch.Tensor:
    target_pred = target_normalizer.inverse_transform_tensor(prediction)
    if target_mode == "relative":
        if relative_direction == "chronological_minus_boneage":
            return batch["chronological"] - target_pred
        return target_pred + batch["chronological"]
    return target_pred


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def _inference_context(train: bool):
    if train:
        return nullcontext()
    return torch.inference_mode()


def _mark_cudagraph_step_begin(model: torch.nn.Module, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not hasattr(model, "_orig_mod"):
        return
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "cudagraph_mark_step_begin"):
        return
    compiler.cudagraph_mark_step_begin()


def _set_progress_postfix(progress, total_loss: float, total_count: int) -> None:
    if progress is None:
        return
    running_loss = (total_loss / total_count) if total_count > 0 else None
    if running_loss is not None:
        progress.set_postfix(loss=f"{running_loss:.4f}")
    else:
        progress.set_postfix(loss="n/a")


def _log_first_batch_wait(logger, phase: str, epoch: int | None, seconds: float) -> None:
    if logger is None:
        return
    logger.info(
        "首个 batch 已就绪 | phase=%s | epoch=%s | wait_seconds=%.2f",
        phase,
        epoch,
        seconds,
        extra={"phase": phase.upper()},
    )


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.2f}s"


def _format_loss_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _format_lr_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _format_memory_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}MB"


def _resolve_scope_label(epoch: int | None, total_epochs: int | None, progress_label: str | None) -> str:
    if progress_label:
        return progress_label
    if total_epochs is not None and epoch is not None:
        return f"{epoch}/{total_epochs}"
    if epoch is not None:
        return str(epoch)
    return "n/a"


def _cuda_memory_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "max_allocated_mb": None,
            "max_reserved_mb": None,
        }
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_mb": torch.cuda.memory_allocated(device_index) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device_index) / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device_index) / (1024**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device_index) / (1024**2),
    }


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    target_mode: str,
    target_normalizer,
    train: bool,
    relative_direction: str = "boneage_minus_chronological",
    optimizer=None,
    scaler=None,
    gradient_clip: float | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    amp: bool = False,
    show_progress: bool = True,
    collect_predictions: bool = True,
    logger=None,
    log_interval: int = 20,
    lr_override: float | None = None,
    progress_label: str | None = None,
    control: TrainingControl | None = None,
    grad_accum_steps: int = 1,
    channels_last: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    model.train(train)
    phase = "train" if train else "eval"
    amp_enabled = bool(amp) and device.type == "cuda"
    grad_accum_steps = max(1, int(grad_accum_steps))
    total_loss = 0.0
    total_count = 0
    final_boneage_true: list[float] = []
    final_boneage_pred: list[float] = []
    relative_age_true: list[float] = []
    relative_age_pred: list[float] = []
    rows: list[dict[str, Any]] | None = [] if collect_predictions else None

    scope_label = _resolve_scope_label(epoch, total_epochs, progress_label)
    if progress_label:
        progress_desc = f"{phase} {progress_label}"
    elif total_epochs is not None and epoch is not None:
        progress_desc = f"{phase} {epoch}/{total_epochs}"
    elif epoch is not None:
        progress_desc = f"{phase} epoch={epoch}"
    else:
        progress_desc = phase
    total_batches = len(loader)
    progress = tqdm(
        loader,
        desc=progress_desc,
        total=total_batches,
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    phase_started = time.perf_counter()
    next_batch_wait_started = phase_started
    total_data_wait = 0.0
    total_batch_time = 0.0
    total_transfer_time = 0.0
    total_compute_time = 0.0
    min_batch_time = None
    max_batch_time = None
    optimizer_steps = 0
    sample_count = 0

    if logger is not None:
        logger.info(
            "Phase started | scope=%s | batches=%d | batch_size=%s | lr=%s | device=%s | amp=%s | grad_accum=%d | channels_last=%s",
            scope_label,
            total_batches,
            getattr(loader, "batch_size", "n/a"),
            _format_lr_value(lr_override if lr_override is not None else (optimizer.param_groups[0]["lr"] if optimizer is not None else None)),
            device,
            amp_enabled,
            grad_accum_steps,
            channels_last and device.type == "cuda",
            extra={"phase": phase.upper()},
        )
        logger.info(
            "正在等待首个 batch | scope=%s | 首次可能因数据预处理、DataLoader worker 启动和 torch.compile 预热而偏慢。",
            scope_label,
            extra={"phase": phase.upper()},
        )

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        with _inference_context(train):
            iterator = iter(progress)
            batch_index = 0
            while True:
                raise_if_stop_requested(
                    control,
                    logger,
                    phase=phase,
                    scope=scope_label,
                    checkpoint="before_batch_fetch",
                )
                try:
                    batch = next(iterator)
                except StopIteration:
                    break

                data_wait_seconds = time.perf_counter() - next_batch_wait_started
                total_data_wait += data_wait_seconds
                if batch_index == 0:
                    _log_first_batch_wait(
                        logger,
                        phase=phase,
                        epoch=epoch,
                        seconds=data_wait_seconds,
                    )
                batch_started = time.perf_counter()
                transfer_started = time.perf_counter()
                batch = move_batch_to_device(batch, device, channels_last=channels_last)
                transfer_time_seconds = time.perf_counter() - transfer_started
                total_transfer_time += transfer_time_seconds
                if batch_index == 0:
                    _log_first_batch_device(batch, model, device, logger, phase=phase, epoch=epoch)
                has_target_mask = batch["has_target"].view(-1).bool()
                compute_started = time.perf_counter()

                _mark_cudagraph_step_begin(model, device)
                with _autocast_context(device, amp_enabled):
                    outputs = model(batch)
                    prediction = outputs["prediction"]
                    loss = None
                    if has_target_mask.any():
                        normalized_target = build_training_target(
                            batch,
                            target_mode,
                            target_normalizer,
                            relative_direction=relative_direction,
                        )
                        loss = criterion(prediction[has_target_mask], normalized_target[has_target_mask])

                batch_number = batch_index + 1
                should_step = train and (
                    batch_number % grad_accum_steps == 0 or batch_number == total_batches
                )

                if train and loss is not None:
                    scaled_loss = loss / grad_accum_steps
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                        if should_step:
                            if gradient_clip and gradient_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_steps += 1
                    else:
                        scaled_loss.backward()
                        if should_step:
                            if gradient_clip and gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_steps += 1

                predicted_relative_age = decode_relative_age_prediction(
                    prediction,
                    batch,
                    target_mode,
                    target_normalizer,
                    relative_direction=relative_direction,
                )
                final_boneage = decode_boneage_prediction(
                    prediction,
                    batch,
                    target_mode,
                    target_normalizer,
                    relative_direction=relative_direction,
                )
                relative_age = build_relative_age(
                    boneage=batch["boneage"],
                    chronological=batch["chronological"],
                    relative_direction=relative_direction,
                )

                batch_size = prediction.shape[0]
                sample_count += batch_size
                if loss is not None:
                    valid_count = int(has_target_mask.sum().item())
                    total_loss += float(loss.detach().item()) * valid_count
                    total_count += valid_count

                if has_target_mask.any():
                    final_boneage_true.extend(batch["boneage"][has_target_mask].detach().view(-1).cpu().tolist())
                    final_boneage_pred.extend(final_boneage[has_target_mask].detach().view(-1).cpu().tolist())
                    relative_age_true.extend(relative_age[has_target_mask].detach().view(-1).cpu().tolist())
                    relative_age_pred.extend(predicted_relative_age[has_target_mask].detach().view(-1).cpu().tolist())

                if collect_predictions:
                    has_target_cpu = batch["has_target"].detach().view(-1).cpu().numpy().astype(bool)
                    boneage_cpu = batch["boneage"].detach().view(-1).cpu().numpy()
                    final_boneage_cpu = final_boneage.detach().view(-1).cpu().numpy()
                    chronological_cpu = batch["chronological"].detach().view(-1).cpu().numpy()
                    male_index_cpu = batch["male_index"].detach().view(-1).cpu().numpy()
                    relative_age_cpu = relative_age.detach().view(-1).cpu().numpy()
                    predicted_relative_age_cpu = predicted_relative_age.detach().view(-1).cpu().numpy()

                    for index in range(batch_size):
                        gt_value = float(boneage_cpu[index]) if has_target_cpu[index] else np.nan
                        pred_value = float(final_boneage_cpu[index])
                        abs_error = abs(pred_value - gt_value) if not np.isnan(gt_value) else np.nan
                        chronological_value = float(chronological_cpu[index])
                        relative_gt = float(relative_age_cpu[index]) if has_target_cpu[index] else np.nan
                        relative_pred = float(predicted_relative_age_cpu[index])
                        rows.append(
                            {
                                "ID": batch["id"][index],
                                "gt_final_boneage": gt_value,
                                "pred_final_boneage": pred_value,
                                "final_abs_error": abs_error,
                                "gt_relative_age": relative_gt,
                                "pred_relative_age": relative_pred,
                                "gt_boneage": gt_value,
                                "pred_boneage": pred_value,
                                "abs_error": abs_error,
                                "sex": int(male_index_cpu[index]),
                                "chronological": chronological_value,
                                "final_boneage": pred_value,
                                "predicted_relative_age": relative_pred,
                                "gt_relative_boneage": relative_gt,
                                "pred_relative_boneage": relative_pred,
                            }
                        )

                if show_progress and ((batch_index + 1) % 10 == 0 or (batch_index + 1) == len(loader)):
                    _set_progress_postfix(progress, total_loss=total_loss, total_count=total_count)

                compute_time = time.perf_counter() - compute_started
                total_compute_time += compute_time
                batch_time = time.perf_counter() - batch_started
                total_batch_time += batch_time
                min_batch_time = batch_time if min_batch_time is None else min(min_batch_time, batch_time)
                max_batch_time = batch_time if max_batch_time is None else max(max_batch_time, batch_time)

                next_batch_wait_started = time.perf_counter()
                batch_index += 1
                raise_if_stop_requested(
                    control,
                    logger,
                    phase=phase,
                    scope=scope_label,
                    checkpoint="after_batch",
                )
    finally:
        if progress is not None:
            progress.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    phase_total_time = time.perf_counter() - phase_started
    final_metrics = compute_regression_metrics(final_boneage_true, final_boneage_pred)
    relative_metrics = compute_regression_metrics(relative_age_true, relative_age_pred)
    metrics = {
        "loss": total_loss / max(total_count, 1) if total_count > 0 else None,
        "final_mae": final_metrics.get("mae"),
        "final_mad": final_metrics.get("mad"),
        "relative_mae": relative_metrics.get("mae"),
        "relative_mad": relative_metrics.get("mad"),
        "mae": final_metrics.get("mae"),
        "mad": final_metrics.get("mad"),
    }
    batch_count = max(total_batches, 0)
    memory_stats = _cuda_memory_stats(device)
    phase_stats = {
        "phase": phase,
        "scope_label": scope_label,
        "total_time": phase_total_time,
        "data_time": total_data_wait,
        "transfer_time": total_transfer_time,
        "compute_time": total_compute_time,
        "avg_batch_time": (total_batch_time / batch_count) if batch_count > 0 else None,
        "min_batch_time": min_batch_time,
        "max_batch_time": max_batch_time,
        "batch_count": batch_count,
        "sample_count": sample_count,
        "samples_per_second": (sample_count / phase_total_time) if phase_total_time > 0 else None,
        "optimizer_steps": optimizer_steps,
        **memory_stats,
    }

    if not collect_predictions:
        metrics["relative_age_error_corr"] = None
        metrics["relative_age_error_slope"] = None
        if logger is not None:
            logger.info(
                "Phase finished | scope=%s | loss=%s | final_mae=%s | final_mad=%s | relative_mae=%s | relative_mad=%s | phase_time=%s | data_time=%s | transfer_time=%s | compute_time=%s | samples_per_sec=%.2f | avg_batch_time=%s | min_batch_time=%s | max_batch_time=%s | gpu_peak_alloc=%s | gpu_peak_reserved=%s | optimizer_steps=%d",
                scope_label,
                _format_loss_value(metrics.get("loss")),
                _format_loss_value(metrics.get("final_mae")),
                _format_loss_value(metrics.get("final_mad")),
                _format_loss_value(metrics.get("relative_mae")),
                _format_loss_value(metrics.get("relative_mad")),
                _format_seconds(phase_total_time),
                _format_seconds(total_data_wait),
                _format_seconds(total_transfer_time),
                _format_seconds(total_compute_time),
                phase_stats["samples_per_second"] or 0.0,
                _format_seconds(phase_stats["avg_batch_time"]),
                _format_seconds(min_batch_time),
                _format_seconds(max_batch_time),
                _format_memory_value(memory_stats["max_allocated_mb"]),
                _format_memory_value(memory_stats["max_reserved_mb"]),
                optimizer_steps,
                extra={"phase": phase.upper()},
            )
        return metrics, pd.DataFrame(), phase_stats

    prediction_df = pd.DataFrame(rows)
    valid_df = prediction_df[prediction_df["gt_boneage"].notna()].copy()
    if len(valid_df) >= 2:
        relative_values = valid_df["gt_relative_age"].to_numpy(dtype=np.float32)
        abs_errors = valid_df["abs_error"].to_numpy(dtype=np.float32)
        if (
            np.std(relative_values) < 1e-8
            or np.std(abs_errors) < 1e-8
            or len(np.unique(relative_values)) < 2
        ):
            corr = None
            slope = None
        else:
            corr = float(np.corrcoef(relative_values, abs_errors)[0, 1])
            slope = float(np.polyfit(relative_values, abs_errors, deg=1)[0])
        metrics["relative_age_error_corr"] = corr
        metrics["relative_age_error_slope"] = slope
    else:
        metrics["relative_age_error_corr"] = None
        metrics["relative_age_error_slope"] = None
    if logger is not None:
        logger.info(
            "Phase finished | scope=%s | loss=%s | final_mae=%s | final_mad=%s | relative_mae=%s | relative_mad=%s | phase_time=%s | data_time=%s | transfer_time=%s | compute_time=%s | samples_per_sec=%.2f | avg_batch_time=%s | min_batch_time=%s | max_batch_time=%s | gpu_peak_alloc=%s | gpu_peak_reserved=%s | optimizer_steps=%d",
            scope_label,
            _format_loss_value(metrics.get("loss")),
            _format_loss_value(metrics.get("final_mae")),
            _format_loss_value(metrics.get("final_mad")),
            _format_loss_value(metrics.get("relative_mae")),
            _format_loss_value(metrics.get("relative_mad")),
            _format_seconds(phase_total_time),
            _format_seconds(total_data_wait),
            _format_seconds(total_transfer_time),
            _format_seconds(total_compute_time),
            phase_stats["samples_per_second"] or 0.0,
            _format_seconds(phase_stats["avg_batch_time"]),
            _format_seconds(min_batch_time),
            _format_seconds(max_batch_time),
            _format_memory_value(memory_stats["max_allocated_mb"]),
            _format_memory_value(memory_stats["max_reserved_mb"]),
            optimizer_steps,
            extra={"phase": phase.upper()},
        )
    return metrics, prediction_df, phase_stats
