from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .metrics import compute_regression_metrics


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def _move(value: Any):
        if torch.is_tensor(value):
            if value.ndim == 4 and device.type == "cuda":
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
    )
    logged_phases.add(phase)
    setattr(logger, "_logged_first_batch_phases", logged_phases)


def build_training_target(
    batch: dict[str, torch.Tensor],
    target_mode: str,
    target_normalizer,
    relative_direction: str = "boneage_minus_chronological",
) -> torch.Tensor:
    if target_mode == "relative":
        if relative_direction == "chronological_minus_boneage":
            raw_target = batch["chronological"] - batch["boneage"]
        else:
            raw_target = batch["boneage"] - batch["chronological"]
    else:
        raw_target = batch["boneage"]
    return target_normalizer.transform_tensor(raw_target)


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


def _set_progress_postfix(progress, total_loss: float, total_count: int) -> None:
    if progress is None:
        return
    running_loss = (total_loss / total_count) if total_count > 0 else None
    if running_loss is not None:
        progress.set_postfix(loss=f"{running_loss:.4f}")
    else:
        progress.set_postfix(loss="nan")


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
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.train(train)
    phase = "train" if train else "eval"
    amp_enabled = bool(amp) and device.type == "cuda"
    total_loss = 0.0
    total_count = 0
    y_true: list[float] = []
    y_pred: list[float] = []
    rows: list[dict[str, Any]] | None = [] if collect_predictions else None

    if total_epochs is not None and epoch is not None:
        progress_desc = f"{phase} {epoch}/{total_epochs}"
    elif epoch is not None:
        progress_desc = f"{phase} epoch={epoch}"
    else:
        progress_desc = phase
    progress = tqdm(
        loader,
        desc=progress_desc,
        total=len(loader),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )

    with _inference_context(train):
        for batch_index, batch in enumerate(progress):
            batch = move_batch_to_device(batch, device)
            if batch_index == 0:
                _log_first_batch_device(batch, model, device, logger, phase=phase, epoch=epoch)
            has_target_mask = batch["has_target"].view(-1).bool()

            if train:
                optimizer.zero_grad(set_to_none=True)

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

            if train and loss is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if gradient_clip and gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip and gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

            pred_boneage = decode_boneage_prediction(
                prediction,
                batch,
                target_mode,
                target_normalizer,
                relative_direction=relative_direction,
            )

            batch_size = prediction.shape[0]
            if loss is not None:
                valid_count = int(has_target_mask.sum().item())
                total_loss += float(loss.detach().item()) * valid_count
                total_count += valid_count

            if has_target_mask.any():
                y_true.extend(batch["boneage"][has_target_mask].detach().view(-1).cpu().tolist())
                y_pred.extend(pred_boneage[has_target_mask].detach().view(-1).cpu().tolist())

            if collect_predictions:
                has_target_cpu = batch["has_target"].detach().view(-1).cpu().numpy().astype(bool)
                boneage_cpu = batch["boneage"].detach().view(-1).cpu().numpy()
                pred_boneage_cpu = pred_boneage.detach().view(-1).cpu().numpy()
                chronological_cpu = batch["chronological"].detach().view(-1).cpu().numpy()
                male_index_cpu = batch["male_index"].detach().view(-1).cpu().numpy()

                for index in range(batch_size):
                    gt_value = float(boneage_cpu[index]) if has_target_cpu[index] else np.nan
                    pred_value = float(pred_boneage_cpu[index])
                    abs_error = abs(pred_value - gt_value) if not np.isnan(gt_value) else np.nan
                    if not np.isnan(gt_value):
                        chronological_value = float(chronological_cpu[index])
                        if relative_direction == "chronological_minus_boneage":
                            relative_gt = chronological_value - gt_value
                            relative_pred = chronological_value - pred_value
                        else:
                            relative_gt = gt_value - chronological_value
                            relative_pred = pred_value - chronological_value
                    else:
                        chronological_value = float(chronological_cpu[index])
                        relative_gt = np.nan
                        relative_pred = np.nan
                    rows.append(
                        {
                            "ID": batch["id"][index],
                            "gt_boneage": gt_value,
                            "pred_boneage": pred_value,
                            "abs_error": abs_error,
                            "sex": int(male_index_cpu[index]),
                            "chronological": chronological_value,
                            "gt_relative_boneage": relative_gt,
                            "pred_relative_boneage": relative_pred,
                        }
                    )

            if show_progress and ((batch_index + 1) % 10 == 0 or (batch_index + 1) == len(loader)):
                _set_progress_postfix(progress, total_loss=total_loss, total_count=total_count)

    if progress is not None:
        progress.close()

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_count, 1) if total_count > 0 else None

    if not collect_predictions:
        metrics["relative_age_error_corr"] = None
        metrics["relative_age_error_slope"] = None
        return metrics, pd.DataFrame()

    prediction_df = pd.DataFrame(rows)
    valid_df = prediction_df[prediction_df["gt_boneage"].notna()].copy()
    if len(valid_df) >= 2:
        relative_values = valid_df["gt_relative_boneage"].to_numpy(dtype=np.float32)
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
    return metrics, prediction_df
