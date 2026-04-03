from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import load_config, save_config
from ..data import RHPEBoneAgeDataset, build_dataset_index, build_manual_split_records
from ..data.dataset import DatasetStats
from ..data.transforms import build_geometric_transform, build_image_intensity_transform
from ..models import build_model
from ..utils import detect_runtime, ensure_dir, seed_everything, setup_logger, suggest_dataloader_kwargs, write_json
from ..utils.device import log_device_probe, maybe_compile_model
from ..utils.io import timestamp
from ..utils.plots import plot_history
from .engine import run_epoch, unwrap_model
from .losses import build_loss
from .normalization import ScalarNormalizer


def _safe_metric_value(value: float | None) -> float:
    return value if value is not None else math.nan


def _log_epoch_metrics(
    logger,
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    lr: float,
) -> None:
    logger.info(
        "Epoch %d | train_loss=%.4f | val_loss=%.4f | train_mae=%.4f | val_mae=%.4f | train_mad=%.4f | val_mad=%.4f | lr=%.6g",
        epoch,
        _safe_metric_value(train_metrics.get("loss")),
        _safe_metric_value(val_metrics.get("loss")),
        _safe_metric_value(train_metrics.get("mae")),
        _safe_metric_value(val_metrics.get("mae")),
        _safe_metric_value(train_metrics.get("mad")),
        _safe_metric_value(val_metrics.get("mad")),
        lr,
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
            default_prefetch = 4 if device.type == "cuda" else 2
            loader_kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", default_prefetch))
    else:
        loader_kwargs = suggest_dataloader_kwargs(
            batch_size=int(training_cfg["batch_size"]),
            use_cuda=device.type == "cuda",
        )

    if training_cfg.get("pin_memory") is not None:
        loader_kwargs["pin_memory"] = bool(training_cfg["pin_memory"])

    if loader_kwargs["num_workers"] > 0:
        if training_cfg.get("persistent_workers") is not None:
            loader_kwargs["persistent_workers"] = bool(training_cfg["persistent_workers"])
        if training_cfg.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(training_cfg["prefetch_factor"])
        elif device.type == "cuda":
            loader_kwargs["prefetch_factor"] = max(4, int(loader_kwargs.get("prefetch_factor", 2)))
    else:
        loader_kwargs.pop("prefetch_factor", None)
        loader_kwargs["persistent_workers"] = False

    if (
        device.type == "cuda"
        and loader_kwargs.get("pin_memory", False)
        and "pin_memory_device" in inspect.signature(DataLoader.__init__).parameters
    ):
        loader_kwargs["pin_memory_device"] = str(device)

    dataloaders = {}
    if "train" in datasets:
        dataloaders["train"] = DataLoader(
            datasets["train"],
            batch_size=int(training_cfg["batch_size"]),
            shuffle=True,
            **loader_kwargs,
        )
    if "val" in datasets:
        dataloaders["val"] = DataLoader(
            datasets["val"],
            batch_size=int(training_cfg["val_batch_size"]),
            shuffle=False,
            **loader_kwargs,
        )
    if "test" in datasets:
        dataloaders["test"] = DataLoader(
            datasets["test"],
            batch_size=int(training_cfg["test_batch_size"]),
            shuffle=False,
            **loader_kwargs,
        )
    return dataloaders, loader_kwargs


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    training_cfg = config["training"]
    params = [param for param in model.parameters() if param.requires_grad]
    name = training_cfg["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=training_cfg["lr"], weight_decay=training_cfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=training_cfg["lr"], weight_decay=training_cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=training_cfg["lr"],
            momentum=training_cfg["momentum"],
            weight_decay=training_cfg["weight_decay"],
            nesterov=True,
        )
    raise ValueError(f"不支持的优化器: {training_cfg['optimizer']}")


def _build_scheduler(optimizer, config: dict[str, Any]):
    training_cfg = config["training"]
    name = training_cfg["scheduler"].lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=training_cfg["min_lr"],
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_cfg["epochs"],
            eta_min=training_cfg["min_lr"],
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
            "数据检查 | split=%s | matched=%d | missing_image=%d | missing_csv=%d | missing_roi=%d | unreadable=%d",
            split,
            report["matched_records"],
            len(issues["missing_images"]),
            len(issues["missing_csv_records"]),
            len(issues["missing_roi_json"]),
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


def _resolve_runtime_settings(config: dict[str, Any]) -> tuple[str, bool]:
    runtime_cfg = config.get("runtime") or {}
    requested_device = str(runtime_cfg.get("device") or "cuda:0")
    allow_cpu_fallback = bool(runtime_cfg.get("allow_cpu_fallback", False))
    return requested_device, allow_cpu_fallback


def train_main(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config, checkpoint_state = _resolve_config(
        config_path=config_path,
        overrides=overrides,
        checkpoint_path=None,
    )
    run_dir = _prepare_run_dir(config, purpose="train")
    logger = setup_logger(run_dir)
    seed_everything(int(config["experiment"]["seed"]))

    requested_device, allow_cpu_fallback = _resolve_runtime_settings(config)
    device, runtime = detect_runtime(
        requested_device=requested_device,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    write_json(runtime.to_dict(), run_dir / "runtime.json")
    logger.info(
        "运行环境 | torch=%s | cuda_build=%s | cuda_available=%s | requested_device=%s | selected_device=%s | gpus=%s | cudnn_benchmark=%s | tf32_matmul=%s | tf32_cudnn=%s | matmul_precision=%s",
        runtime.torch_version,
        runtime.cuda_build,
        runtime.cuda_available,
        runtime.requested_device,
        runtime.selected_device,
        runtime.device_names,
        runtime.cudnn_benchmark,
        runtime.tf32_matmul,
        runtime.tf32_cudnn,
        runtime.float32_matmul_precision,
    )
    if runtime.requested_device.startswith("cuda") and device.type != "cuda":
        logger.warning("请求设备 %s 不可用，训练已回退到 CPU。", runtime.requested_device)

    payload, reports = _build_data_payload(config, run_dir, checkpoint_state=checkpoint_state)
    _log_reports(logger, reports)
    datasets, normalizers = _build_datasets(payload, config, checkpoint_state=checkpoint_state)
    dataloaders, loader_kwargs = _build_dataloaders(datasets, config, device)
    write_json(loader_kwargs, run_dir / "dataloader.json")

    model = build_model(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    log_device_probe(model, device, logger)
    model = maybe_compile_model(model, bool(config["training"]["compile"]), logger)
    criterion = build_loss(config["training"]["loss"], config["training"]["smooth_l1_beta"])
    optimizer = _build_optimizer(unwrap_model(model), config)
    scheduler = _build_scheduler(optimizer, config)
    scaler = None
    use_amp = device.type == "cuda" and bool(config["training"]["amp"])
    if use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    show_progress = bool(config["training"].get("progress_bar", True))
    total_epochs = int(config["training"]["epochs"])

    start_epoch = 1
    best_metric = None
    resume_checkpoint = config["training"].get("resume_checkpoint")
    if resume_checkpoint:
        resume_state = _load_checkpoint_state(resume_checkpoint)
        start_epoch = _restore_training_state(model, optimizer, scheduler, scaler, resume_state)
        best_metric = resume_state.get("best_metric")
        logger.info("已从 checkpoint 续训: %s | next_epoch=%d", resume_checkpoint, start_epoch)

    save_config(config, run_dir / "config.yaml")
    history_rows = []
    best_metric_name = config["training"]["best_metric"]
    best_checkpoint_path = run_dir / "best_model.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"

    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics, _ = run_epoch(
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
        )
        val_metrics, val_predictions = run_epoch(
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
        )

        if scheduler is not None:
            if config["training"]["scheduler"].lower() == "plateau":
                scheduler.step(val_metrics[best_metric_name] if val_metrics[best_metric_name] is not None else val_metrics["loss"])
            else:
                scheduler.step()

        current_metric = val_metrics[best_metric_name]
        if current_metric is not None and (best_metric is None or current_metric < best_metric):
            best_metric = current_metric
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

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_mad": train_metrics["mad"],
            "val_mad": val_metrics["mad"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(history_row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        _log_epoch_metrics(
            logger=logger,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=history_row["lr"],
        )

    history_df = pd.DataFrame(history_rows)
    plot_history(history_df, run_dir / "curves.png")

    best_state = _load_checkpoint_state(best_checkpoint_path)
    unwrap_model(model).load_state_dict(best_state["model"])
    val_metrics, val_predictions = run_epoch(
        model=model,
        loader=dataloaders["val"],
        criterion=criterion,
        device=device,
        target_mode=config["model"]["target_mode"],
        target_normalizer=normalizers["target"],
        train=False,
        relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
        epoch=9999,
        total_epochs=None,
        amp=use_amp,
        show_progress=show_progress,
        collect_predictions=True,
        logger=logger,
    )
    val_predictions.to_csv(run_dir / "val_predictions.csv", index=False)
    write_json(val_metrics, run_dir / "val_metrics.json")

    output = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "val_metrics": val_metrics,
    }
    if "test" in dataloaders:
        test_metrics, test_predictions = run_epoch(
            model=model,
            loader=dataloaders["test"],
            criterion=criterion,
            device=device,
            target_mode=config["model"]["target_mode"],
            target_normalizer=normalizers["target"],
            train=False,
            relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
            epoch=10000,
            total_epochs=None,
            amp=use_amp,
            show_progress=show_progress,
            collect_predictions=True,
            logger=logger,
        )
        test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)
        write_json(test_metrics, run_dir / "test_metrics.json")
        output["test_metrics"] = test_metrics
    return output


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
    seed_everything(int(config["experiment"]["seed"]))

    requested_device, allow_cpu_fallback = _resolve_runtime_settings(config)
    device, runtime = detect_runtime(
        requested_device=requested_device,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    write_json(runtime.to_dict(), run_dir / "runtime.json")
    logger.info(
        "运行环境 | torch=%s | cuda_build=%s | cuda_available=%s | requested_device=%s | selected_device=%s | gpus=%s | cudnn_benchmark=%s | tf32_matmul=%s | tf32_cudnn=%s | matmul_precision=%s",
        runtime.torch_version,
        runtime.cuda_build,
        runtime.cuda_available,
        runtime.requested_device,
        runtime.selected_device,
        runtime.device_names,
        runtime.cudnn_benchmark,
        runtime.tf32_matmul,
        runtime.tf32_cudnn,
        runtime.float32_matmul_precision,
    )
    payload, reports = _build_data_payload(config, run_dir, checkpoint_state=checkpoint_state, manual_split=manual_split)
    _log_reports(logger, reports)
    datasets, normalizers = _build_datasets(payload, config, checkpoint_state=checkpoint_state)
    dataloaders, loader_kwargs = _build_dataloaders(datasets, config, device)
    write_json(loader_kwargs, run_dir / "dataloader.json")

    if split not in dataloaders:
        raise ValueError(f"请求评估的 split 不存在: {split}")

    model = build_model(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    log_device_probe(model, device, logger)
    unwrap_model(model).load_state_dict(checkpoint_state["model"])
    criterion = build_loss(config["training"]["loss"], config["training"]["smooth_l1_beta"])
    use_amp = device.type == "cuda" and bool(config["training"]["amp"])
    show_progress = bool(config["training"].get("progress_bar", True))

    metrics, predictions = run_epoch(
        model=model,
        loader=dataloaders[split],
        criterion=criterion,
        device=device,
        target_mode=config["model"]["target_mode"],
        target_normalizer=normalizers["target"],
        train=False,
        relative_direction=config["model"].get("relative_target_direction", "boneage_minus_chronological"),
        epoch=0,
        total_epochs=None,
        amp=use_amp,
        show_progress=show_progress,
        collect_predictions=True,
        logger=logger,
    )
    predictions.to_csv(run_dir / f"{split}_predictions.csv", index=False)
    write_json(metrics, run_dir / f"{split}_metrics.json")
    save_config(config, run_dir / "config.yaml")
    logger.info("%s 评估完成 | metrics=%s", split, metrics)
    return {"run_dir": str(run_dir), "metrics": metrics}


def tune_main(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    import optuna

    base_config, _ = _resolve_config(config_path=config_path, overrides=overrides, checkpoint_path=None)
    tune_dir = _prepare_run_dir(base_config, purpose="optuna")
    logger = setup_logger(tune_dir)
    save_config(base_config, tune_dir / "base_config.yaml")

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
        seed_everything(int(trial_config["experiment"]["seed"]))
        requested_device, allow_cpu_fallback = _resolve_runtime_settings(trial_config)
        device, runtime = detect_runtime(
            requested_device=requested_device,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        write_json(runtime.to_dict(), run_dir / "runtime.json")
        logger_trial.info(
            "运行环境 | torch=%s | cuda_build=%s | cuda_available=%s | requested_device=%s | selected_device=%s | gpus=%s | cudnn_benchmark=%s | tf32_matmul=%s | tf32_cudnn=%s | matmul_precision=%s",
            runtime.torch_version,
            runtime.cuda_build,
            runtime.cuda_available,
            runtime.requested_device,
            runtime.selected_device,
            runtime.device_names,
            runtime.cudnn_benchmark,
            runtime.tf32_matmul,
            runtime.tf32_cudnn,
            runtime.float32_matmul_precision,
        )
        payload, reports = _build_data_payload(trial_config, run_dir)
        _log_reports(logger_trial, reports)
        datasets, normalizers = _build_datasets(payload, trial_config, checkpoint_state=None)
        dataloaders, _ = _build_dataloaders(datasets, trial_config, device)
        model = build_model(trial_config).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        log_device_probe(model, device, logger_trial)
        model = maybe_compile_model(model, bool(trial_config["training"]["compile"]), logger_trial)
        criterion = build_loss(trial_config["training"]["loss"], trial_config["training"]["smooth_l1_beta"])
        optimizer = _build_optimizer(unwrap_model(model), trial_config)
        scheduler = _build_scheduler(optimizer, trial_config)
        scaler = None
        use_amp = device.type == "cuda" and bool(trial_config["training"]["amp"])
        if use_amp:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        show_progress = bool(trial_config["training"].get("progress_bar", True))

        best_metric = None
        for epoch in range(1, int(trial_config["training"]["epochs"]) + 1):
            train_metrics, _ = run_epoch(
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
                total_epochs=int(trial_config["training"]["epochs"]),
                amp=use_amp,
                show_progress=show_progress,
                collect_predictions=False,
                logger=logger_trial,
            )
            val_metrics, _ = run_epoch(
                model=model,
                loader=dataloaders["val"],
                criterion=criterion,
                device=device,
                target_mode=trial_config["model"]["target_mode"],
                target_normalizer=normalizers["target"],
                train=False,
                relative_direction=trial_config["model"].get("relative_target_direction", "boneage_minus_chronological"),
                epoch=epoch,
                total_epochs=int(trial_config["training"]["epochs"]),
                amp=use_amp,
                show_progress=show_progress,
                collect_predictions=False,
                logger=logger_trial,
            )
            value = val_metrics["mae"] if val_metrics["mae"] is not None else 1e9
            trial.report(value, step=epoch)
            if scheduler is not None:
                if trial_config["training"]["scheduler"].lower() == "plateau":
                    scheduler.step(value)
                else:
                    scheduler.step()
            _log_epoch_metrics(
                logger=logger_trial,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=optimizer.param_groups[0]["lr"],
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
