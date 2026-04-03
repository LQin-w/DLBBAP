from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import sys
import torch


@dataclass
class RuntimeInfo:
    python: str
    torch_version: str
    torchvision_version: str
    cuda_build: str | None
    cuda_available: bool
    device_count: int
    device_names: list[str]
    requested_device: str
    selected_device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_requested_device(requested_device: str | None) -> str:
    normalized = (requested_device or "cuda:0").strip().lower()
    if not normalized:
        return "cuda:0"
    if normalized == "cuda":
        return "cuda:0"
    if normalized not in {"cpu"} and not normalized.startswith("cuda:"):
        raise ValueError(f"不支持的 device 配置: {requested_device}")
    return normalized


def detect_runtime(
    requested_device: str | None = None,
    allow_cpu_fallback: bool = False,
) -> tuple[torch.device, RuntimeInfo]:
    normalized_device = _normalize_requested_device(requested_device)
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    names: list[str] = []
    if cuda_available:
        for index in range(device_count):
            names.append(torch.cuda.get_device_name(index))

    if normalized_device == "cpu":
        device = torch.device("cpu")
    elif cuda_available:
        device = torch.device(normalized_device)
        if device.index is not None and device.index >= device_count:
            raise RuntimeError(
                f"请求设备 {normalized_device}，但当前仅检测到 {device_count} 张可见 GPU。"
            )
        torch.cuda.set_device(device)
    elif allow_cpu_fallback:
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            f"请求设备 {normalized_device}，但当前 torch.cuda.is_available() == False。"
        )

    runtime = RuntimeInfo(
        python=sys.version.replace("\n", " "),
        torch_version=torch.__version__,
        torchvision_version=__import__("torchvision").__version__,
        cuda_build=torch.version.cuda,
        cuda_available=cuda_available,
        device_count=device_count,
        device_names=names,
        requested_device=normalized_device,
        selected_device=str(device),
    )
    return device, runtime


def suggest_dataloader_kwargs(
    batch_size: int,
    use_cuda: bool,
    cpu_count: int | None = None,
) -> dict[str, Any]:
    if cpu_count is None:
        cpu_count = 0
        try:
            import os

            cpu_count = os.cpu_count() or 0
        except Exception:
            cpu_count = 0

    if cpu_count <= 2:
        workers = 0
    elif cpu_count <= 8:
        workers = min(2, max(1, cpu_count - 1))
    else:
        workers = min(8, max(2, cpu_count // 2))
        workers = min(workers, max(2, batch_size * 2))

    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def maybe_compile_model(model: torch.nn.Module, enabled: bool, logger) -> torch.nn.Module:
    if not enabled:
        logger.info("torch.compile: 配置关闭，跳过。")
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile: 当前 torch 不支持，自动降级。")
        return model
    try:
        compiled = torch.compile(model)
        logger.info("torch.compile: 已启用。")
        return compiled
    except Exception as exc:  # pragma: no cover - 编译失败时的保护逻辑
        logger.warning("torch.compile: 启用失败，自动降级。原因: %s", exc)
        return model


def log_device_probe(model: torch.nn.Module, device: torch.device, logger) -> None:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device

    if device.type != "cuda":
        logger.info("设备自检 | model_device=%s | selected_device=%s", model_device, device)
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    probe = torch.empty(1, device=device)
    properties = torch.cuda.get_device_properties(device_index)
    logger.info(
        "设备自检 | model_device=%s | probe_device=%s | current_device=%s | name=%s | total_memory_mb=%d | allocated_mb=%.1f | reserved_mb=%.1f",
        model_device,
        probe.device,
        torch.cuda.current_device(),
        properties.name,
        int(properties.total_memory / (1024**2)),
        torch.cuda.memory_allocated(device_index) / (1024**2),
        torch.cuda.memory_reserved(device_index) / (1024**2),
    )
