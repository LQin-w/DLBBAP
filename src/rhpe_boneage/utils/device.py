from __future__ import annotations

import os
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
    cudnn_benchmark: bool
    tf32_matmul: bool
    tf32_cudnn: bool
    float32_matmul_precision: str

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

    cudnn_benchmark = False
    tf32_matmul = False
    tf32_cudnn = False
    matmul_precision = "default"
    if device.type == "cuda":
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
                tf32_cudnn = bool(torch.backends.cudnn.allow_tf32)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
            tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
            matmul_precision = "high"

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
        cudnn_benchmark=cudnn_benchmark,
        tf32_matmul=tf32_matmul,
        tf32_cudnn=tf32_cudnn,
        float32_matmul_precision=matmul_precision,
    )
    return device, runtime


def suggest_dataloader_kwargs(
    batch_size: int,
    use_cuda: bool,
    cpu_count: int | None = None,
) -> dict[str, Any]:
    if cpu_count is None:
        cpu_count = 0
        cpu_count = os.cpu_count() or 0

    if cpu_count <= 2:
        workers = 0
    else:
        # Windows uses spawn for DataLoader workers, so aggressive worker counts
        # can make the first batch look frozen for a long time.
        if sys.platform.startswith("win"):
            if cpu_count <= 8:
                workers = 2
            else:
                workers = min(4, max(2, cpu_count // 4))
        elif cpu_count <= 4:
            workers = 2
        elif cpu_count <= 8:
            workers = min(4, max(2, cpu_count // 2))
        else:
            workers = min(12, max(4, cpu_count // 2))
    if workers > 0:
        workers = min(workers, max(2, batch_size * 2))

    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4 if use_cuda else 2
    return kwargs


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _cuda_compile_is_available() -> bool:
    try:
        from torch.utils import _triton

        return bool(_triton.has_triton())
    except Exception:
        return False


def maybe_compile_model(model: torch.nn.Module, enabled: bool, logger) -> torch.nn.Module:
    if not enabled:
        logger.info("torch.compile: 配置关闭，跳过。")
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile: 当前 torch 不支持，自动降级。")
        return model
    model_device = _model_device(model)
    if model_device.type == "cuda" and not _cuda_compile_is_available():
        logger.warning("torch.compile: 当前 CUDA 环境缺少可用 Triton，自动降级。")
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
