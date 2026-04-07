from __future__ import annotations

import glob
import importlib
import json
import locale
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Any

import sys
import torch


@dataclass
class RuntimeInfo:
    python: str
    python_executable: str
    platform: str
    is_wsl: bool
    filesystem_encoding: str
    preferred_encoding: str
    stdout_encoding: str | None
    stderr_encoding: str | None
    torch_version: str
    torchvision_version: str
    torchaudio_version: str | None
    numpy_version: str | None
    pandas_version: str | None
    cuda_build: str | None
    torch_cuda_built: bool
    cuda_available: bool
    amp_available: bool
    compile_available: bool
    compile_triton_available: bool
    device_count: int
    device_names: list[str]
    requested_device: str
    selected_device: str
    deterministic: bool
    cudnn_benchmark: bool
    tf32_matmul: bool
    tf32_cudnn: bool
    float32_matmul_precision: str
    nvidia_smi_available: bool
    nvidia_smi_summary: str | None
    nvidia_smi_gpu_names: list[str]
    device_nodes: list[str]
    cuda_diagnostic: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompileInfo:
    requested: bool
    available: bool
    actually_used: bool
    mode: str
    backend: str = "inductor"
    effective_mode: str | None = None
    call_mode: str | None = None
    call_options: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    cudagraphs_enabled: bool | None = None
    disabled_features: list[str] = field(default_factory=list)
    safety_downgrade: bool = False
    options_override_used: bool = False

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


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def _device_nodes() -> list[str]:
    nodes = sorted(glob.glob("/dev/nvidia*"))
    if os.path.exists("/dev/dxg"):
        nodes.append("/dev/dxg")
    return nodes


def _probe_nvidia_smi() -> tuple[bool, str | None, list[str]]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return False, None, []

    command = [
        binary,
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        return False, f"nvidia-smi probe failed: {exc}", []

    if result.returncode != 0:
        raw_message = (result.stderr or result.stdout or "").strip() or f"returncode={result.returncode}"
        message = " ".join(part for part in raw_message.splitlines() if part.strip())
        return False, message, []

    gpu_names: list[str] = []
    driver_versions: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "," in line:
            name, driver = [part.strip() for part in line.split(",", 1)]
            gpu_names.append(name)
            driver_versions.append(driver)
        else:
            gpu_names.append(line)
    driver_versions = [value for value in driver_versions if value]
    summary_parts = []
    if driver_versions:
        summary_parts.append(f"driver={driver_versions[0]}")
    if gpu_names:
        summary_parts.append(f"gpus={gpu_names}")
    summary = " | ".join(summary_parts) if summary_parts else "nvidia-smi reachable"
    return True, summary, gpu_names


def _cuda_diagnostic(cuda_available: bool, nvidia_smi_available: bool, device_nodes: list[str]) -> str | None:
    if cuda_available:
        return None
    if nvidia_smi_available and not device_nodes:
        return "nvidia-smi 可见，但当前会话缺少 /dev/dxg 或 /dev/nvidia* 设备节点，torch 无法直接访问 CUDA 设备。"
    if nvidia_smi_available:
        return "nvidia-smi 可见，但 torch 仍无法初始化 CUDA；请检查驱动映射、WSL/容器权限以及当前 Python 运行环境。"
    return "当前会话未检测到可供 torch 使用的 NVIDIA 驱动或设备。"


def detect_runtime(
    requested_device: str | None = None,
    allow_cpu_fallback: bool = False,
    deterministic: bool = False,
) -> tuple[torch.device, RuntimeInfo]:
    normalized_device = _normalize_requested_device(requested_device)
    nvidia_smi_available, nvidia_smi_summary, nvidia_smi_gpu_names = _probe_nvidia_smi()
    device_nodes = _device_nodes()
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
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = not bool(deterministic)
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
        python_executable=sys.executable,
        platform=platform.platform(),
        is_wsl=_is_wsl(),
        filesystem_encoding=sys.getfilesystemencoding(),
        preferred_encoding=locale.getpreferredencoding(False),
        stdout_encoding=getattr(sys.stdout, "encoding", None),
        stderr_encoding=getattr(sys.stderr, "encoding", None),
        torch_version=torch.__version__,
        torchvision_version=_module_version("torchvision") or "unknown",
        torchaudio_version=_module_version("torchaudio"),
        numpy_version=_module_version("numpy"),
        pandas_version=_module_version("pandas"),
        cuda_build=torch.version.cuda,
        torch_cuda_built=torch.backends.cuda.is_built(),
        cuda_available=cuda_available,
        amp_available=hasattr(torch, "amp"),
        compile_available=hasattr(torch, "compile"),
        compile_triton_available=_cuda_compile_is_available(),
        device_count=device_count,
        device_names=names,
        requested_device=normalized_device,
        selected_device=str(device),
        deterministic=bool(deterministic),
        cudnn_benchmark=cudnn_benchmark,
        tf32_matmul=tf32_matmul,
        tf32_cudnn=tf32_cudnn,
        float32_matmul_precision=matmul_precision,
        nvidia_smi_available=nvidia_smi_available,
        nvidia_smi_summary=nvidia_smi_summary,
        nvidia_smi_gpu_names=nvidia_smi_gpu_names,
        device_nodes=device_nodes,
        cuda_diagnostic=_cuda_diagnostic(cuda_available, nvidia_smi_available, device_nodes),
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
        if sys.platform.startswith("win"):
            if cpu_count <= 8:
                workers = 1
            else:
                workers = min(2, max(1, cpu_count // 4))
        elif cpu_count <= 4:
            workers = 1
        elif cpu_count <= 8:
            workers = 2
        elif cpu_count <= 16:
            workers = 3
        else:
            workers = min(6, max(4, cpu_count // 4))
    if workers > 0:
        workers = min(workers, max(1, batch_size))

    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
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


def _compile_mode_options(mode: str) -> dict[str, Any]:
    if not hasattr(torch, "_inductor") or not hasattr(torch._inductor, "list_mode_options"):
        return {}
    try:
        options = torch._inductor.list_mode_options().get(mode, {})
    except Exception:
        return {}
    return dict(options)


def _available_compile_modes() -> set[str]:
    if not hasattr(torch, "_inductor") or not hasattr(torch._inductor, "list_mode_options"):
        return {"default"}
    try:
        modes = set(torch._inductor.list_mode_options().keys())
    except Exception:
        return {"default"}
    modes.add("default")
    return modes


def _has_multi_backbone_ensemble(model: torch.nn.Module) -> bool:
    return bool(
        getattr(model, "resnet", None) is not None
        and getattr(model, "efficientnet", None) is not None
    )


def _compile_disabled_features_text(features: list[str]) -> str:
    if not features:
        return "none"
    return ", ".join(features)


def _format_compile_options(options: dict[str, Any]) -> str:
    if not options:
        return "none"
    try:
        return json.dumps(options, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(options)


def _match_compile_mode_by_options(options: dict[str, Any]) -> str | None:
    if not hasattr(torch, "_inductor") or not hasattr(torch._inductor, "list_mode_options"):
        return "default" if not options else None
    try:
        mode_options = torch._inductor.list_mode_options()
    except Exception:
        return "default" if not options else None
    normalized = dict(options)
    if not normalized:
        return "default"
    for mode_name, candidate_options in mode_options.items():
        if dict(candidate_options) == normalized:
            return str(mode_name)
    if normalized.get("triton.cudagraphs") is False:
        trimmed = {key: value for key, value in normalized.items() if not (key == "triton.cudagraphs" and value is False)}
        for mode_name, candidate_options in mode_options.items():
            if "cudagraphs" not in str(mode_name):
                continue
            if dict(candidate_options) == trimmed:
                return str(mode_name)
    return None


def _log_compile_info(logger, info: CompileInfo, *, level: str = "info") -> None:
    log_method = getattr(logger, level, logger.info)
    log_method("torch.compile actually used: %s", info.actually_used)
    if info.reason:
        log_method("torch.compile reason: %s", info.reason)
    log_method(
        "torch.compile call: requested_mode=%s | effective_mode=%s | backend=%s | call_mode=%s | options=%s | cudagraphs=%s | disabled_features=%s | safety_downgrade=%s | options_override=%s",
        info.mode,
        info.effective_mode or info.mode,
        info.backend,
        info.call_mode or "none",
        _format_compile_options(info.call_options),
        info.cudagraphs_enabled,
        _compile_disabled_features_text(info.disabled_features),
        info.safety_downgrade,
        info.options_override_used,
    )


def _build_compile_kwargs(
    mode: str,
    mode_options: dict[str, Any],
    *,
    is_cuda: bool,
    disable_cudagraphs_for_safety: bool,
) -> tuple[dict[str, Any], str, str | None, dict[str, Any], bool | None, list[str], bool, bool]:
    compile_kwargs: dict[str, Any] = {}
    effective_mode = mode
    call_mode = None
    call_options: dict[str, Any] = {}
    disabled_features: list[str] = []
    options_override_used = False
    safety_downgrade = False
    cudagraphs_enabled = mode_options.get("triton.cudagraphs") if is_cuda else None
    if is_cuda and cudagraphs_enabled is None and "no-cudagraphs" in mode:
        cudagraphs_enabled = False
    cudagraphs_requested = bool(cudagraphs_enabled) if cudagraphs_enabled is not None else False

    if disable_cudagraphs_for_safety and cudagraphs_requested:
        safety_downgrade = True
        disabled_features.append("triton.cudagraphs")
        safe_options = dict(mode_options)
        safe_options["triton.cudagraphs"] = False
        matched_safe_mode = _match_compile_mode_by_options(safe_options)
        if matched_safe_mode is not None and matched_safe_mode != "default":
            compile_kwargs["mode"] = matched_safe_mode
            call_mode = matched_safe_mode
            effective_mode = matched_safe_mode
        else:
            compile_kwargs["options"] = safe_options
            call_options = dict(safe_options)
            options_override_used = True
            effective_mode = f"{mode} (no-cudagraphs safe)"
        cudagraphs_enabled = False
    elif mode != "default":
        compile_kwargs["mode"] = mode
        call_mode = mode

    return (
        compile_kwargs,
        effective_mode,
        call_mode,
        call_options,
        cudagraphs_enabled,
        disabled_features,
        safety_downgrade,
        options_override_used,
    )


def maybe_compile_model(
    model: torch.nn.Module,
    enabled: bool,
    logger,
    mode: str | None = None,
) -> tuple[torch.nn.Module, CompileInfo]:
    normalized_mode = (mode or "default").strip().lower() or "default"
    model_device = _model_device(model)
    torch_compile_available = hasattr(torch, "compile")
    triton_available = model_device.type != "cuda" or _cuda_compile_is_available()
    available_modes = _available_compile_modes()
    mode_available = normalized_mode in available_modes
    compile_available = bool(torch_compile_available and triton_available and mode_available)

    logger.info("torch.compile requested: %s", bool(enabled))
    logger.info("torch.compile available: %s", compile_available)

    if not enabled:
        info = CompileInfo(
            requested=False,
            available=compile_available,
            actually_used=False,
            mode=normalized_mode,
            effective_mode=normalized_mode,
            reason="disabled by config",
        )
        _log_compile_info(logger, info, level="info")
        return model, info
    if not torch_compile_available:
        info = CompileInfo(
            requested=True,
            available=False,
            actually_used=False,
            mode=normalized_mode,
            effective_mode=normalized_mode,
            reason="current torch build does not provide torch.compile",
        )
        _log_compile_info(logger, info, level="warning")
        return model, info
    if not mode_available:
        info = CompileInfo(
            requested=True,
            available=False,
            actually_used=False,
            mode=normalized_mode,
            effective_mode=normalized_mode,
            reason=f"unsupported compile mode: {normalized_mode}",
        )
        _log_compile_info(logger, info, level="warning")
        return model, info
    if not triton_available:
        info = CompileInfo(
            requested=True,
            available=False,
            actually_used=False,
            mode=normalized_mode,
            effective_mode=normalized_mode,
            reason="current CUDA environment lacks usable Triton support for torch.compile",
        )
        _log_compile_info(logger, info, level="warning")
        return model, info

    model_device = _model_device(model)
    try:
        mode_options = _compile_mode_options(normalized_mode)
        multi_backbone_ensemble = model_device.type == "cuda" and _has_multi_backbone_ensemble(model)
        (
            compile_kwargs,
            effective_mode,
            call_mode,
            call_options,
            cudagraphs_enabled,
            disabled_features,
            safety_downgrade,
            options_override_used,
        ) = (
            _build_compile_kwargs(
                normalized_mode,
                mode_options,
                is_cuda=model_device.type == "cuda",
                disable_cudagraphs_for_safety=multi_backbone_ensemble,
            )
        )
        if safety_downgrade:
            logger.warning(
                "torch.compile: 检测到双-backbone ensemble，已自动禁用 CUDAGraphs，以规避顺序分支前向时的输出覆盖问题。"
            )
        compiled = torch.compile(model, **compile_kwargs)
        info = CompileInfo(
            requested=True,
            available=True,
            actually_used=True,
            mode=normalized_mode,
            effective_mode=effective_mode,
            call_mode=call_mode,
            call_options=call_options,
            reason=None,
            cudagraphs_enabled=bool(cudagraphs_enabled) if cudagraphs_enabled is not None else None,
            disabled_features=disabled_features,
            safety_downgrade=safety_downgrade,
            options_override_used=options_override_used,
        )
        _log_compile_info(logger, info, level="info")
        return compiled, info
    except Exception as exc:  # pragma: no cover - 编译失败时的保护逻辑
        info = CompileInfo(
            requested=True,
            available=True,
            actually_used=False,
            mode=normalized_mode,
            effective_mode=effective_mode if "effective_mode" in locals() else normalized_mode,
            call_mode=call_mode if "call_mode" in locals() else None,
            call_options=call_options if "call_options" in locals() else {},
            reason=str(exc),
            cudagraphs_enabled=None,
            disabled_features=disabled_features if "disabled_features" in locals() else [],
            safety_downgrade=safety_downgrade if "safety_downgrade" in locals() else False,
            options_override_used=options_override_used if "options_override_used" in locals() else False,
        )
        _log_compile_info(logger, info, level="warning")
        return model, info


def get_cuda_memory_snapshot(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_mb": torch.cuda.memory_allocated(device_index) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device_index) / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device_index) / (1024**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device_index) / (1024**2),
    }


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
