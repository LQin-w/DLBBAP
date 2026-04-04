from __future__ import annotations

import argparse
import copy
import json
import platform
import queue
import sys
import threading
import traceback
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, scrolledtext, simpledialog, ttk
from typing import Any

import yaml

try:
    from _bootstrap import bootstrap, run_cli
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap, run_cli


PRESET_OPTIONS: dict[str, list[str]] = {
    "runtime.device": ["cuda:0", "cuda:1", "cpu"],
    "data.global_crop_mode": ["bbox", "full"],
    "model.ensemble_mode": ["ensemble", "resnet", "efficientnet"],
    "model.resnet_name": ["resnet18", "resnet34", "resnet50"],
    "model.efficientnet_name": ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
    "model.branch_mode": ["global_local", "global_only", "local_only"],
    "model.target_mode": ["relative", "direct"],
    "model.relative_target_direction": ["boneage_minus_chronological", "chronological_minus_boneage"],
    "model.metadata.mode": ["simba_hybrid", "simba_multiplier", "mlp"],
    "model.local_branch.mode": ["patch_heatmap", "patch", "heatmap"],
    "training.optimizer": ["adamw", "adam", "sgd"],
    "training.scheduler": ["plateau", "cosine", "none"],
    "training.loss": ["smoothl1", "l1", "mse"],
    "training.best_metric": ["mae", "mad", "loss"],
    "optuna.direction": ["minimize", "maximize"],
}

STRICT_OPTIONS: dict[str, set[str]] = {
    "data.global_crop_mode": {"bbox", "full", "none", "image"},
    "model.ensemble_mode": {"ensemble", "resnet", "efficientnet"},
    "model.resnet_name": {"resnet18", "resnet34", "resnet50"},
    "model.efficientnet_name": {"efficientnet_b0", "efficientnet_b1", "efficientnet_b2"},
    "model.branch_mode": {"global_local", "global_only", "local_only"},
    "model.target_mode": {"relative", "direct"},
    "model.relative_target_direction": {"boneage_minus_chronological", "chronological_minus_boneage"},
    "model.metadata.mode": {"simba_hybrid", "simba_multiplier", "mlp"},
    "model.local_branch.mode": {"patch_heatmap", "patch", "heatmap"},
    "training.optimizer": {"adamw", "adam", "sgd"},
    "training.scheduler": {"plateau", "cosine", "none"},
    "training.loss": {"smoothl1", "l1", "mse"},
    "training.best_metric": {"mae", "mad", "loss"},
    "optuna.direction": {"minimize", "maximize"},
}

HIDDEN_UI_PREFIXES: tuple[str, ...] = ("optuna.",)
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
FONT_CANDIDATES_UI: tuple[str, ...] = (
    "Noto Sans CJK SC",
    "Noto Serif CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "AR PL UMing CN",
    "AR PL UKai CN",
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "DengXian",
    "PingFang SC",
    "Heiti SC",
    "SimHei",
    "Arial Unicode MS",
)
FONT_CANDIDATES_MONO: tuple[str, ...] = (
    "Sarasa Mono SC",
    "Noto Sans Mono CJK SC",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei Mono",
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Source Han Sans SC",
    "PingFang SC",
    "Heiti SC",
    "SimHei",
)

OPTION_META: dict[str, tuple[str, str]] = {
    "experiment.name": ("实验名称", "本次实验的名称前缀，用于输出目录与日志标识。"),
    "experiment.output_root": ("输出根目录", "训练产物保存的根目录。"),
    "experiment.seed": ("随机种子", "控制随机性，保证实验可复现。"),
    "runtime.device": ("运行设备", "训练使用的设备，例如 cuda:0 或 cpu。"),
    "runtime.allow_cpu_fallback": ("允许回退 CPU", "当请求 GPU 不可用时，是否自动回退到 CPU。"),
    "data.dataset_root": ("数据集根目录", "数据集所在根路径。"),
    "data.input_size": ("输入分辨率", "全局输入图像 resize 的边长。"),
    "data.local_patch_size": ("局部 patch 尺寸", "关键点局部裁剪 patch 的边长。"),
    "data.max_keypoints": ("最大关键点数", "单样本使用的关键点上限。"),
    "data.heatmap_sigma_ratio": ("热图 sigma 比例", "关键点热图高斯核与手部尺度的比例。"),
    "data.global_crop_mode": ("全局裁剪模式", "全局图像裁剪策略。"),
    "data.global_crop_margin_ratio": ("全局裁剪边距比例", "基于 bbox 裁剪时额外保留的上下文比例。"),
    "data.verify_images": ("图像有效性检查", "构建索引时是否逐张检查图像可读性。"),
    "model.ensemble_mode": ("集成模式", "选择 ResNet / EfficientNet / 双模型集成。"),
    "model.resnet_name": ("ResNet 主干", "全局分支使用的 ResNet 版本。"),
    "model.efficientnet_name": ("EfficientNet 主干", "全局分支使用的 EfficientNet 版本。"),
    "model.pretrained": ("使用预训练权重", "是否加载 torchvision 预训练权重。"),
    "model.branch_mode": ("分支模式", "使用全局分支、局部分支或两者联合。"),
    "model.target_mode": ("目标模式", "直接预测骨龄或预测相对骨龄偏差。"),
    "model.relative_target_direction": ("相对骨龄方向", "相对目标的正负方向定义。"),
    "model.global_dim": ("全局特征维度", "全局分支投影后的特征维度。"),
    "model.heatmap_guidance.enabled": ("热图引导开关", "是否在全局分支使用 heatmap 引导特征。"),
    "model.cbam.enabled": ("CBAM 总开关", "是否启用 CBAM 注意力模块。"),
    "model.cbam.global_branch": ("全局 CBAM", "是否在全局分支启用 CBAM。"),
    "model.cbam.local_branch": ("局部 CBAM", "是否在局部分支启用 CBAM。"),
    "model.metadata.enabled": ("元信息融合开关", "是否融合性别和真实年龄等元信息。"),
    "model.metadata.mode": ("元信息融合模式", "元信息编码策略（SIMBA 相关变体）。"),
    "model.metadata.hidden_dim": ("元信息隐藏维度", "元信息 MLP 的隐藏层维度。"),
    "model.metadata.gender_embedding_dim": ("性别嵌入维度", "性别 embedding 维度。"),
    "model.metadata.chronological_hidden_dim": ("年龄特征维度", "真实年龄投影后的特征维度。"),
    "model.metadata.dropout": ("元信息 dropout", "元信息分支的 dropout 比例。"),
    "model.local_branch.mode": ("局部分支输入模式", "局部分支使用 patch、heatmap 或二者拼接。"),
    "model.local_branch.feature_dim": ("局部特征维度", "局部分支输出特征维度。"),
    "model.local_branch.geometry_dim": ("几何特征维度", "ROI 几何编码向量维度。"),
    "model.local_branch.dropout": ("局部分支 dropout", "局部分支融合层 dropout 比例。"),
    "model.head.hidden_dim": ("回归头隐藏维度", "最终融合回归头的隐藏层维度。"),
    "model.head.dropout": ("回归头 dropout", "最终回归头的 dropout 比例。"),
    "augmentation.affine_p": ("仿射增强概率", "随机仿射变换执行概率。"),
    "augmentation.rotation_limit": ("旋转范围", "仿射增强旋转角度上限（度）。"),
    "augmentation.translate_limit": ("平移范围", "仿射增强平移比例上限。"),
    "augmentation.scale_limit": ("缩放范围", "仿射增强缩放比例上限。"),
    "augmentation.shear_limit": ("错切范围", "仿射增强错切角度上限。"),
    "augmentation.horizontal_flip": ("水平翻转", "是否在训练中启用随机水平翻转。"),
    "augmentation.use_noise": ("高斯噪声开关", "是否启用高斯噪声增强。"),
    "augmentation.noise_std_min": ("噪声下限", "高斯噪声标准差最小值。"),
    "augmentation.noise_std_max": ("噪声上限", "高斯噪声标准差最大值。"),
    "augmentation.noise_p": ("噪声概率", "高斯噪声增强执行概率。"),
    "augmentation.use_blur": ("模糊增强开关", "是否启用高斯模糊增强。"),
    "augmentation.blur_limit": ("模糊核上限", "高斯模糊 kernel 尺寸上限。"),
    "augmentation.blur_p": ("模糊概率", "高斯模糊增强执行概率。"),
    "training.epochs": ("训练轮数", "完整遍历训练集的轮次数。"),
    "training.batch_size": ("训练 batch 大小", "训练集 DataLoader 的 batch size。"),
    "training.val_batch_size": ("验证 batch 大小", "验证集 DataLoader 的 batch size。"),
    "training.test_batch_size": ("测试 batch 大小", "测试集 DataLoader 的 batch size。"),
    "training.optimizer": ("优化器", "训练使用的优化算法。"),
    "training.lr": ("学习率", "优化器初始学习率。"),
    "training.weight_decay": ("权重衰减", "L2 正则化强度。"),
    "training.momentum": ("动量", "SGD 优化器使用的动量参数。"),
    "training.scheduler": ("学习率调度器", "学习率衰减策略。"),
    "training.min_lr": ("最小学习率", "调度器允许下降到的最小学习率。"),
    "training.loss": ("损失函数", "回归训练使用的损失函数。"),
    "training.smooth_l1_beta": ("SmoothL1 beta", "SmoothL1 损失的 beta 参数。"),
    "training.amp": ("混合精度训练", "是否启用 AMP 以提高吞吐并降低显存。"),
    "training.gradient_clip": ("梯度裁剪阈值", "梯度范数裁剪上限，0 或 null 表示关闭。"),
    "training.compile": ("torch.compile", "是否对模型进行编译优化。"),
    "training.best_metric": ("最佳模型指标", "用于保存 best checkpoint 的验证指标名。"),
    "training.resume_checkpoint": ("续训 checkpoint", "从指定 checkpoint 恢复训练。"),
    "training.progress_bar": ("进度条开关", "是否显示 epoch 内 batch 进度条。"),
    "training.log_interval": ("日志间隔", "每隔多少个 batch 输出一次批次日志；0 表示仅输出首尾 batch。"),
    "training.workers_override": ("DataLoader 线程覆盖", "手动指定 DataLoader num_workers。"),
    "training.prefetch_factor": ("预取因子", "每个 worker 预取 batch 数。"),
    "training.persistent_workers": ("常驻 worker", "epoch 间保持 DataLoader worker 常驻。"),
    "training.pin_memory": ("固定内存", "是否启用 pin_memory 加速主机到 GPU 拷贝。"),
    "debug.limit_train_samples": ("训练样本限制", "仅用于调试，限制训练样本数量。"),
    "debug.limit_val_samples": ("验证样本限制", "仅用于调试，限制验证样本数量。"),
    "debug.limit_test_samples": ("测试样本限制", "仅用于调试，限制测试样本数量。"),
    "optuna.direction": ("调参优化方向", "Optuna 目标方向，minimize 或 maximize。"),
    "optuna.n_trials": ("调参试验次数", "Optuna 运行的 trial 总数。"),
    "optuna.timeout": ("调参超时时间", "Optuna 总超时（秒），null 表示不限制。"),
    "optuna.epochs_per_trial": ("每轮 trial epoch 数", "每个 Optuna trial 的训练 epoch 数。"),
}

FALLBACK_NAME: dict[str, str] = {
    "name": "名称",
    "enabled": "开关",
    "mode": "模式",
    "dropout": "Dropout",
    "hidden_dim": "隐藏维度",
    "device": "设备",
    "epochs": "轮数",
    "batch_size": "批大小",
    "lr": "学习率",
}


def _flatten_config(config: dict[str, Any], prefix: str = "") -> OrderedDict[str, Any]:
    flat: OrderedDict[str, Any] = OrderedDict()
    for key, value in config.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, dotted))
        else:
            flat[dotted] = value
    return flat


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"配置文件顶层必须是字典: {path}")
    return data


def _build_train_ui_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    merged_config: dict[str, Any] = {}
    if DEFAULT_SCHEMA_PATH.exists():
        merged_config = _deep_merge(merged_config, _load_yaml_dict(DEFAULT_SCHEMA_PATH))
    selected_config = _load_yaml_dict(config_path)
    merged_config = _deep_merge(merged_config, selected_config)
    return merged_config, selected_config


def _assign_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def _is_visible_in_train_ui(dotted_key: str) -> bool:
    return not any(dotted_key.startswith(prefix) for prefix in HIDDEN_UI_PREFIXES)


def _to_display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _scalar_to_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return None
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def _float_suggestions(value: float) -> list[str]:
    if value == 0:
        return ["0.0", "0.1", "1.0"]
    candidates = [value, value * 0.5, value * 2.0]
    ordered: list[str] = []
    for item in candidates:
        shown = f"{item:.6g}"
        if shown not in ordered:
            ordered.append(shown)
    return ordered


def _int_suggestions(value: int) -> list[str]:
    if value <= 0:
        candidates = [value, 1, 2, 4]
    else:
        candidates = [value, max(1, value // 2), value * 2]
    ordered: list[str] = []
    for item in candidates:
        shown = str(int(item))
        if shown not in ordered:
            ordered.append(shown)
    return ordered


def _build_options(path: str, value: Any) -> list[str]:
    options = list(PRESET_OPTIONS.get(path, []))
    if isinstance(value, bool):
        options.extend(["true", "false"])
    elif value is None:
        options.append("null")
    elif isinstance(value, int) and not isinstance(value, bool):
        options.extend(_int_suggestions(value))
    elif isinstance(value, float):
        options.extend(_float_suggestions(value))
    else:
        options.append(_to_display_value(value))

    deduped: list[str] = []
    for item in options:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _get_option_meta(path: str) -> tuple[str, str]:
    meta = OPTION_META.get(path)
    if meta is not None:
        return meta
    key = path.split(".")[-1]
    cn_name = FALLBACK_NAME.get(key, key)
    cn_desc = f"自定义参数 `{path}`。该值会在点击开始训练时作为配置覆盖项传入。"
    return cn_name, cn_desc


def _validate_ui_value(dotted_key: str, value: Any) -> None:
    allowed = STRICT_OPTIONS.get(dotted_key)
    if allowed is None:
        return
    normalized = str(value).strip().lower()
    if normalized in allowed:
        return
    allowed_text = ", ".join(sorted(allowed))
    raise ValueError(f"{dotted_key} 仅支持以下取值: {allowed_text}")


def _pick_available_font(candidates: tuple[str, ...], available_fonts: dict[str, str]) -> str | None:
    for candidate in candidates:
        matched = available_fonts.get(candidate.casefold())
        if matched is not None:
            return matched
    return None


def _set_tcl_system_encoding(root: tk.Misc) -> str:
    """Force Tcl/Tk to use UTF-8 on Unix-like systems to avoid \\uXXXX fallback rendering."""

    try:
        current_encoding = str(root.tk.call("encoding", "system"))
    except tk.TclError:
        return ""

    normalized = current_encoding.strip().lower()
    if normalized in {"utf-8", "utf8"}:
        return current_encoding

    preferred_encodings = ("utf-8",)
    if platform.system().lower().startswith("win"):
        preferred_encodings = ("utf-8", current_encoding)

    for encoding_name in preferred_encodings:
        try:
            root.tk.call("encoding", "system", encoding_name)
            return str(root.tk.call("encoding", "system"))
        except tk.TclError:
            continue
    return current_encoding


def _configure_tk_font_fallback(root: tk.Misc) -> dict[str, str]:
    """Prefer installed CJK-capable fonts, but keep Tk defaults as a safe fallback."""

    available_fonts = {family.casefold(): family for family in tkfont.families(root)}
    ui_family = _pick_available_font(FONT_CANDIDATES_UI, available_fonts)
    mono_family = _pick_available_font(FONT_CANDIDATES_MONO, available_fonts) or ui_family

    named_fonts = ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont", "TkTooltipFont")
    if ui_family:
        for font_name in named_fonts:
            try:
                tkfont.nametofont(font_name).configure(family=ui_family)
            except tk.TclError:
                continue
    if mono_family:
        try:
            tkfont.nametofont("TkFixedFont").configure(family=mono_family)
        except tk.TclError:
            pass

    return {
        "ui_family": ui_family or "",
        "mono_family": mono_family or ui_family or "",
    }


def _apply_ttk_font_styles(root: tk.Misc, font_info: dict[str, str]) -> None:
    """ttk themes on Linux do not always honor Tk named fonts, so configure styles directly."""

    default_font = tkfont.nametofont("TkDefaultFont")
    text_font = tkfont.nametofont("TkTextFont")
    fixed_font = tkfont.nametofont("TkFixedFont")
    style = ttk.Style(root)
    style.configure(".", font=default_font)
    for style_name in ("TLabel", "TButton", "TCheckbutton", "TRadiobutton", "TMenubutton", "TLabelframe"):
        style.configure(style_name, font=default_font)
    for style_name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(style_name, font=text_font)
    style.configure("Treeview", font=text_font)
    style.configure("Treeview.Heading", font=default_font)

    default_font_desc = (default_font.cget("family"), default_font.cget("size"))
    fixed_font_desc = (fixed_font.cget("family"), fixed_font.cget("size"))
    root.option_add("*Font", default_font_desc)
    root.option_add("*Text.font", fixed_font_desc)
    root.option_add("*Entry.font", text_font)
    root.option_add("*Listbox.font", text_font)
    root.option_add("*TCombobox*Listbox.font", text_font)


class _UiTextStream:
    """Tee stdout/stderr to the terminal and to the UI log box with UTF-8-safe writes."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, owner: "TrainUI", original_stream) -> None:
        self.owner = owner
        self.original_stream = original_stream

    def write(self, data: Any) -> int:
        if data is None:
            return 0
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = str(data)
        if not text:
            return 0
        if self.original_stream is not None:
            try:
                self.original_stream.write(text)
            except UnicodeEncodeError:
                safe_text = text.encode(
                    getattr(self.original_stream, "encoding", "utf-8") or "utf-8",
                    errors="replace",
                ).decode(getattr(self.original_stream, "encoding", "utf-8") or "utf-8", errors="replace")
                self.original_stream.write(safe_text)
        if getattr(self.owner, "_output_capture_enabled", False):
            self.owner.enqueue_output(text)
        return len(text)

    def flush(self) -> None:
        if self.original_stream is not None and hasattr(self.original_stream, "flush"):
            self.original_stream.flush()

    def isatty(self) -> bool:
        if self.original_stream is not None and hasattr(self.original_stream, "isatty"):
            return bool(self.original_stream.isatty())
        return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        if self.original_stream is None or not hasattr(self.original_stream, "fileno"):
            raise OSError("underlying stream has no file descriptor")
        return self.original_stream.fileno()

    def __getattr__(self, item: str) -> Any:
        return getattr(self.original_stream, item)


class TrainUI:
    def __init__(self, root: tk.Tk, config_path: str) -> None:
        self.root = root
        self.root.title("RHPE BoneAge Training UI")
        self.root.geometry("980x900")
        self.tk_system_encoding = _set_tcl_system_encoding(self.root)
        self.font_info = _configure_tk_font_fallback(self.root)
        _apply_ttk_font_styles(self.root, self.font_info)

        self.config_path_var = tk.StringVar(value=config_path)
        self.status_var = tk.StringVar(value="就绪")
        self.widgets: dict[str, ttk.Combobox] = {}
        self.base_flat: OrderedDict[str, Any] = OrderedDict()
        self.loaded_config: dict[str, Any] = {}
        self.running = False
        self.log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._log_flush_scheduled = False
        self._max_log_chars = 200_000
        self._output_capture_enabled = True
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_proxy = _UiTextStream(self, self._original_stdout)
        self._stderr_proxy = _UiTextStream(self, self._original_stderr)

        self._build_layout()
        self._install_output_redirects()
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._load_config_into_form(config_path)
        if self.tk_system_encoding:
            self.enqueue_output(f"[UI] Tcl/Tk system encoding: {self.tk_system_encoding}\n")
        if self.font_info["ui_family"]:
            self.enqueue_output(f"[UI] 已启用中文字体: {self.font_info['ui_family']}\n")
        else:
            self.enqueue_output("[UI] 未找到显式中文字体，当前使用 Tk 默认字体。\n")

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="配置文件").pack(side=tk.LEFT)
        config_entry = ttk.Entry(top, textvariable=self.config_path_var)
        config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

        ttk.Button(top, text="选择文件", command=self._choose_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="保存配置", command=self._save_current_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="重新加载", command=self._reload_current_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="选择续训点", command=self._choose_resume_checkpoint).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="清空续训", command=self._clear_resume_checkpoint).pack(side=tk.LEFT)

        body = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        body.pack(fill=tk.BOTH, expand=True)

        pane = ttk.Panedwindow(body, orient=tk.VERTICAL)
        pane.pack(fill=tk.BOTH, expand=True)

        form_container = ttk.Frame(pane)
        log_container = ttk.LabelFrame(pane, text="训练输出", padding=(8, 8))
        pane.add(form_container, weight=4)
        pane.add(log_container, weight=2)

        self.canvas = tk.Canvas(form_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(form_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.form_frame = ttk.Frame(self.canvas)

        self.form_frame.bind(
            "<Configure>",
            lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.form_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        log_toolbar = ttk.Frame(log_container)
        log_toolbar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(log_toolbar, text="stdout / stderr / logger").pack(side=tk.LEFT)
        ttk.Button(log_toolbar, text="清空输出", command=self._clear_output).pack(side=tk.RIGHT)

        log_font = tkfont.nametofont("TkFixedFont").copy()
        log_font.configure(size=10)
        self.output_text = scrolledtext.ScrolledText(
            log_container,
            wrap=tk.WORD,
            height=12,
            state=tk.DISABLED,
            font=log_font,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(self.root, padding=10)
        bottom.pack(fill=tk.X)

        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=180)
        self.progress.pack(side=tk.RIGHT, padx=(8, 8))
        self.run_button = ttk.Button(bottom, text="开始训练", command=self._start_training)
        self.run_button.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(bottom, text="恢复默认值", command=self._reset_to_defaults).pack(side=tk.RIGHT)

    def _install_output_redirects(self) -> None:
        sys.stdout = self._stdout_proxy
        sys.stderr = self._stderr_proxy

    def _restore_output_redirects(self) -> None:
        if sys.stdout is self._stdout_proxy:
            sys.stdout = self._original_stdout
        if sys.stderr is self._stderr_proxy:
            sys.stderr = self._original_stderr

    def _handle_close(self) -> None:
        self._output_capture_enabled = False
        self._restore_output_redirects()
        self.root.destroy()

    def enqueue_output(self, text: str) -> None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        self.log_queue.put(normalized)
        if not self._log_flush_scheduled:
            self._log_flush_scheduled = True
            self.root.after(30, self._flush_output)

    def _flush_output(self) -> None:
        chunks: list[str] = []
        while True:
            try:
                chunks.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        self._log_flush_scheduled = False
        if not chunks:
            return

        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, "".join(chunks))
        last_line = int(self.output_text.index("end-1c").split(".")[0])
        max_lines = max(1000, self._max_log_chars // 80)
        if last_line > max_lines:
            self.output_text.delete("1.0", f"{last_line - max_lines}.0")
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

        if not self.log_queue.empty():
            self._log_flush_scheduled = True
            self.root.after(30, self._flush_output)

    def _clear_output(self) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def _choose_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not selected:
            return
        self.config_path_var.set(selected)
        self._load_config_into_form(selected)

    def _reload_current_config(self) -> None:
        self._load_config_into_form(self.config_path_var.get().strip())

    def _set_field_value(self, dotted_key: str, value: str) -> None:
        widget = self.widgets.get(dotted_key)
        if widget is None:
            raise KeyError(f"表单中不存在配置项: {dotted_key}")
        widget.set(value)

    def _choose_resume_checkpoint(self) -> None:
        initial_value = ""
        widget = self.widgets.get("training.resume_checkpoint")
        if widget is not None:
            current_value = widget.get().strip()
            if current_value and current_value.lower() != "null":
                initial_value = current_value
        initial_dir = ""
        initial_file = ""
        if initial_value:
            candidate = Path(initial_value)
            if candidate.is_file():
                initial_dir = str(candidate.parent)
                initial_file = candidate.name
        if not initial_dir:
            initial_dir = str(Path("outputs").resolve())

        selected = filedialog.askopenfilename(
            title="选择续训 checkpoint",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[
                ("PyTorch checkpoints", "*.pt *.pth *.ckpt"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        try:
            self._set_field_value("training.resume_checkpoint", selected)
        except KeyError as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        self.status_var.set(f"已选择续训 checkpoint: {selected}")

    def _clear_resume_checkpoint(self) -> None:
        try:
            self._set_field_value("training.resume_checkpoint", "null")
        except KeyError as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        self.status_var.set("已清空续训 checkpoint，将从头开始训练。")

    def _clear_form(self) -> None:
        for child in self.form_frame.winfo_children():
            child.destroy()
        self.widgets.clear()

    def _collect_current_config(self) -> dict[str, Any]:
        current_config = copy.deepcopy(self.loaded_config)
        for dotted_key, base_value in self.base_flat.items():
            widget = self.widgets.get(dotted_key)
            if widget is None:
                continue
            current_raw = widget.get()
            if current_raw.strip() == "":
                parsed_value = base_value
            else:
                parsed_value = _parse_value(current_raw)
                if parsed_value is None and base_value is not None:
                    raise ValueError(f"{dotted_key} 不能设置为 null，请填写有效值或恢复默认值。")
                _validate_ui_value(dotted_key, parsed_value)
            _assign_nested_value(current_config, dotted_key, parsed_value)
        return current_config

    def _save_current_config(self) -> None:
        config_path = self.config_path_var.get().strip()
        if not config_path:
            messagebox.showerror("保存失败", "当前没有可保存的配置文件路径。")
            return
        current_path = Path(config_path)
        default_name = current_path.name or "config.yaml"
        file_name = simpledialog.askstring(
            "重命名配置文件",
            "请输入保存文件名：",
            parent=self.root,
            initialvalue=default_name,
        )
        if file_name is None:
            self.status_var.set("已取消保存配置。")
            return
        normalized_name = file_name.strip()
        if not normalized_name:
            messagebox.showerror("保存失败", "文件名不能为空。")
            return
        path = current_path.parent / normalized_name
        if path.suffix.lower() not in {".yaml", ".yml"}:
            path = path.with_suffix(".yaml")
        try:
            config_to_save = self._collect_current_config()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config_to_save, handle, allow_unicode=True, sort_keys=False)
        except ValueError as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("保存失败", f"无法保存配置文件:\n{exc}")
            return
        self.config_path_var.set(str(path))
        self._load_config_into_form(str(path))
        self.status_var.set(f"配置已保存到: {path}")

    def _load_config_into_form(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            messagebox.showerror("配置文件不存在", f"找不到配置文件: {path}")
            return
        try:
            merged_config, selected_config = _build_train_ui_config(path)
        except Exception as exc:
            messagebox.showerror("读取失败", f"无法读取配置文件:\n{exc}")
            return

        self._clear_form()
        self.loaded_config = copy.deepcopy(merged_config)
        merged_flat = _flatten_config(merged_config)
        selected_flat = _flatten_config(selected_config)
        hidden_count = 0
        added_from_default = 0
        self.base_flat = OrderedDict()
        for dotted_key, value in merged_flat.items():
            if not _is_visible_in_train_ui(dotted_key):
                hidden_count += 1
                continue
            if dotted_key not in selected_flat:
                added_from_default += 1
            self.base_flat[dotted_key] = value

        headers = ["配置路径", "中文名称", "参数取值", "参数释义"]
        for index, header in enumerate(headers):
            ttk.Label(self.form_frame, text=header).grid(
                row=0,
                column=index,
                sticky="w",
                padx=(0, 8),
                pady=(2, 8),
            )

        row = 1
        for dotted_key, value in self.base_flat.items():
            cn_name, cn_desc = _get_option_meta(dotted_key)
            ttk.Label(self.form_frame, text=dotted_key).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=4)
            ttk.Label(self.form_frame, text=cn_name).grid(row=row, column=1, sticky="nw", padx=(0, 8), pady=4)
            options = _build_options(dotted_key, value)
            combo = ttk.Combobox(self.form_frame, values=options, state="normal")
            combo.set(_to_display_value(value))
            combo.grid(row=row, column=2, sticky="ew", padx=(0, 8), pady=4)
            self.widgets[dotted_key] = combo
            ttk.Label(
                self.form_frame,
                text=cn_desc,
                justify=tk.LEFT,
                wraplength=520,
            ).grid(row=row, column=3, sticky="nw", padx=(0, 8), pady=4)
            row += 1

        self.form_frame.columnconfigure(0, weight=2)
        self.form_frame.columnconfigure(1, weight=2)
        self.form_frame.columnconfigure(2, weight=1)
        self.form_frame.columnconfigure(3, weight=4)
        self.status_var.set(
            f"已加载 {len(self.base_flat)} 个训练参数，补全 {added_from_default} 个默认参数，隐藏 {hidden_count} 个当前训练模式不生效的参数。"
        )

    def _reset_to_defaults(self) -> None:
        for dotted_key, value in self.base_flat.items():
            widget = self.widgets.get(dotted_key)
            if widget is not None:
                widget.set(_to_display_value(value))
        self.status_var.set("参数已恢复为配置默认值。")

    def _collect_overrides(self) -> list[str]:
        overrides: list[str] = []
        for dotted_key, base_value in self.base_flat.items():
            widget = self.widgets.get(dotted_key)
            if widget is None:
                continue
            current_raw = widget.get()
            if current_raw.strip() == "":
                # An empty editable combobox should behave like "keep current value",
                # otherwise required numeric fields can accidentally become None.
                continue
            parsed_value = _parse_value(current_raw)
            if parsed_value is None and base_value is not None:
                raise ValueError(f"{dotted_key} 不能设置为 null，请填写有效值或恢复默认值。")
            _validate_ui_value(dotted_key, parsed_value)
            if parsed_value != base_value:
                overrides.append(f"{dotted_key}={_scalar_to_override(parsed_value)}")
        return overrides

    def _set_running(self, running: bool, message: str) -> None:
        self.running = running
        self.status_var.set(message)
        self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        if running:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _handle_training_success(self, run_dir: str) -> None:
        self._set_running(False, f"训练完成。输出目录: {run_dir}")
        self.enqueue_output(f"\n[UI] 训练完成，输出目录: {run_dir}\n")
        messagebox.showinfo("训练完成", f"训练已完成。\n输出目录:\n{run_dir}")

    def _handle_training_error(self, error_text: str) -> None:
        self._set_running(False, "训练失败。")
        self.enqueue_output(f"\n[UI] 训练失败: {error_text}\n")
        messagebox.showerror("训练失败", error_text)

    def _start_training(self) -> None:
        if self.running:
            return
        config_path = self.config_path_var.get().strip()
        if not Path(config_path).exists():
            messagebox.showerror("配置错误", f"配置文件不存在: {config_path}")
            return
        try:
            overrides = self._collect_overrides()
        except ValueError as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        self._set_running(True, "训练启动中...")
        self.enqueue_output(
            f"\n{'=' * 96}\n"
            f"[UI] 启动训练 | config={config_path} | overrides={len(overrides)}\n"
        )
        if overrides:
            for item in overrides:
                self.enqueue_output(f"[UI] override | {item}\n")

        def _worker() -> None:
            try:
                from rhpe_boneage.training.runner import train_main

                result = train_main(config_path=config_path, overrides=overrides)
                run_dir = result.get("run_dir", "")
                self.root.after(0, self._handle_training_success, run_dir)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                self.root.after(0, self._handle_training_error, str(exc))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()


def main() -> None:
    bootstrap()

    parser = argparse.ArgumentParser(description="训练前 UI 参数配置面板")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="覆盖配置，格式 key=value")
    parser.add_argument("--auto-run", action="store_true", help="不启动图形界面，直接按当前参数启动训练")
    args = parser.parse_args()

    if args.auto_run:
        from rhpe_boneage.training.runner import train_main

        train_main(config_path=args.config, overrides=args.overrides)
        return

    root = tk.Tk()
    TrainUI(root=root, config_path=args.config)
    root.minsize(747, 620)
    root.mainloop()


if __name__ == "__main__":
    run_cli(main)
