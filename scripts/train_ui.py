from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import queue
import re
import sys
import threading
import time
import traceback
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, scrolledtext, simpledialog, ttk
from typing import Any

import yaml

try:
    from _bootstrap import bootstrap, run_cli
    from ui_text import UITextManager, normalize_visible_text
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap, run_cli
    from scripts.ui_text import UITextManager, normalize_visible_text


PRESET_OPTIONS: dict[str, list[str]] = {
    "experiment.mode": ["enhanced", "simba", "bonet_like"],
    "data.global_crop_mode": ["bbox", "full"],
    "data.normalization.source": ["auto_train_stats", "manual"],
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
    "training.compile_mode": ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs", "lite"],
}

STRICT_OPTIONS: dict[str, set[str]] = {
    key: {item.lower() for item in values}
    for key, values in PRESET_OPTIONS.items()
}

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_SCIENTIFIC_NOTATION_PATTERN = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))[eE][+-]?\d+$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
_WSL_MOUNT_PATTERN = re.compile(r"^/mnt/(?P<drive>[A-Za-z])/(?P<rest>.*)$")
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
SECTION_TITLE_KEYS: dict[str, str] = {
    "basic": "section.basic",
    "method": "section.method",
    "input": "section.input",
    "augmentation": "section.augmentation",
    "output": "section.output",
    "advanced": "section.advanced",
}
SECTION_ORDER: tuple[str, ...] = ("basic", "method", "input", "augmentation", "output", "advanced")


@dataclass(frozen=True)
class UiFieldSpec:
    path: str
    group: str
    kind: str
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    increment: float | None = None
    allow_none: bool = False
    width: int = 18


@dataclass
class FieldBinding:
    spec: UiFieldSpec
    variable: tk.Variable
    widget: tk.Widget
    name_label: ttk.Label
    desc_label: ttk.Label


def _enum_field(path: str, group: str, width: int = 20) -> UiFieldSpec:
    return UiFieldSpec(path=path, group=group, kind="enum", options=tuple(PRESET_OPTIONS[path]), width=width)


def _bool_field(path: str, group: str) -> UiFieldSpec:
    return UiFieldSpec(path=path, group=group, kind="bool", width=8)


def _int_field(
    path: str,
    group: str,
    *,
    minimum: int,
    maximum: int,
    increment: int = 1,
    width: int = 10,
) -> UiFieldSpec:
    return UiFieldSpec(
        path=path,
        group=group,
        kind="int",
        minimum=float(minimum),
        maximum=float(maximum),
        increment=float(increment),
        width=width,
    )


def _float_field(
    path: str,
    group: str,
    *,
    minimum: float,
    maximum: float,
    increment: float,
    allow_none: bool = False,
    width: int = 12,
) -> UiFieldSpec:
    return UiFieldSpec(
        path=path,
        group=group,
        kind="float",
        minimum=minimum,
        maximum=maximum,
        increment=increment,
        allow_none=allow_none,
        width=width,
    )


def _text_field(path: str, group: str, width: int = 26) -> UiFieldSpec:
    return UiFieldSpec(path=path, group=group, kind="text", width=width)


VISIBLE_FIELD_SPECS: tuple[UiFieldSpec, ...] = (
    _bool_field("training.compile", "basic"),
    _bool_field("training.amp", "basic"),
    _text_field("runtime.device", "basic", width=18),
    _bool_field("runtime.allow_cpu_fallback", "basic"),
    _enum_field("experiment.mode", "basic"),
    _text_field("experiment.name", "basic", width=30),
    _int_field("experiment.seed", "basic", minimum=0, maximum=99999999),
    _int_field("training.epochs", "basic", minimum=1, maximum=2000),
    _int_field("training.batch_size", "basic", minimum=1, maximum=512),
    _int_field("training.gradient_accumulation_steps", "basic", minimum=1, maximum=128),
    _enum_field("training.optimizer", "basic"),
    _float_field("training.lr", "basic", minimum=1e-7, maximum=1.0, increment=1e-5, width=14),
    _float_field("training.weight_decay", "basic", minimum=0.0, maximum=1.0, increment=1e-5, width=14),
    _enum_field("training.scheduler", "basic"),
    _int_field("training.warmup_epochs", "basic", minimum=0, maximum=200),
    _enum_field("training.loss", "basic"),
    _enum_field("model.ensemble_mode", "method"),
    _enum_field("model.resnet_name", "method"),
    _enum_field("model.efficientnet_name", "method"),
    _bool_field("model.pretrained", "method"),
    _enum_field("model.branch_mode", "method"),
    _enum_field("model.target_mode", "method"),
    _enum_field("model.relative_target_direction", "method", width=28),
    _bool_field("model.metadata.enabled", "method"),
    _enum_field("model.metadata.mode", "method"),
    _bool_field("model.heatmap_guidance.enabled", "method"),
    _bool_field("model.cbam.enabled", "method"),
    _bool_field("model.cbam.global_branch", "method"),
    _bool_field("model.cbam.local_branch", "method"),
    _int_field("data.input_size", "input", minimum=64, maximum=1024, increment=16),
    _int_field("data.local_patch_size", "input", minimum=16, maximum=256, increment=8),
    _enum_field("data.global_crop_mode", "input"),
    _float_field("data.global_crop_margin_ratio", "input", minimum=0.0, maximum=1.0, increment=0.01),
    _float_field("data.heatmap_sigma_ratio", "input", minimum=0.0, maximum=1.0, increment=0.005),
    _float_field("data.heatmap_sigma_min", "input", minimum=0.0, maximum=64.0, increment=0.5),
    _enum_field("data.normalization.source", "input"),
    _float_field("data.normalization.mean", "input", minimum=0.0, maximum=1.0, increment=0.01, allow_none=True),
    _float_field("data.normalization.std", "input", minimum=1e-6, maximum=1.0, increment=0.01, allow_none=True),
    _enum_field("model.local_branch.mode", "input"),
    _float_field("augmentation.affine_p", "augmentation", minimum=0.0, maximum=1.0, increment=0.05),
    _float_field("augmentation.rotation_limit", "augmentation", minimum=0.0, maximum=90.0, increment=1.0),
    _float_field("augmentation.translate_limit", "augmentation", minimum=0.0, maximum=0.5, increment=0.01),
    _float_field("augmentation.scale_limit", "augmentation", minimum=0.0, maximum=0.5, increment=0.01),
    _float_field("augmentation.shear_limit", "augmentation", minimum=0.0, maximum=30.0, increment=1.0),
    _bool_field("augmentation.horizontal_flip", "augmentation"),
    _float_field("augmentation.horizontal_flip_p", "augmentation", minimum=0.0, maximum=1.0, increment=0.05),
    _bool_field("augmentation.use_noise", "augmentation"),
    _float_field("augmentation.noise_std_min", "augmentation", minimum=0.0, maximum=0.5, increment=0.001),
    _float_field("augmentation.noise_std_max", "augmentation", minimum=0.0, maximum=0.5, increment=0.001),
    _float_field("augmentation.noise_p", "augmentation", minimum=0.0, maximum=1.0, increment=0.05),
    _bool_field("augmentation.use_blur", "augmentation"),
    _int_field("augmentation.blur_limit", "augmentation", minimum=3, maximum=31, increment=1),
    _float_field("augmentation.blur_p", "augmentation", minimum=0.0, maximum=1.0, increment=0.05),
    _enum_field("training.best_metric", "output"),
    _int_field("training.early_stopping_patience", "output", minimum=0, maximum=100),
    _float_field("training.early_stopping_min_delta", "output", minimum=0.0, maximum=10.0, increment=0.005),
    _bool_field("runtime.deterministic", "advanced"),
    _bool_field("runtime.channels_last", "advanced"),
    _enum_field("training.compile_mode", "advanced", width=26),
    _int_field("training.prefetch_factor", "advanced", minimum=1, maximum=16),
    _bool_field("training.persistent_workers", "advanced"),
    _bool_field("training.pin_memory", "advanced"),
    _int_field("model.global_dim", "advanced", minimum=16, maximum=2048, increment=16),
    _int_field("model.metadata.hidden_dim", "advanced", minimum=8, maximum=1024, increment=8),
    _float_field("model.metadata.dropout", "advanced", minimum=0.0, maximum=0.9, increment=0.05),
    _int_field("model.local_branch.feature_dim", "advanced", minimum=16, maximum=1024, increment=16),
    _int_field("model.local_branch.geometry_dim", "advanced", minimum=8, maximum=512, increment=8),
    _float_field("model.local_branch.dropout", "advanced", minimum=0.0, maximum=0.9, increment=0.05),
    _int_field("model.head.hidden_dim", "advanced", minimum=8, maximum=2048, increment=16),
    _float_field("model.head.dropout", "advanced", minimum=0.0, maximum=0.9, increment=0.05),
    _float_field("training.momentum", "advanced", minimum=0.0, maximum=0.999, increment=0.01),
    _float_field("training.warmup_start_factor", "advanced", minimum=1e-4, maximum=1.0, increment=0.05),
    _float_field("training.min_lr", "advanced", minimum=0.0, maximum=1.0, increment=1e-6, width=14),
    _float_field("training.smooth_l1_beta", "advanced", minimum=1e-4, maximum=10.0, increment=0.05),
    _float_field("training.gradient_clip", "advanced", minimum=0.0, maximum=100.0, increment=0.1, allow_none=True),
    _float_field("training.scheduler_factor", "advanced", minimum=1e-4, maximum=0.9999, increment=0.05),
    _int_field("training.scheduler_patience", "advanced", minimum=0, maximum=50),
)
FIELD_SPEC_MAP: dict[str, UiFieldSpec] = {spec.path: spec for spec in VISIBLE_FIELD_SPECS}
VISIBLE_FIELD_PATHS: tuple[str, ...] = tuple(spec.path for spec in VISIBLE_FIELD_SPECS)


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
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


def _load_yaml_dict(path: Path, texts: UITextManager) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(texts.get_text("error.config_root_dict", path=path))
    return data


def _build_train_ui_config(config_path: Path, texts: UITextManager) -> tuple[dict[str, Any], dict[str, Any]]:
    merged_config: dict[str, Any] = {}
    if DEFAULT_SCHEMA_PATH.exists():
        merged_config = _deep_merge(merged_config, _load_yaml_dict(DEFAULT_SCHEMA_PATH, texts))
    selected_config = _load_yaml_dict(config_path, texts)
    merged_config = _deep_merge(merged_config, selected_config)
    return merged_config, selected_config


def _lookup_nested_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


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


def _to_display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return normalize_visible_text(str(value))


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
        value = yaml.safe_load(text)
    except Exception:
        if _SCIENTIFIC_NOTATION_PATTERN.fullmatch(text):
            try:
                return float(text)
            except ValueError:
                pass
        return text
    if isinstance(value, str) and _SCIENTIFIC_NOTATION_PATTERN.fullmatch(text):
        try:
            return float(text)
        except ValueError:
            pass
    return value


def _normalize_resume_checkpoint_path(raw_value: Any) -> str | None:
    if raw_value is None:
        return None

    text = normalize_visible_text(str(raw_value)).strip()
    if not text or text.lower() == "null":
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if not text:
        return None

    text = os.path.expandvars(os.path.expanduser(text))
    system_name = platform.system().lower()
    if system_name.startswith("win"):
        wsl_match = _WSL_MOUNT_PATTERN.fullmatch(text.replace("\\", "/"))
        if wsl_match is not None:
            drive = wsl_match.group("drive").upper()
            rest = re.sub(r"/+", "/", wsl_match.group("rest")).lstrip("/").replace("/", "\\")
            return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
        return text

    windows_match = _WINDOWS_DRIVE_PATTERN.fullmatch(text)
    if windows_match is None:
        return text
    drive = windows_match.group("drive").lower()
    rest = re.sub(r"[\\/]+", "/", windows_match.group("rest")).lstrip("/")
    return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"


def _format_elapsed_clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _validate_ui_value(texts: UITextManager, dotted_key: str, value: Any) -> None:
    allowed = STRICT_OPTIONS.get(dotted_key)
    if allowed is None:
        return
    normalized = str(value).strip().lower()
    if normalized in allowed:
        return
    allowed_text = ", ".join(sorted(allowed))
    raise ValueError(texts.get_text("error.value_not_allowed", key=dotted_key, allowed=allowed_text))


def _pick_available_font(candidates: tuple[str, ...], available_fonts: dict[str, str]) -> str | None:
    for candidate in candidates:
        matched = available_fonts.get(candidate.casefold())
        if matched is not None:
            return matched
    return None


def _set_tcl_system_encoding(root: tk.Misc) -> str:
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


def _apply_ttk_font_styles(root: tk.Misc, _font_info: dict[str, str]) -> None:
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
        text = normalize_visible_text(text)
        if not text:
            return 0
        if self.original_stream is not None:
            try:
                self.original_stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(self.original_stream, "encoding", "utf-8") or "utf-8"
                safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
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
        self.texts = UITextManager()
        self.root.title(self.t("window.title"))
        self.root.geometry("1040x920")
        self.tk_system_encoding = _set_tcl_system_encoding(self.root)
        self.font_info = _configure_tk_font_fallback(self.root)
        _apply_ttk_font_styles(self.root, self.font_info)

        self.config_path_var = tk.StringVar(value=config_path)
        self.resume_checkpoint_var = tk.StringVar(value="")
        self.resume_mode_var = tk.StringVar(value="")
        self.language_var = tk.StringVar(value=self.texts.get_language())
        self.status_var = tk.StringVar(value=self.t("status.ready"))
        self.elapsed_var = tk.StringVar(value=self.t("timer.idle"))
        self.advanced_visible_var = tk.BooleanVar(value=False)
        self._status_key = "status.ready"
        self._status_kwargs: dict[str, Any] = {}
        self._elapsed_key = "timer.idle"

        self.field_bindings: dict[str, FieldBinding] = {}
        self.section_frames: dict[str, ttk.LabelFrame] = {}
        self.advanced_header_label: ttk.Label | None = None
        self.advanced_toggle_button: ttk.Button | None = None
        self.advanced_body: ttk.Frame | None = None
        self.base_values: dict[str, Any] = {}
        self.base_resume_checkpoint: str | None = None
        self.loaded_config: dict[str, Any] = {}
        self.running = False
        self.stop_requested = False
        self.training_thread: threading.Thread | None = None
        self.training_control = None
        self.training_started_at: float | None = None
        self._elapsed_after_id: str | None = None
        self._last_elapsed_seconds: float | None = None
        self.log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._log_flush_scheduled = False
        self._max_log_chars = 200_000
        self._output_capture_enabled = True
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_proxy = _UiTextStream(self, self._original_stdout)
        self._stderr_proxy = _UiTextStream(self, self._original_stderr)

        self._build_layout()
        self.resume_checkpoint_var.trace_add("write", self._handle_resume_checkpoint_changed)
        self._update_resume_mode_text()
        self._install_output_redirects()
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._load_config_into_form(config_path, prompt_resume_selection=False)
        if self.tk_system_encoding:
            self.enqueue_output(self.t("log.tk_encoding", encoding=self.tk_system_encoding))
        if self.font_info["ui_family"]:
            self.enqueue_output(self.t("log.font_enabled", font=self.font_info["ui_family"]))
        else:
            self.enqueue_output(self.t("log.font_default"))

    def t(self, key: str, **kwargs: Any) -> str:
        return self.texts.get_text(key, **kwargs)

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        self.config_label = ttk.Label(top, text=self.t("top.config_file"))
        self.config_label.pack(side=tk.LEFT)
        self.config_entry = ttk.Entry(top, textvariable=self.config_path_var)
        self.config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

        self.select_config_button = ttk.Button(top, text=self.t("top.select_file"), command=self._choose_config)
        self.select_config_button.pack(side=tk.LEFT, padx=(0, 8))
        self.save_config_button = ttk.Button(top, text=self.t("top.save_config"), command=self._save_current_config)
        self.save_config_button.pack(side=tk.LEFT, padx=(0, 8))
        self.reload_config_button = ttk.Button(top, text=self.t("top.reload"), command=self._reload_current_config)
        self.reload_config_button.pack(side=tk.LEFT)

        self.language_combo = ttk.Combobox(top, state="readonly", width=10, textvariable=self.language_var)
        self.language_combo.bind("<<ComboboxSelected>>", self._handle_language_selected)
        self.language_combo.pack(side=tk.RIGHT)
        self.language_label = ttk.Label(top, text=self.t("top.language"))
        self.language_label.pack(side=tk.RIGHT, padx=(12, 6))

        self.resume_section = ttk.LabelFrame(self.root, text=self.t("resume.section_title"), padding=(10, 8))
        self.resume_section.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.resume_description_label = ttk.Label(
            self.resume_section,
            text=self.t("resume.description"),
            justify=tk.LEFT,
            wraplength=980,
        )
        self.resume_description_label.pack(fill=tk.X, pady=(0, 8))

        resume_row = ttk.Frame(self.resume_section)
        resume_row.pack(fill=tk.X)
        self.resume_path_label = ttk.Label(resume_row, text=self._resume_field_label())
        self.resume_path_label.pack(side=tk.LEFT)
        self.resume_entry = ttk.Entry(resume_row, textvariable=self.resume_checkpoint_var)
        self.resume_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.select_resume_button = ttk.Button(
            resume_row,
            text=self.t("top.select_resume"),
            command=self._choose_resume_checkpoint,
        )
        self.select_resume_button.pack(side=tk.LEFT, padx=(0, 8))
        self.clear_resume_button = ttk.Button(
            resume_row,
            text=self.t("top.clear_resume"),
            command=self._clear_resume_checkpoint,
        )
        self.clear_resume_button.pack(side=tk.LEFT)
        self.resume_mode_label = ttk.Label(
            self.resume_section,
            textvariable=self.resume_mode_var,
            justify=tk.LEFT,
            wraplength=980,
        )
        self.resume_mode_label.pack(fill=tk.X, pady=(8, 0))

        body = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        body.pack(fill=tk.BOTH, expand=True)

        pane = ttk.Panedwindow(body, orient=tk.VERTICAL)
        pane.pack(fill=tk.BOTH, expand=True)

        form_container = ttk.Frame(pane)
        self.log_container = ttk.LabelFrame(pane, text=self.t("panel.training_output"), padding=(8, 8))
        pane.add(form_container, weight=4)
        pane.add(self.log_container, weight=2)

        self.canvas = tk.Canvas(form_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(form_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.form_frame = ttk.Frame(self.canvas)
        self.form_frame.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.form_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        log_toolbar = ttk.Frame(self.log_container)
        log_toolbar.pack(fill=tk.X, pady=(0, 6))
        self.log_streams_label = ttk.Label(log_toolbar, text=self.t("panel.output_streams"))
        self.log_streams_label.pack(side=tk.LEFT)
        self.clear_output_button = ttk.Button(log_toolbar, text=self.t("button.clear_output"), command=self._clear_output)
        self.clear_output_button.pack(side=tk.RIGHT)

        log_font = tkfont.nametofont("TkFixedFont").copy()
        log_font.configure(size=10)
        self.output_text = scrolledtext.ScrolledText(
            self.log_container,
            wrap=tk.WORD,
            height=12,
            state=tk.DISABLED,
            font=log_font,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(self.root, padding=10)
        bottom.pack(fill=tk.X)

        self.status_label = ttk.Label(bottom, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=180)
        self.progress.pack(side=tk.RIGHT, padx=(8, 8))
        self.elapsed_label = ttk.Label(bottom, textvariable=self.elapsed_var, width=22, anchor="e")
        self.elapsed_label.pack(side=tk.RIGHT)
        self.run_button = ttk.Button(bottom, text=self.t("button.start_training"), command=self._start_training)
        self.run_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.reset_defaults_button = ttk.Button(bottom, text=self.t("button.reset_defaults"), command=self._reset_to_defaults)
        self.reset_defaults_button.pack(side=tk.RIGHT)

        self._refresh_language_selector()

    def _install_output_redirects(self) -> None:
        sys.stdout = self._stdout_proxy
        sys.stderr = self._stderr_proxy

    def _restore_output_redirects(self) -> None:
        if sys.stdout is self._stdout_proxy:
            sys.stdout = self._original_stdout
        if sys.stderr is self._stderr_proxy:
            sys.stderr = self._original_stderr

    def _handle_close(self) -> None:
        if self.running and self.training_control is not None:
            if not self.stop_requested:
                self.enqueue_output(self.t("log.window_close_stop_requested"))
                self._request_stop_training()
            return
        self._cancel_elapsed_updates()
        self._output_capture_enabled = False
        self._restore_output_redirects()
        self.root.destroy()

    def _refresh_language_selector(self) -> None:
        labels = [self.texts.get_language_label(code) for code in self.texts.get_languages()]
        self.language_combo.configure(values=labels)
        current_index = self.texts.get_languages().index(self.texts.get_language())
        self.language_combo.current(current_index)

    def _render_status(self) -> None:
        self.status_var.set(self.t(self._status_key, **self._status_kwargs))

    def _render_elapsed(self) -> None:
        kwargs: dict[str, Any] = {}
        if self._elapsed_key != "timer.idle" and self._last_elapsed_seconds is not None:
            kwargs["elapsed"] = _format_elapsed_clock(self._last_elapsed_seconds)
        self.elapsed_var.set(self.t(self._elapsed_key, **kwargs))

    def _set_status(self, key: str, **kwargs: Any) -> None:
        self._status_key = key
        self._status_kwargs = kwargs
        self._render_status()

    def _set_elapsed_state(self, key: str, elapsed_seconds: float | None = None) -> None:
        self._elapsed_key = key
        if key == "timer.idle":
            self._last_elapsed_seconds = None
        elif elapsed_seconds is not None:
            self._last_elapsed_seconds = max(0.0, float(elapsed_seconds))
        self._render_elapsed()

    def _show_error(self, title_key: str, message_key: str, **kwargs: Any) -> None:
        messagebox.showerror(self.t(title_key), self.t(message_key, **kwargs))

    def _show_error_text(self, title_key: str, message: str) -> None:
        messagebox.showerror(self.t(title_key), normalize_visible_text(message))

    def _show_info(self, title_key: str, message_key: str, **kwargs: Any) -> None:
        messagebox.showinfo(self.t(title_key), self.t(message_key, **kwargs))

    def _resume_field_label(self) -> str:
        name, _description = self.texts.get_option_meta("training.resume_checkpoint")
        return name

    def _get_resume_checkpoint_value(self) -> str | None:
        return _normalize_resume_checkpoint_path(self.resume_checkpoint_var.get())

    def _update_resume_mode_text(self) -> None:
        resume_checkpoint = self._get_resume_checkpoint_value()
        if resume_checkpoint:
            self.resume_mode_var.set(self.t("resume.mode_resume", path=resume_checkpoint))
            return
        self.resume_mode_var.set(self.t("resume.mode_fresh"))

    def _set_resume_checkpoint(self, value: Any, *, status_key: str | None = None) -> None:
        normalized = _normalize_resume_checkpoint_path(value)
        display_value = normalized or ""
        if self.resume_checkpoint_var.get() != display_value:
            self.resume_checkpoint_var.set(display_value)
        else:
            self._update_resume_mode_text()
        if status_key is not None:
            status_kwargs = {"path": normalized} if normalized else {}
            self._set_status(status_key, **status_kwargs)

    def _handle_resume_checkpoint_changed(self, *_args: Any) -> None:
        self._update_resume_mode_text()

    def _refresh_texts(self) -> None:
        self.root.title(self.t("window.title"))
        self.config_label.configure(text=self.t("top.config_file"))
        self.select_config_button.configure(text=self.t("top.select_file"))
        self.save_config_button.configure(text=self.t("top.save_config"))
        self.reload_config_button.configure(text=self.t("top.reload"))
        self.language_label.configure(text=self.t("top.language"))
        self.resume_section.configure(text=self.t("resume.section_title"))
        self.resume_description_label.configure(text=self.t("resume.description"))
        self.resume_path_label.configure(text=self._resume_field_label())
        self.select_resume_button.configure(text=self.t("top.select_resume"))
        self.clear_resume_button.configure(text=self.t("top.clear_resume"))
        self.log_container.configure(text=self.t("panel.training_output"))
        self.log_streams_label.configure(text=self.t("panel.output_streams"))
        self.clear_output_button.configure(text=self.t("button.clear_output"))
        self.reset_defaults_button.configure(text=self.t("button.reset_defaults"))
        self._configure_run_button(running=self.running, stopping=self.stop_requested)
        self._refresh_language_selector()
        self._update_resume_mode_text()
        self._render_status()
        self._render_elapsed()
        self._refresh_form_texts()

    def _field_description(self, dotted_key: str) -> str:
        _name, description = self.texts.get_option_meta(dotted_key)
        return description

    def _refresh_form_texts(self) -> None:
        for group, frame in self.section_frames.items():
            frame.configure(text=self.t(SECTION_TITLE_KEYS[group]))
        if self.advanced_header_label is not None:
            self.advanced_header_label.configure(text=self.t(SECTION_TITLE_KEYS["advanced"]))
        if self.advanced_toggle_button is not None:
            self.advanced_toggle_button.configure(
                text=self.t("button.hide_advanced") if self.advanced_visible_var.get() else self.t("button.show_advanced")
            )
        for dotted_key, binding in self.field_bindings.items():
            name, _description = self.texts.get_option_meta(dotted_key)
            binding.name_label.configure(text=name)
            binding.desc_label.configure(text=self._field_description(dotted_key))

    def _handle_language_selected(self, _event: tk.Event | None = None) -> None:
        index = self.language_combo.current()
        if index < 0:
            return
        selected_language = self.texts.get_languages()[index]
        if selected_language == self.texts.get_language():
            return
        self.texts.set_language(selected_language)
        self._refresh_texts()
        self.enqueue_output(
            self.t(
                "log.language_switched",
                language=self.texts.get_language_label(selected_language),
            )
        )

    def enqueue_output(self, text: str) -> None:
        normalized = normalize_visible_text(text).replace("\r\n", "\n").replace("\r", "\n")
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

    def _cancel_elapsed_updates(self) -> None:
        if self._elapsed_after_id is not None:
            self.root.after_cancel(self._elapsed_after_id)
            self._elapsed_after_id = None

    def _refresh_training_elapsed(self) -> None:
        if not self.running or self.training_started_at is None:
            self._elapsed_after_id = None
            return
        self._last_elapsed_seconds = max(0.0, time.perf_counter() - self.training_started_at)
        self._elapsed_key = "timer.running"
        self._render_elapsed()
        self._elapsed_after_id = self.root.after(500, self._refresh_training_elapsed)

    def _begin_training_elapsed(self, started_at: float) -> None:
        self.training_started_at = float(started_at)
        self._last_elapsed_seconds = 0.0
        self._cancel_elapsed_updates()
        self._set_elapsed_state("timer.running", 0.0)
        self._elapsed_after_id = self.root.after(500, self._refresh_training_elapsed)

    def _finish_training_elapsed(self, state_key: str, elapsed_seconds: float | None = None) -> str:
        if elapsed_seconds is None:
            if self.training_started_at is not None:
                elapsed_seconds = time.perf_counter() - self.training_started_at
            else:
                elapsed_seconds = self._last_elapsed_seconds or 0.0
        elapsed_seconds = max(0.0, float(elapsed_seconds))
        self._cancel_elapsed_updates()
        self.training_started_at = None
        self._set_elapsed_state(state_key, elapsed_seconds)
        return _format_elapsed_clock(elapsed_seconds)

    def _log_control_message(self, message: str, level: str = "info") -> None:
        logger = logging.getLogger("rhpe_boneage")
        if not logger.handlers:
            return
        log_method = getattr(logger, level, logger.info)
        log_method(normalize_visible_text(message), extra={"phase": "SYSTEM"})

    def _configure_run_button(self, *, running: bool, stopping: bool = False) -> None:
        if running and not stopping:
            self.run_button.configure(text=self.t("button.stop_training"), command=self._request_stop_training, state=tk.NORMAL)
            return
        if running and stopping:
            self.run_button.configure(text=self.t("button.stopping"), command=self._request_stop_training, state=tk.DISABLED)
            return
        self.run_button.configure(text=self.t("button.start_training"), command=self._start_training, state=tk.NORMAL)

    def _choose_config(self) -> None:
        selected = filedialog.askopenfilename(
            title=self.t("file_dialog.select_config"),
            filetypes=[
                (self.t("filetype.yaml"), "*.yaml *.yml"),
                (self.t("filetype.all"), "*.*"),
            ],
        )
        if not selected:
            return
        self.config_path_var.set(selected)
        self._load_config_into_form(selected, prompt_resume_selection=True)

    def _choose_resume_checkpoint(self) -> None:
        self._prompt_resume_checkpoint_selection(self.config_path_var.get().strip())

    def _clear_resume_checkpoint(self) -> None:
        self._set_resume_checkpoint(None, status_key="status.resume_cleared")

    def _reload_current_config(self) -> None:
        self._load_config_into_form(self.config_path_var.get().strip(), prompt_resume_selection=False)

    def _clear_form(self) -> None:
        for child in self.form_frame.winfo_children():
            child.destroy()
        self.field_bindings.clear()
        self.section_frames.clear()
        self.advanced_header_label = None
        self.advanced_toggle_button = None
        self.advanced_body = None

    def _display_field_value(self, spec: UiFieldSpec, value: Any) -> Any:
        if spec.kind == "bool":
            return bool(value)
        if value is None:
            return ""
        if spec.kind == "text":
            return normalize_visible_text(str(value))
        return _to_display_value(value)

    def _create_widget(self, parent: ttk.Frame, spec: UiFieldSpec, value: Any) -> tuple[tk.Variable, tk.Widget]:
        if spec.kind == "bool":
            variable = tk.BooleanVar(value=bool(value))
            widget = ttk.Checkbutton(parent, variable=variable)
            widget.pack(side=tk.RIGHT)
            return variable, widget

        initial_text = self._display_field_value(spec, value)
        variable = tk.StringVar(value=str(initial_text))
        if spec.kind == "enum":
            widget = ttk.Combobox(parent, textvariable=variable, values=list(spec.options), state="readonly", width=spec.width)
        elif spec.kind in {"int", "float"}:
            widget = ttk.Spinbox(
                parent,
                textvariable=variable,
                from_=spec.minimum if spec.minimum is not None else -1_000_000_000,
                to=spec.maximum if spec.maximum is not None else 1_000_000_000,
                increment=spec.increment if spec.increment is not None else 1.0,
                width=spec.width,
            )
        else:
            widget = ttk.Entry(parent, textvariable=variable, width=spec.width)
        widget.pack(side=tk.RIGHT)
        return variable, widget

    def _build_section_frames(self) -> dict[str, ttk.Frame]:
        containers: dict[str, ttk.Frame] = {}
        for group in SECTION_ORDER:
            if group == "advanced":
                outer = ttk.Frame(self.form_frame)
                outer.pack(fill=tk.X, pady=(0, 12))
                header = ttk.Frame(outer)
                header.pack(fill=tk.X)
                self.advanced_header_label = ttk.Label(header, text=self.t(SECTION_TITLE_KEYS[group]))
                self.advanced_header_label.pack(side=tk.LEFT)
                self.advanced_toggle_button = ttk.Button(header, command=self._toggle_advanced)
                self.advanced_toggle_button.pack(side=tk.RIGHT)
                self.advanced_body = ttk.Frame(outer, padding=(10, 8, 10, 0))
                if self.advanced_visible_var.get():
                    self.advanced_body.pack(fill=tk.X)
                containers[group] = self.advanced_body
                continue

            frame = ttk.LabelFrame(self.form_frame, text=self.t(SECTION_TITLE_KEYS[group]), padding=(10, 8))
            frame.pack(fill=tk.X, pady=(0, 12))
            self.section_frames[group] = frame
            containers[group] = frame
        self._refresh_form_texts()
        return containers

    def _build_field_row(self, container: ttk.Frame, spec: UiFieldSpec, value: Any) -> None:
        row = ttk.Frame(container)
        row.pack(fill=tk.X, pady=(0, 10))

        head = ttk.Frame(row)
        head.pack(fill=tk.X)
        name, _description = self.texts.get_option_meta(spec.path)
        name_label = ttk.Label(head, text=name)
        name_label.pack(side=tk.LEFT, anchor="w")
        variable, widget = self._create_widget(head, spec, value)
        desc_label = ttk.Label(row, text=self._field_description(spec.path), justify=tk.LEFT, wraplength=760)
        desc_label.pack(fill=tk.X, pady=(2, 0))

        binding = FieldBinding(
            spec=spec,
            variable=variable,
            widget=widget,
            name_label=name_label,
            desc_label=desc_label,
        )
        self.field_bindings[spec.path] = binding
        variable.trace_add("write", self._handle_form_changed)

    def _handle_form_changed(self, *_args: Any) -> None:
        self._update_field_states()

    def _toggle_advanced(self) -> None:
        self.advanced_visible_var.set(not self.advanced_visible_var.get())
        if self.advanced_body is not None:
            if self.advanced_visible_var.get():
                self.advanced_body.pack(fill=tk.X)
            else:
                self.advanced_body.pack_forget()
        if self.advanced_toggle_button is not None:
            self.advanced_toggle_button.configure(
                text=self.t("button.hide_advanced") if self.advanced_visible_var.get() else self.t("button.show_advanced")
            )

    def _coerce_field_value(self, spec: UiFieldSpec, raw_value: Any, base_value: Any, *, strict: bool) -> Any:
        if spec.kind == "bool":
            return bool(raw_value)

        text = str(raw_value).strip()
        if text == "":
            if base_value is not None or spec.kind == "text":
                return base_value
            if spec.allow_none:
                return None
            if strict:
                raise ValueError(self.t("error.null_not_allowed", key=spec.path))
            return base_value

        if spec.kind == "text":
            return normalize_visible_text(text)

        if spec.kind == "enum":
            _validate_ui_value(self.texts, spec.path, text)
            return text

        parsed = _parse_value(text)
        if parsed is None:
            if spec.allow_none:
                return None
            if strict:
                raise ValueError(self.t("error.null_not_allowed", key=spec.path))
            return base_value

        if isinstance(parsed, bool):
            if strict:
                raise ValueError(f"{spec.path} 需要数值类型。")
            return base_value

        if spec.kind == "int":
            if not isinstance(parsed, (int, float)) or not float(parsed).is_integer():
                if strict:
                    raise ValueError(f"{spec.path} 需要整数。")
                return base_value
            value = int(parsed)
        else:
            if not isinstance(parsed, (int, float)):
                if strict:
                    raise ValueError(f"{spec.path} 需要数值。")
                return base_value
            value = float(parsed)

        if spec.minimum is not None and value < spec.minimum:
            if strict:
                raise ValueError(f"{spec.path} 不能小于 {spec.minimum}.")
            value = base_value
        if spec.maximum is not None and value > spec.maximum:
            if strict:
                raise ValueError(f"{spec.path} 不能大于 {spec.maximum}.")
            value = base_value
        return value

    def _snapshot_form_values(self, *, strict: bool) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for spec in VISIBLE_FIELD_SPECS:
            binding = self.field_bindings.get(spec.path)
            if binding is None:
                continue
            raw_value = binding.variable.get()
            base_value = self.base_values.get(spec.path)
            try:
                values[spec.path] = self._coerce_field_value(spec, raw_value, base_value, strict=strict)
            except ValueError:
                if strict:
                    raise
                values[spec.path] = base_value
        return values

    def _validate_cross_field_values(self, values: dict[str, Any]) -> None:
        if values.get("data.normalization.source") == "manual":
            mean = values.get("data.normalization.mean")
            std = values.get("data.normalization.std")
            if mean is None or std is None:
                raise ValueError("data.normalization.source=manual 时必须填写 mean 和 std。")
            if float(std) <= 0:
                raise ValueError("data.normalization.std 必须大于 0。")
        if bool(values.get("augmentation.use_noise")):
            noise_min = float(values.get("augmentation.noise_std_min") or 0.0)
            noise_max = float(values.get("augmentation.noise_std_max") or 0.0)
            if noise_max < noise_min:
                raise ValueError("augmentation.noise_std_max 不能小于 augmentation.noise_std_min。")
        if str(values.get("training.scheduler") or "none").lower() == "plateau":
            factor = float(values.get("training.scheduler_factor") or 0.0)
            if not (0.0 < factor < 1.0):
                raise ValueError("training.scheduler_factor 必须在 (0, 1) 范围内。")

    def _field_enabled(self, path: str, values: dict[str, Any]) -> bool:
        branch_mode = str(values.get("model.branch_mode") or "global_local").lower()
        ensemble_mode = str(values.get("model.ensemble_mode") or "ensemble").lower()
        metadata_enabled = bool(values.get("model.metadata.enabled"))
        metadata_mode = str(values.get("model.metadata.mode") or "mlp").lower()
        global_branch_enabled = branch_mode in {"global_only", "global_local"}
        local_branch_enabled = branch_mode in {"local_only", "global_local"}
        local_mode = str(values.get("model.local_branch.mode") or "patch_heatmap").lower()
        heatmap_guidance = bool(values.get("model.heatmap_guidance.enabled"))
        cbam_enabled = bool(values.get("model.cbam.enabled"))
        normalization_manual = str(values.get("data.normalization.source") or "auto_train_stats").lower() == "manual"
        scheduler_name = str(values.get("training.scheduler") or "none").lower()
        optimizer_name = str(values.get("training.optimizer") or "adamw").lower()
        loss_name = str(values.get("training.loss") or "smoothl1").lower()
        warmup_epochs = int(values.get("training.warmup_epochs") or 0)
        early_stop_patience = int(values.get("training.early_stopping_patience") or 0)
        heatmap_used = heatmap_guidance or (local_branch_enabled and local_mode in {"heatmap", "patch_heatmap"})

        if path == "model.pretrained":
            return global_branch_enabled
        if path == "model.resnet_name":
            return global_branch_enabled and ensemble_mode in {"ensemble", "resnet"}
        if path == "model.efficientnet_name":
            return global_branch_enabled and ensemble_mode in {"ensemble", "efficientnet"}
        if path == "model.relative_target_direction":
            return str(values.get("model.target_mode") or "relative").lower() == "relative"
        if path == "model.metadata.mode":
            return metadata_enabled
        if path in {"model.metadata.hidden_dim", "model.metadata.dropout"}:
            return metadata_enabled and metadata_mode in {"mlp", "simba_hybrid"}
        if path == "model.heatmap_guidance.enabled":
            return global_branch_enabled
        if path == "model.cbam.global_branch":
            return cbam_enabled and global_branch_enabled
        if path == "model.cbam.local_branch":
            return cbam_enabled and local_branch_enabled
        if path in {"data.local_patch_size", "model.local_branch.mode", "model.local_branch.feature_dim", "model.local_branch.geometry_dim", "model.local_branch.dropout"}:
            return local_branch_enabled
        if path == "model.global_dim":
            return global_branch_enabled
        if path == "data.global_crop_margin_ratio":
            return str(values.get("data.global_crop_mode") or "bbox").lower() == "bbox"
        if path in {"data.heatmap_sigma_ratio", "data.heatmap_sigma_min"}:
            return heatmap_used
        if path in {"data.normalization.mean", "data.normalization.std"}:
            return normalization_manual
        if path == "augmentation.horizontal_flip_p":
            return bool(values.get("augmentation.horizontal_flip"))
        if path in {"augmentation.noise_std_min", "augmentation.noise_std_max", "augmentation.noise_p"}:
            return bool(values.get("augmentation.use_noise"))
        if path in {"augmentation.blur_limit", "augmentation.blur_p"}:
            return bool(values.get("augmentation.use_blur"))
        if path == "training.momentum":
            return optimizer_name == "sgd"
        if path == "training.compile_mode":
            return bool(values.get("training.compile"))
        if path in {"training.scheduler_factor", "training.scheduler_patience"}:
            return scheduler_name == "plateau"
        if path == "training.min_lr":
            return scheduler_name in {"plateau", "cosine"}
        if path == "training.smooth_l1_beta":
            return loss_name == "smoothl1"
        if path == "training.warmup_start_factor":
            return warmup_epochs > 0
        if path == "training.early_stopping_min_delta":
            return early_stop_patience > 0
        return True

    def _apply_widget_state(self, binding: FieldBinding, enabled: bool) -> None:
        if binding.spec.kind == "bool":
            if enabled:
                binding.widget.state(["!disabled"])
            else:
                binding.widget.state(["disabled"])
            return
        if binding.spec.kind == "enum":
            binding.widget.configure(state="readonly" if enabled else "disabled")
            return
        binding.widget.configure(state="normal" if enabled else "disabled")

    def _update_field_states(self) -> None:
        values = self._snapshot_form_values(strict=False)
        for path, binding in self.field_bindings.items():
            self._apply_widget_state(binding, self._field_enabled(path, values))

    def _collect_current_config(self) -> dict[str, Any]:
        current_values = self._snapshot_form_values(strict=True)
        self._validate_cross_field_values(current_values)
        current_config = copy.deepcopy(self.loaded_config)
        for dotted_key, value in current_values.items():
            _assign_nested_value(current_config, dotted_key, value)
        _assign_nested_value(current_config, "training.resume_checkpoint", self._get_resume_checkpoint_value())
        return current_config

    def _save_current_config(self) -> None:
        config_path = self.config_path_var.get().strip()
        if not config_path:
            self._show_error("dialog.save_failed_title", "dialog.no_save_path")
            return
        current_path = Path(config_path)
        default_name = current_path.name or "config.yaml"
        file_name = simpledialog.askstring(
            self.t("dialog.rename_config_title"),
            self.t("dialog.rename_config_prompt"),
            parent=self.root,
            initialvalue=default_name,
        )
        if file_name is None:
            self._set_status("status.save_cancelled")
            return
        normalized_name = file_name.strip()
        if not normalized_name:
            self._show_error("dialog.save_failed_title", "dialog.empty_file_name")
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
            self._show_error_text("dialog.config_error_title", str(exc))
            return
        except Exception as exc:
            self._show_error("dialog.save_failed_title", "dialog.save_failed_detail", error=exc)
            return
        self.config_path_var.set(str(path))
        self._load_config_into_form(str(path))
        self._set_status("status.config_saved", path=path)

    def _prompt_resume_checkpoint_selection(self, config_path: str) -> None:
        current_resume = self._get_resume_checkpoint_value()
        initialdir = ""
        initialfile = ""
        if current_resume:
            current_resume_path = Path(current_resume)
            initialfile = current_resume_path.name
            if current_resume_path.parent.exists():
                initialdir = str(current_resume_path.parent)
        if not initialdir:
            config_parent = Path(config_path).resolve().parent
            if config_parent.exists():
                initialdir = str(config_parent)

        selected = filedialog.askopenfilename(
            title=self.t("file_dialog.select_resume"),
            initialdir=initialdir or None,
            initialfile=initialfile or None,
            filetypes=[
                (self.t("filetype.checkpoint"), "*.pt *.pth *.ckpt"),
                (self.t("filetype.all"), "*.*"),
            ],
        )
        if not selected:
            return
        self._set_resume_checkpoint(selected, status_key="status.resume_selected")

    def _load_config_into_form(self, config_path: str, *, prompt_resume_selection: bool = False) -> None:
        path = Path(config_path)
        if not path.exists():
            self._show_error("dialog.config_not_found_title", "dialog.config_not_found_detail", path=path)
            return
        try:
            merged_config, selected_config = _build_train_ui_config(path, self.texts)
        except Exception as exc:
            self._show_error("dialog.read_failed_title", "dialog.read_failed_detail", error=exc)
            return

        self._clear_form()
        self.loaded_config = copy.deepcopy(merged_config)
        merged_flat = _flatten_config(merged_config)
        selected_flat = _flatten_config(selected_config)
        hidden_count = sum(1 for dotted_key in merged_flat if dotted_key not in FIELD_SPEC_MAP)
        added_from_default = 0
        self.base_values = {}
        self.base_resume_checkpoint = _normalize_resume_checkpoint_path(
            _lookup_nested_value(merged_config, "training.resume_checkpoint", None)
        )
        self._set_resume_checkpoint(self.base_resume_checkpoint)

        section_containers = self._build_section_frames()
        for spec in VISIBLE_FIELD_SPECS:
            value = _lookup_nested_value(merged_config, spec.path, None)
            self.base_values[spec.path] = value
            if spec.path not in selected_flat:
                added_from_default += 1
            container = section_containers[spec.group]
            self._build_field_row(container, spec, value)

        self._update_field_states()
        self._set_status(
            "status.config_loaded_summary",
            visible_count=len(self.base_values),
            default_count=added_from_default,
            hidden_count=hidden_count,
        )
        if prompt_resume_selection:
            self.root.after_idle(lambda path_str=str(path): self._prompt_resume_checkpoint_selection(path_str))

    def _reset_to_defaults(self) -> None:
        for spec in VISIBLE_FIELD_SPECS:
            binding = self.field_bindings.get(spec.path)
            if binding is None:
                continue
            base_value = self.base_values.get(spec.path)
            if spec.kind == "bool":
                binding.variable.set(bool(base_value))
            else:
                binding.variable.set(str(self._display_field_value(spec, base_value)))
        self._set_resume_checkpoint(self.base_resume_checkpoint)
        self._update_field_states()
        self._set_status("status.defaults_restored")

    def _collect_overrides(self) -> list[str]:
        current_values = self._snapshot_form_values(strict=True)
        self._validate_cross_field_values(current_values)
        overrides: list[str] = []
        for spec in VISIBLE_FIELD_SPECS:
            base_value = self.base_values.get(spec.path)
            current_value = current_values.get(spec.path)
            if current_value != base_value:
                overrides.append(f"{spec.path}={_scalar_to_override(current_value)}")
        current_resume_checkpoint = self._get_resume_checkpoint_value()
        if current_resume_checkpoint != self.base_resume_checkpoint:
            overrides.append(f"training.resume_checkpoint={_scalar_to_override(current_resume_checkpoint)}")
        return overrides

    def _set_running(self, running: bool, status_key: str, **status_kwargs: Any) -> None:
        self.running = running
        self._set_status(status_key, **status_kwargs)
        if running:
            self._configure_run_button(running=True, stopping=self.stop_requested)
            self.progress.start(10)
        else:
            self.progress.stop()
            self._configure_run_button(running=False)

    def _reset_training_runtime(self) -> None:
        self.running = False
        self.stop_requested = False
        self.training_thread = None
        self.training_control = None
        self.training_started_at = None
        self._cancel_elapsed_updates()

    def _handle_training_stopped(self, stop_text: str) -> None:
        elapsed_text = self._finish_training_elapsed("timer.stopped")
        self._reset_training_runtime()
        self._set_running(False, "status.training_stopped", elapsed=elapsed_text)
        self.enqueue_output(self.t("log.training_stopped", reason=stop_text, elapsed=elapsed_text))

    def _handle_training_success(self, run_dir: str, elapsed_seconds: float | None = None) -> None:
        elapsed_text = self._finish_training_elapsed("timer.finished", elapsed_seconds)
        self._reset_training_runtime()
        self._set_running(False, "status.training_finished", run_dir=run_dir, elapsed=elapsed_text)
        self.enqueue_output(self.t("log.training_finished", run_dir=run_dir, elapsed=elapsed_text))
        self._show_info("dialog.training_complete_title", "dialog.training_complete_detail", run_dir=run_dir, elapsed=elapsed_text)

    def _handle_training_error(self, error_text: str) -> None:
        elapsed_text = self._finish_training_elapsed("timer.failed")
        self._reset_training_runtime()
        self._set_running(False, "status.training_failed", elapsed=elapsed_text)
        self.enqueue_output(self.t("log.training_failed", error=error_text, elapsed=elapsed_text))
        self._show_error_text("dialog.training_failed_title", f"{error_text}\n{self.t('timer.failed', elapsed=elapsed_text)}")

    def _request_stop_training(self) -> None:
        if not self.running or self.training_control is None or self.stop_requested:
            return
        self.stop_requested = True
        self.training_control.request_stop()
        phase, scope, _ = self.training_control.snapshot()
        self._set_status("status.stop_requested")
        self._configure_run_button(running=True, stopping=True)
        stop_scope = scope or "n/a"
        elapsed_text = _format_elapsed_clock(
            (time.perf_counter() - self.training_started_at) if self.training_started_at is not None else self._last_elapsed_seconds
        )
        message = self.t("log.user_stop_requested", phase=phase, scope=stop_scope, elapsed=elapsed_text)
        self.enqueue_output(message)
        self._log_control_message(message, level="warning")

    def _start_training(self) -> None:
        if self.running or (self.training_thread is not None and self.training_thread.is_alive()):
            return
        config_path = self.config_path_var.get().strip()
        if not Path(config_path).exists():
            self._show_error("dialog.config_error_title", "dialog.config_path_missing_detail", path=config_path)
            return
        try:
            overrides = self._collect_overrides()
        except ValueError as exc:
            self._show_error_text("dialog.config_error_title", str(exc))
            return
        resume_checkpoint = self._get_resume_checkpoint_value()
        if resume_checkpoint:
            resume_path = Path(resume_checkpoint)
            if not resume_path.exists():
                self._show_error(
                    "dialog.config_error_title",
                    "dialog.resume_checkpoint_not_found_detail",
                    path=resume_checkpoint,
                )
                return
            if not resume_path.is_file():
                self._show_error(
                    "dialog.config_error_title",
                    "dialog.resume_checkpoint_not_file_detail",
                    path=resume_checkpoint,
                )
                return
        from rhpe_boneage.training.control import TrainingCancelledError, TrainingControl

        self.training_control = TrainingControl()
        started_at = time.perf_counter()
        self.training_control.set_run_started_at(started_at)
        self.stop_requested = False
        self._begin_training_elapsed(started_at)
        self._set_running(True, "status.training_starting")
        self.enqueue_output(
            self.t(
                "log.training_start_banner",
                separator="=" * 96,
                config_path=config_path,
                override_count=len(overrides),
            )
        )
        if resume_checkpoint:
            self.enqueue_output(self.t("log.training_mode_resume", path=resume_checkpoint))
        else:
            self.enqueue_output(self.t("log.training_mode_fresh"))
        if overrides:
            for item in overrides:
                self.enqueue_output(self.t("log.override", item=item))

        def _worker() -> None:
            try:
                from rhpe_boneage.training.runner import train_main

                result = train_main(
                    config_path=config_path,
                    overrides=overrides,
                    control=self.training_control,
                )
                run_dir = result.get("run_dir", "")
                elapsed_seconds = result.get("total_elapsed_seconds")
                self.root.after(0, self._handle_training_success, run_dir, elapsed_seconds)
            except TrainingCancelledError as exc:
                self.root.after(0, self._handle_training_stopped, str(exc))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                self.root.after(0, self._handle_training_error, str(exc))

        self.training_thread = threading.Thread(target=_worker, daemon=True)
        self.training_thread.start()


def main() -> None:
    bootstrap()
    cli_texts = UITextManager()

    parser = argparse.ArgumentParser(description=cli_texts.get_text("cli.description"))
    parser.add_argument("--config", default="configs/default.yaml", help=cli_texts.get_text("cli.help.config"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], help=cli_texts.get_text("cli.help.set"))
    parser.add_argument("--auto-run", action="store_true", help=cli_texts.get_text("cli.help.auto_run"))
    args = parser.parse_args()

    if args.auto_run:
        from rhpe_boneage.training.runner import train_main

        train_main(config_path=args.config, overrides=args.overrides)
        return

    root = tk.Tk()
    TrainUI(root=root, config_path=args.config)
    root.minsize(780, 620)
    root.mainloop()


if __name__ == "__main__":
    run_cli(main)
