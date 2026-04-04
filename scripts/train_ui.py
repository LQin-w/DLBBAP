from __future__ import annotations

import argparse
import copy
import json
import logging
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
    from ui_text import UITextManager, normalize_visible_text
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap, run_cli
    from scripts.ui_text import UITextManager, normalize_visible_text


PRESET_OPTIONS: dict[str, list[str]] = {
    "experiment.mode": ["enhanced", "simba", "bonet_like"],
    "runtime.device": ["cuda:0", "cuda:1", "cpu"],
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
    "training.compile_mode": ["reduce-overhead", "default", "max-autotune"],
    "training.best_metric": ["mae", "mad", "loss"],
    "optuna.direction": ["minimize", "maximize"],
}

STRICT_OPTIONS: dict[str, set[str]] = {
    "experiment.mode": {"enhanced", "simba", "bonet_like"},
    "data.global_crop_mode": {"bbox", "full", "none", "image"},
    "data.normalization.source": {"auto_train_stats", "manual"},
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
    "training.compile_mode": {"reduce-overhead", "default", "max-autotune"},
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
FORM_HEADER_KEYS: tuple[str, ...] = (
    "form.header.path",
    "form.header.name",
    "form.header.value",
    "form.header.description",
)


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
        text = normalize_visible_text(text)
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
        self.texts = UITextManager()
        self.root.title(self.t("window.title"))
        self.root.geometry("980x900")
        self.tk_system_encoding = _set_tcl_system_encoding(self.root)
        self.font_info = _configure_tk_font_fallback(self.root)
        _apply_ttk_font_styles(self.root, self.font_info)

        self.config_path_var = tk.StringVar(value=config_path)
        self.language_var = tk.StringVar(value=self.texts.get_language())
        self.status_var = tk.StringVar(value=self.t("status.ready"))
        self._status_key = "status.ready"
        self._status_kwargs: dict[str, Any] = {}

        self.widgets: dict[str, ttk.Combobox] = {}
        self.header_labels: list[ttk.Label] = []
        self.field_rows: dict[str, dict[str, Any]] = {}
        self.base_flat: OrderedDict[str, Any] = OrderedDict()
        self.loaded_config: dict[str, Any] = {}
        self.running = False
        self.stop_requested = False
        self.training_thread: threading.Thread | None = None
        self.training_control = None
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
        self.reload_config_button.pack(side=tk.LEFT, padx=(0, 8))
        self.select_resume_button = ttk.Button(
            top,
            text=self.t("top.select_resume"),
            command=self._choose_resume_checkpoint,
        )
        self.select_resume_button.pack(side=tk.LEFT, padx=(0, 8))
        self.clear_resume_button = ttk.Button(top, text=self.t("top.clear_resume"), command=self._clear_resume_checkpoint)
        self.clear_resume_button.pack(side=tk.LEFT)

        self.language_combo = ttk.Combobox(top, state="readonly", width=10, textvariable=self.language_var)
        self.language_combo.bind("<<ComboboxSelected>>", self._handle_language_selected)
        self.language_combo.pack(side=tk.RIGHT)
        self.language_label = ttk.Label(top, text=self.t("top.language"))
        self.language_label.pack(side=tk.RIGHT, padx=(12, 6))

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

        self.form_frame.bind(
            "<Configure>",
            lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
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

    def _set_status(self, key: str, **kwargs: Any) -> None:
        self._status_key = key
        self._status_kwargs = kwargs
        self._render_status()

    def _show_error(self, title_key: str, message_key: str, **kwargs: Any) -> None:
        messagebox.showerror(self.t(title_key), self.t(message_key, **kwargs))

    def _show_error_text(self, title_key: str, message: str) -> None:
        messagebox.showerror(self.t(title_key), normalize_visible_text(message))

    def _show_info(self, title_key: str, message_key: str, **kwargs: Any) -> None:
        messagebox.showinfo(self.t(title_key), self.t(message_key, **kwargs))

    def _refresh_texts(self) -> None:
        self.root.title(self.t("window.title"))
        self.config_label.configure(text=self.t("top.config_file"))
        self.select_config_button.configure(text=self.t("top.select_file"))
        self.save_config_button.configure(text=self.t("top.save_config"))
        self.reload_config_button.configure(text=self.t("top.reload"))
        self.select_resume_button.configure(text=self.t("top.select_resume"))
        self.clear_resume_button.configure(text=self.t("top.clear_resume"))
        self.language_label.configure(text=self.t("top.language"))
        self.log_container.configure(text=self.t("panel.training_output"))
        self.log_streams_label.configure(text=self.t("panel.output_streams"))
        self.clear_output_button.configure(text=self.t("button.clear_output"))
        self.reset_defaults_button.configure(text=self.t("button.reset_defaults"))
        self._configure_run_button(running=self.running, stopping=self.stop_requested)
        self._refresh_language_selector()
        self._render_status()
        self._refresh_form_texts()

    def _refresh_form_texts(self) -> None:
        for label, key in zip(self.header_labels, FORM_HEADER_KEYS):
            label.configure(text=self.t(key))
        for dotted_key, row in self.field_rows.items():
            name, desc = self.texts.get_option_meta(dotted_key)
            row["name_label"].configure(text=name)
            row["desc_label"].configure(text=desc)

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
        self._load_config_into_form(selected)

    def _reload_current_config(self) -> None:
        self._load_config_into_form(self.config_path_var.get().strip())

    def _set_field_value(self, dotted_key: str, value: str) -> None:
        widget = self.widgets.get(dotted_key)
        if widget is None:
            raise ValueError(self.t("error.form_field_missing", key=dotted_key))
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
            title=self.t("file_dialog.select_resume"),
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[
                (self.t("filetype.checkpoint"), "*.pt *.pth *.ckpt"),
                (self.t("filetype.all"), "*.*"),
            ],
        )
        if not selected:
            return
        try:
            self._set_field_value("training.resume_checkpoint", selected)
        except ValueError as exc:
            self._show_error_text("dialog.config_error_title", str(exc))
            return
        self._set_status("status.resume_selected", path=selected)

    def _clear_resume_checkpoint(self) -> None:
        try:
            self._set_field_value("training.resume_checkpoint", "null")
        except ValueError as exc:
            self._show_error_text("dialog.config_error_title", str(exc))
            return
        self._set_status("status.resume_cleared")

    def _clear_form(self) -> None:
        for child in self.form_frame.winfo_children():
            child.destroy()
        self.widgets.clear()
        self.header_labels.clear()
        self.field_rows.clear()

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
                    raise ValueError(self.t("error.null_not_allowed", key=dotted_key))
                _validate_ui_value(self.texts, dotted_key, parsed_value)
            _assign_nested_value(current_config, dotted_key, parsed_value)
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

    def _load_config_into_form(self, config_path: str) -> None:
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

        for index, header_key in enumerate(FORM_HEADER_KEYS):
            header_label = ttk.Label(self.form_frame, text=self.t(header_key))
            header_label.grid(
                row=0,
                column=index,
                sticky="w",
                padx=(0, 8),
                pady=(2, 8),
            )
            self.header_labels.append(header_label)

        row = 1
        for dotted_key, value in self.base_flat.items():
            display_name, description = self.texts.get_option_meta(dotted_key)
            path_label = ttk.Label(self.form_frame, text=dotted_key)
            path_label.grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=4)
            name_label = ttk.Label(self.form_frame, text=display_name)
            name_label.grid(row=row, column=1, sticky="nw", padx=(0, 8), pady=4)
            options = _build_options(dotted_key, value)
            combo = ttk.Combobox(self.form_frame, values=options, state="normal")
            combo.set(_to_display_value(value))
            combo.grid(row=row, column=2, sticky="ew", padx=(0, 8), pady=4)
            self.widgets[dotted_key] = combo
            desc_label = ttk.Label(
                self.form_frame,
                text=description,
                justify=tk.LEFT,
                wraplength=520,
            )
            desc_label.grid(row=row, column=3, sticky="nw", padx=(0, 8), pady=4)
            self.field_rows[dotted_key] = {
                "path_label": path_label,
                "name_label": name_label,
                "desc_label": desc_label,
                "combo": combo,
            }
            row += 1

        self.form_frame.columnconfigure(0, weight=2)
        self.form_frame.columnconfigure(1, weight=2)
        self.form_frame.columnconfigure(2, weight=1)
        self.form_frame.columnconfigure(3, weight=4)
        self._set_status(
            "status.config_loaded_summary",
            visible_count=len(self.base_flat),
            default_count=added_from_default,
            hidden_count=hidden_count,
        )

    def _reset_to_defaults(self) -> None:
        for dotted_key, value in self.base_flat.items():
            widget = self.widgets.get(dotted_key)
            if widget is not None:
                widget.set(_to_display_value(value))
        self._set_status("status.defaults_restored")

    def _collect_overrides(self) -> list[str]:
        overrides: list[str] = []
        for dotted_key, base_value in self.base_flat.items():
            widget = self.widgets.get(dotted_key)
            if widget is None:
                continue
            current_raw = widget.get()
            if current_raw.strip() == "":
                continue
            parsed_value = _parse_value(current_raw)
            if parsed_value is None and base_value is not None:
                raise ValueError(self.t("error.null_not_allowed", key=dotted_key))
            _validate_ui_value(self.texts, dotted_key, parsed_value)
            if parsed_value != base_value:
                overrides.append(f"{dotted_key}={_scalar_to_override(parsed_value)}")
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

    def _handle_training_stopped(self, stop_text: str) -> None:
        self._reset_training_runtime()
        self._set_running(False, "status.training_stopped")
        self.enqueue_output(self.t("log.training_stopped", reason=stop_text))

    def _handle_training_success(self, run_dir: str) -> None:
        self._reset_training_runtime()
        self._set_running(False, "status.training_finished", run_dir=run_dir)
        self.enqueue_output(self.t("log.training_finished", run_dir=run_dir))
        self._show_info("dialog.training_complete_title", "dialog.training_complete_detail", run_dir=run_dir)

    def _handle_training_error(self, error_text: str) -> None:
        self._reset_training_runtime()
        self._set_running(False, "status.training_failed")
        self.enqueue_output(self.t("log.training_failed", error=error_text))
        self._show_error_text("dialog.training_failed_title", error_text)

    def _request_stop_training(self) -> None:
        if not self.running or self.training_control is None or self.stop_requested:
            return
        self.stop_requested = True
        self.training_control.request_stop()
        phase, scope, _ = self.training_control.snapshot()
        self._set_status("status.stop_requested")
        self._configure_run_button(running=True, stopping=True)
        stop_scope = scope or "n/a"
        message = self.t("log.user_stop_requested", phase=phase, scope=stop_scope)
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
        from rhpe_boneage.training.control import TrainingCancelledError, TrainingControl

        self.training_control = TrainingControl()
        self.stop_requested = False
        self._set_running(True, "status.training_starting")
        self.enqueue_output(
            self.t(
                "log.training_start_banner",
                separator="=" * 96,
                config_path=config_path,
                override_count=len(overrides),
            )
        )
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
                self.root.after(0, self._handle_training_success, run_dir)
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
    root.minsize(747, 620)
    root.mainloop()


if __name__ == "__main__":
    run_cli(main)
