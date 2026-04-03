from __future__ import annotations

import argparse
import json
import threading
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import yaml

try:
    from _bootstrap import bootstrap, run_cli
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap, run_cli


PRESET_OPTIONS: dict[str, list[str]] = {
    "runtime.device": ["cuda:0", "cuda:1", "cpu"],
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
}

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


class TrainUI:
    def __init__(self, root: tk.Tk, config_path: str) -> None:
        self.root = root
        self.root.title("RHPE BoneAge Training UI")
        self.root.geometry("1360x820")

        self.config_path_var = tk.StringVar(value=config_path)
        self.status_var = tk.StringVar(value="就绪")
        self.widgets: dict[str, ttk.Combobox] = {}
        self.base_flat: OrderedDict[str, Any] = OrderedDict()
        self.running = False

        self._build_layout()
        self._load_config_into_form(config_path)

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="配置文件").pack(side=tk.LEFT)
        config_entry = ttk.Entry(top, textvariable=self.config_path_var)
        config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

        ttk.Button(top, text="选择文件", command=self._choose_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="重新加载", command=self._reload_current_config).pack(side=tk.LEFT)

        body = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.canvas.yview)
        self.form_frame = ttk.Frame(self.canvas)

        self.form_frame.bind(
            "<Configure>",
            lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.form_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        bottom = ttk.Frame(self.root, padding=10)
        bottom.pack(fill=tk.X)

        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.run_button = ttk.Button(bottom, text="开始训练", command=self._start_training)
        self.run_button.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(bottom, text="恢复默认值", command=self._reset_to_defaults).pack(side=tk.RIGHT)

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

    def _clear_form(self) -> None:
        for child in self.form_frame.winfo_children():
            child.destroy()
        self.widgets.clear()

    def _load_config_into_form(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            messagebox.showerror("配置文件不存在", f"找不到配置文件: {path}")
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except Exception as exc:
            messagebox.showerror("读取失败", f"无法读取配置文件:\n{exc}")
            return
        if not isinstance(config, dict):
            messagebox.showerror("配置错误", "配置文件顶层必须是字典。")
            return

        self._clear_form()
        self.base_flat = _flatten_config(config)

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
        self.status_var.set(f"已加载 {len(self.base_flat)} 个参数，并显示中文名称与释义。")

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
            if parsed_value != base_value:
                overrides.append(f"{dotted_key}={_scalar_to_override(parsed_value)}")
        return overrides

    def _set_running(self, running: bool, message: str) -> None:
        self.running = running
        self.status_var.set(message)
        self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)

    def _handle_training_success(self, run_dir: str) -> None:
        self._set_running(False, f"训练完成。输出目录: {run_dir}")
        messagebox.showinfo("训练完成", f"训练已完成。\n输出目录:\n{run_dir}")

    def _handle_training_error(self, error_text: str) -> None:
        self._set_running(False, "训练失败。")
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

        def _worker() -> None:
            try:
                from rhpe_boneage.training.runner import train_main

                result = train_main(config_path=config_path, overrides=overrides)
                run_dir = result.get("run_dir", "")
                self.root.after(0, self._handle_training_success, run_dir)
            except Exception as exc:
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
    root.minsize(1120, 620)
    root.mainloop()


if __name__ == "__main__":
    run_cli(main)
