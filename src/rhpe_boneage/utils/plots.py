from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _prepare_output_dir(output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"输出路径不是目录: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _get_pyplot(output_dir: str | Path):
    plot_dir = _prepare_output_dir(output_dir)
    mplconfig_dir = Path(tempfile.mkdtemp(prefix="rhpe_boneage_mplconfig_"))
    cache_dir = Path(tempfile.mkdtemp(prefix="rhpe_boneage_cache_"))
    os.environ["MPLCONFIGDIR"] = str(mplconfig_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _apply_axis_style(axis) -> None:
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axis.set_axisbelow(True)


def _save_figure(fig, output_path: str | Path, dpi: int = 320) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return float(value)


def _validate_history_df(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        raise ValueError("history_df 为空，无法生成训练曲线。")
    required_columns = {"epoch"}
    missing_columns = sorted(required_columns - set(history_df.columns))
    if missing_columns:
        raise ValueError(f"history_df 缺少必要列: {', '.join(missing_columns)}")
    return history_df.copy()


def _validate_prediction_df(prediction_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if prediction_df is None:
        raise ValueError(f"{split_name} 预测结果为空，无法生成论文图表。")
    if not isinstance(prediction_df, pd.DataFrame):
        raise TypeError(f"{split_name} 预测结果必须是 DataFrame，收到: {type(prediction_df)}")
    required_columns = {"gt_boneage", "pred_boneage", "abs_error"}
    missing_columns = sorted(required_columns - set(prediction_df.columns))
    if missing_columns:
        raise ValueError(f"{split_name} 预测结果缺少必要列: {', '.join(missing_columns)}")
    if len(prediction_df["gt_boneage"]) != len(prediction_df["pred_boneage"]):
        raise ValueError(f"{split_name} 真实值和预测值长度不一致，无法绘图。")

    valid_df = prediction_df[
        prediction_df["gt_boneage"].notna() & prediction_df["pred_boneage"].notna()
    ].copy()
    if valid_df.empty:
        raise ValueError(f"{split_name} 没有带真实值的预测记录，无法生成图表。")

    valid_df["gt_boneage"] = valid_df["gt_boneage"].astype(float)
    valid_df["pred_boneage"] = valid_df["pred_boneage"].astype(float)
    valid_df["abs_error"] = valid_df["abs_error"].astype(float)
    valid_df["residual"] = valid_df["pred_boneage"] - valid_df["gt_boneage"]
    return valid_df


def _validate_prediction_only_df(prediction_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if prediction_df is None:
        raise ValueError(f"{split_name} 预测结果为空。")
    if not isinstance(prediction_df, pd.DataFrame):
        raise TypeError(f"{split_name} 预测结果必须是 DataFrame，收到: {type(prediction_df)}")
    if "pred_boneage" not in prediction_df.columns:
        raise ValueError(f"{split_name} 预测结果缺少 pred_boneage 列。")

    valid_df = prediction_df[prediction_df["pred_boneage"].notna()].copy()
    if valid_df.empty:
        raise ValueError(f"{split_name} 没有有效预测值，无法生成预测分布图。")
    valid_df["pred_boneage"] = valid_df["pred_boneage"].astype(float)
    return valid_df


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2:
        return None
    total_variance = float(np.sum((y_true - y_true.mean()) ** 2))
    if total_variance <= 1e-12:
        return None
    residual_variance = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - residual_variance / total_variance


def _build_metric_text(metrics: dict[str, Any] | None, r2: float | None = None) -> str:
    lines: list[str] = []
    if metrics is not None:
        metric_items = [
            ("loss", "Loss"),
            ("final_mae", "Final MAE"),
            ("final_mad", "Final MAD"),
            ("relative_mae", "Relative MAE"),
            ("relative_mad", "Relative MAD"),
        ]
        fallback_items = [
            ("mae", "MAE"),
            ("mad", "MAD"),
        ]
        for key, label in metric_items:
            value = _safe_float(metrics.get(key))
            if value is not None:
                lines.append(f"{label} = {value:.4f}")
        if not any(item.startswith("Final MAE") for item in lines):
            for key, label in fallback_items:
                value = _safe_float(metrics.get(key))
                if value is not None:
                    lines.append(f"{label} = {value:.4f}")
    if r2 is not None:
        lines.append(f"R^2 = {r2:.4f}")
    return "\n".join(lines)


def _plot_metric_curve(
    history_df: pd.DataFrame,
    train_column: str,
    val_column: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    history_df = _validate_history_df(history_df)
    plt = _get_pyplot(Path(output_path).parent)
    fig, axis = plt.subplots(figsize=(8, 5))

    if train_column in history_df:
        axis.plot(history_df["epoch"], history_df[train_column], label="Train", linewidth=2.0)
    if val_column in history_df:
        axis.plot(history_df["epoch"], history_df[val_column], label="Validation", linewidth=2.0)

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.legend()
    _apply_axis_style(axis)

    _save_figure(fig, output_path)
    plt.close(fig)


def plot_history(history_df: pd.DataFrame, output_path: str | Path) -> None:
    history_df = _validate_history_df(history_df)
    plt = _get_pyplot(Path(output_path).parent)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metric_specs = [
        ("train_loss", "val_loss", "Loss", "Loss"),
        ("train_mae", "val_mae", "MAE", "MAE"),
        ("train_mad", "val_mad", "MAD", "MAD"),
    ]

    for axis, (train_column, val_column, title, ylabel) in zip(axes, metric_specs):
        if train_column in history_df:
            axis.plot(history_df["epoch"], history_df[train_column], label="Train", linewidth=2.0)
        if val_column in history_df:
            axis.plot(history_df["epoch"], history_df[val_column], label="Validation", linewidth=2.0)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.legend()
        _apply_axis_style(axis)

    _save_figure(fig, output_path)
    plt.close(fig)


def plot_scatter(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    split_name: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    valid_df = _validate_prediction_df(prediction_df, split_name)
    y_true = valid_df["gt_boneage"].to_numpy(dtype=np.float64)
    y_pred = valid_df["pred_boneage"].to_numpy(dtype=np.float64)
    r2_value = _compute_r2(y_true, y_pred)

    plt = _get_pyplot(Path(output_path).parent)
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(y_true, y_pred, s=22, alpha=0.72, color="#1f77b4", edgecolors="none", label="Samples")

    axis_min = min(float(np.min(y_true)), float(np.min(y_pred)))
    axis_max = max(float(np.max(y_true)), float(np.max(y_pred)))
    margin = max((axis_max - axis_min) * 0.05, 1.0)
    line_min = axis_min - margin
    line_max = axis_max + margin
    axis.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="#444444", linewidth=1.5, label="y = x")

    fit_slope = None
    fit_intercept = None
    if len(valid_df) >= 2 and np.std(y_true) > 1e-8:
        fit_slope, fit_intercept = np.polyfit(y_true, y_pred, deg=1)
        fit_x = np.linspace(line_min, line_max, 100)
        fit_y = fit_slope * fit_x + fit_intercept
        axis.plot(fit_x, fit_y, color="#d62728", linewidth=1.8, label="Linear Fit")

    axis.set_title(f"{split_name} Scatter Plot")
    axis.set_xlabel("Ground Truth Bone Age")
    axis.set_ylabel("Predicted Bone Age")
    axis.set_xlim(line_min, line_max)
    axis.set_ylim(line_min, line_max)
    axis.legend(loc="upper left")
    _apply_axis_style(axis)

    metric_text = _build_metric_text(metrics, r2=r2_value)
    if metric_text:
        axis.text(
            0.98,
            0.02,
            metric_text,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#999999"},
        )

    _save_figure(fig, output_path)
    plt.close(fig)

    return {
        "r2": r2_value,
        "fit_slope": float(fit_slope) if fit_slope is not None else None,
        "fit_intercept": float(fit_intercept) if fit_intercept is not None else None,
        "sample_count": int(len(valid_df)),
    }


def plot_residual(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    split_name: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    valid_df = _validate_prediction_df(prediction_df, split_name)

    plt = _get_pyplot(Path(output_path).parent)
    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(
        valid_df["gt_boneage"],
        valid_df["residual"],
        s=22,
        alpha=0.72,
        color="#ff7f0e",
        edgecolors="none",
    )
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.5)
    axis.set_title(f"{split_name} Residual Plot")
    axis.set_xlabel("Ground Truth Bone Age")
    axis.set_ylabel("Residual (Prediction - Ground Truth)")
    _apply_axis_style(axis)

    metric_text = _build_metric_text(metrics)
    if metric_text:
        axis.text(
            0.98,
            0.98,
            metric_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#999999"},
        )

    _save_figure(fig, output_path)
    plt.close(fig)


def plot_error_histogram(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    split_name: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    valid_df = _validate_prediction_df(prediction_df, split_name)
    absolute_errors = valid_df["abs_error"].to_numpy(dtype=np.float64)

    plt = _get_pyplot(Path(output_path).parent)
    fig, axis = plt.subplots(figsize=(8, 5.5))
    bins = max(10, min(30, int(math.sqrt(len(absolute_errors)) * 2)))
    axis.hist(absolute_errors, bins=bins, color="#2ca02c", alpha=0.78, edgecolor="white")

    mean_error = float(np.mean(absolute_errors))
    median_error = float(np.median(absolute_errors))
    axis.axvline(mean_error, color="#d62728", linestyle="--", linewidth=1.6, label=f"Mean = {mean_error:.2f}")
    axis.axvline(median_error, color="#1f77b4", linestyle="-.", linewidth=1.6, label=f"Median = {median_error:.2f}")

    axis.set_title(f"{split_name} Absolute Error Histogram")
    axis.set_xlabel("Absolute Error")
    axis.set_ylabel("Sample Count")
    axis.legend()
    _apply_axis_style(axis)

    metric_text = _build_metric_text(metrics)
    if metric_text:
        axis.text(
            0.98,
            0.98,
            metric_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#999999"},
        )

    _save_figure(fig, output_path)
    plt.close(fig)


def plot_prediction_histogram(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    split_name: str,
) -> dict[str, float]:
    valid_df = _validate_prediction_only_df(prediction_df, split_name)
    predicted_values = valid_df["pred_boneage"].to_numpy(dtype=np.float64)

    plt = _get_pyplot(Path(output_path).parent)
    fig, axis = plt.subplots(figsize=(8, 5.5))
    bins = max(10, min(30, int(math.sqrt(len(predicted_values)) * 2)))
    axis.hist(predicted_values, bins=bins, color="#9467bd", alpha=0.80, edgecolor="white")
    axis.axvline(float(np.mean(predicted_values)), color="#d62728", linestyle="--", linewidth=1.6, label="Mean")
    axis.axvline(float(np.median(predicted_values)), color="#1f77b4", linestyle="-.", linewidth=1.6, label="Median")
    axis.set_title(f"{split_name} Prediction Histogram")
    axis.set_xlabel("Predicted Bone Age")
    axis.set_ylabel("Sample Count")
    axis.legend()
    _apply_axis_style(axis)

    _save_figure(fig, output_path)
    plt.close(fig)

    return {
        "sample_count": int(len(valid_df)),
        "pred_mean": float(np.mean(predicted_values)),
        "pred_std": float(np.std(predicted_values)),
        "pred_min": float(np.min(predicted_values)),
        "pred_max": float(np.max(predicted_values)),
    }


def _build_metrics_summary_df(
    history_df: pd.DataFrame,
    best_metric_name: str,
    best_epoch: int,
    best_val_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
) -> pd.DataFrame:
    history_df = _validate_history_df(history_df)
    summary_df = history_df.copy()
    summary_df["record_type"] = "epoch"
    summary_df["is_best_epoch"] = summary_df["epoch"].astype(int) == int(best_epoch)
    summary_df["selection_metric"] = best_metric_name
    summary_df["best_val_loss"] = np.nan
    summary_df["best_val_mae"] = np.nan
    summary_df["best_val_mad"] = np.nan
    summary_df["test_loss"] = np.nan
    summary_df["test_mae"] = np.nan
    summary_df["test_mad"] = np.nan

    best_summary_row = {column: np.nan for column in summary_df.columns}
    best_summary_row.update(
        {
            "epoch": best_epoch,
            "record_type": "best_model",
            "is_best_epoch": True,
            "selection_metric": best_metric_name,
            "best_val_loss": _safe_float(best_val_metrics.get("loss")),
            "best_val_mae": _safe_float(best_val_metrics.get("mae")),
            "best_val_mad": _safe_float(best_val_metrics.get("mad")),
            "test_loss": _safe_float(test_metrics.get("loss")) if test_metrics else None,
            "test_mae": _safe_float(test_metrics.get("mae")) if test_metrics else None,
            "test_mad": _safe_float(test_metrics.get("mad")) if test_metrics else None,
        }
    )
    return pd.concat([summary_df, pd.DataFrame([best_summary_row])], ignore_index=True)


def _write_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_text(text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def generate_training_report(
    output_dir: str | Path,
    history_df: pd.DataFrame,
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame | None,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    config: dict[str, Any],
    runtime: dict[str, Any],
    best_metric_name: str,
    best_checkpoint_path: str | Path,
    last_checkpoint_path: str | Path,
) -> dict[str, Any]:
    report_dir = _prepare_output_dir(output_dir)
    plots_dir = _prepare_output_dir(report_dir / "plots")
    history_df = _validate_history_df(history_df)
    metric_column = f"val_{best_metric_name}"
    if metric_column not in history_df.columns:
        raise ValueError(f"history_df 中缺少最佳指标列: {metric_column}")
    if history_df[metric_column].dropna().empty:
        raise ValueError(f"{metric_column} 全为空，无法确定 best epoch。")

    best_row_index = history_df[metric_column].astype(float).idxmin()
    best_epoch = int(history_df.loc[best_row_index, "epoch"])
    metrics_summary_df = _build_metrics_summary_df(
        history_df=history_df,
        best_metric_name=best_metric_name,
        best_epoch=best_epoch,
        best_val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    metrics_summary_df.to_csv(report_dir / "metrics_summary.csv", index=False)

    plot_history(history_df, plots_dir / "curves.png")
    _plot_metric_curve(history_df, "train_loss", "val_loss", plots_dir / "loss_curve.png", "Loss Curve", "Loss")
    _plot_metric_curve(history_df, "train_mae", "val_mae", plots_dir / "mae_curve.png", "MAE Curve", "MAE")
    _plot_metric_curve(history_df, "train_mad", "val_mad", plots_dir / "mad_curve.png", "MAD Curve", "MAD")

    val_scatter_stats = plot_scatter(val_predictions, plots_dir / "val_scatter.png", "Validation", metrics=val_metrics)
    plot_residual(val_predictions, plots_dir / "val_residual.png", "Validation", metrics=val_metrics)
    plot_error_histogram(val_predictions, plots_dir / "error_histogram_val.png", "Validation", metrics=val_metrics)

    test_scatter_stats = None
    test_note = None
    if test_predictions is not None:
        has_test_targets = False
        if "gt_boneage" in test_predictions.columns:
            has_test_targets = bool(test_predictions["gt_boneage"].notna().any())
        if has_test_targets:
            test_scatter_stats = plot_scatter(test_predictions, plots_dir / "test_scatter.png", "Test", metrics=test_metrics)
            plot_residual(test_predictions, plots_dir / "test_residual.png", "Test", metrics=test_metrics)
            plot_error_histogram(test_predictions, plots_dir / "error_histogram_test.png", "Test", metrics=test_metrics)
        else:
            test_scatter_stats = plot_prediction_histogram(
                test_predictions,
                plots_dir / "test_prediction_histogram.png",
                "Test",
            )
            test_note = (
                "当前 test 集缺少 Boneage 真值列，因此无法计算 test loss/MAE/MAD，"
                "也无法生成 plots/test_scatter.png、plots/test_residual.png 和 plots/error_histogram_test.png。"
                "系统已自动改为输出 plots/test_prediction_histogram.png 与 test_prediction_summary.json。"
            )
            _write_text(test_note, report_dir / "test_report_note.txt")
            _write_json(
                {
                    "note": test_note,
                    "prediction_summary": test_scatter_stats,
                },
                report_dir / "test_prediction_summary.json",
            )

    run_config_payload = {
        "config": config,
        "runtime": runtime,
    }
    _write_json(run_config_payload, report_dir / "run_config.json")

    best_metrics_payload = {
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "best_val_loss": _safe_float(val_metrics.get("loss")),
        "best_val_mae": _safe_float(val_metrics.get("mae")),
        "best_val_mad": _safe_float(val_metrics.get("mad")),
        "best_val_final_mae": _safe_float(val_metrics.get("final_mae")),
        "best_val_final_mad": _safe_float(val_metrics.get("final_mad")),
        "best_val_relative_mae": _safe_float(val_metrics.get("relative_mae")),
        "best_val_relative_mad": _safe_float(val_metrics.get("relative_mad")),
        "test_loss": _safe_float(test_metrics.get("loss")) if test_metrics else None,
        "test_mae": _safe_float(test_metrics.get("mae")) if test_metrics else None,
        "test_mad": _safe_float(test_metrics.get("mad")) if test_metrics else None,
        "test_final_mae": _safe_float(test_metrics.get("final_mae")) if test_metrics else None,
        "test_final_mad": _safe_float(test_metrics.get("final_mad")) if test_metrics else None,
        "test_relative_mae": _safe_float(test_metrics.get("relative_mae")) if test_metrics else None,
        "test_relative_mad": _safe_float(test_metrics.get("relative_mad")) if test_metrics else None,
        "val_relative_age_error_corr": _safe_float(val_metrics.get("relative_age_error_corr")),
        "val_relative_age_error_slope": _safe_float(val_metrics.get("relative_age_error_slope")),
        "test_relative_age_error_corr": _safe_float(test_metrics.get("relative_age_error_corr")) if test_metrics else None,
        "test_relative_age_error_slope": _safe_float(test_metrics.get("relative_age_error_slope")) if test_metrics else None,
        "val_r2": _safe_float(val_scatter_stats.get("r2")),
        "test_r2": _safe_float(test_scatter_stats.get("r2")) if test_scatter_stats else None,
        "val_sample_count": int(val_scatter_stats.get("sample_count", 0)),
        "test_sample_count": int(test_scatter_stats.get("sample_count", 0)) if test_scatter_stats else 0,
        "test_note": test_note,
        "best_epoch_history": {
            key: _safe_float(value) if key != "epoch" else int(value)
            for key, value in history_df.loc[best_row_index].to_dict().items()
        },
    }
    _write_json(best_metrics_payload, report_dir / "best_metrics.json")
    _write_json(best_metrics_payload, report_dir / "metrics.json")
    return best_metrics_payload
