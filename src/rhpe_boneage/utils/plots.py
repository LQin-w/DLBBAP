from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd


def plot_history(history_df: pd.DataFrame, output_path: str | Path) -> None:
    if history_df.empty:
        return

    plot_dir = Path(output_path).parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    mplconfig_dir = Path(tempfile.mkdtemp(prefix="mplconfig_", dir=plot_dir))
    os.environ["MPLCONFIGDIR"] = str(mplconfig_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    if "val_loss" in history_df:
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    if "train_mae" in history_df:
        axes[1].plot(history_df["epoch"], history_df["train_mae"], label="train")
    if "val_mae" in history_df:
        axes[1].plot(history_df["epoch"], history_df["val_mae"], label="val")
    axes[1].set_title("MAE")
    axes[1].legend()

    if "train_mad" in history_df:
        axes[2].plot(history_df["epoch"], history_df["train_mad"], label="train")
    if "val_mad" in history_df:
        axes[2].plot(history_df["epoch"], history_df["val_mad"], label="val")
    axes[2].set_title("MAD")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
