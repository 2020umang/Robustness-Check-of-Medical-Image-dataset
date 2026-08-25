"""Figure generation for the paper: accuracy-vs-severity curves per corruption
type (one line per architecture) and an R_df summary bar chart."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_severity_curves(df: pd.DataFrame, save_dir: str) -> None:
    """One figure per corruption type; x-axis = corruption percent,
    y-axis = accuracy, one line per architecture."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for corruption_name, sub_df in df.groupby("corruption"):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for arch_name, arch_df in sub_df.groupby("architecture"):
            arch_df = arch_df.sort_values("severity")
            ax.plot(arch_df["corruption_percent"], arch_df["accuracy"], marker="o", label=arch_name)
            clean_acc = arch_df["acc_clean"].iloc[0]
            ax.axhline(clean_acc, linestyle=":", alpha=0.3)

        ax.set_xlabel("Nominal corruption intensity (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs. {corruption_name.replace('_', ' ').title()} Severity")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(save_dir) / f"severity_curve_{corruption_name}.png", dpi=150)
        plt.close(fig)


def plot_rdf_bar_chart(df: pd.DataFrame, save_dir: str) -> None:
    """Bar chart of R_df per architecture per corruption type -- the headline
    "which model degrades most" figure."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    pivot = df.drop_duplicates(["architecture", "corruption"]).pivot(
        index="corruption", columns="architecture", values="r_df"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Robustness Degradation Factor (R_df)")
    ax.set_xlabel("Corruption type")
    ax.set_title("R_df by Architecture and Corruption Type (lower = more robust)")
    ax.legend(title="Architecture")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "rdf_summary.png", dpi=150)
    plt.close(fig)
