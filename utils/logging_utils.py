"""Helpers to turn the nested robustness-sweep result dicts into paper-ready
artifacts: a flat CSV-able DataFrame and a JSON dump for archival."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def results_to_dataframe(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    all_results: {architecture_name: sweep_result_dict} where sweep_result_dict
    is the output of engine.evaluator.evaluate_corruption_sweep.

    Returns a tidy long-format DataFrame with one row per
    (architecture, corruption, severity), suitable for `df.to_csv(...)`
    and for plotting with seaborn/matplotlib directly.
    """
    rows = []
    for arch_name, sweep in all_results.items():
        acc_clean = sweep["clean"]["accuracy"]
        for corruption_name, summary in sweep.items():
            if corruption_name == "clean":
                continue
            for severity, point in summary["severity_curve"].items():
                rows.append({
                    "architecture": arch_name,
                    "corruption": corruption_name,
                    "severity": severity,
                    "corruption_percent": point["percent"],
                    "accuracy": point["accuracy"],
                    "retention": point["retention"],
                    "acc_clean": acc_clean,
                    "r_df": summary["r_df"],
                    "breaking_severity": summary["breaking_severity"],
                    "breaking_percent": summary["breaking_percent"],
                })
    return pd.DataFrame(rows)


def save_results_json(all_results: Dict[str, Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
