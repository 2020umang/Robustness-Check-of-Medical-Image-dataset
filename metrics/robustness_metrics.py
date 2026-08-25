"""
Robustness Degradation Factor (R_df) -- the paper's proposed metric.

Motivation
----------
Standard "mean Corruption Error" style metrics (Hendrycks & Dietterich, 2019)
report a single averaged error rate across severities. That is useful for
ranking models, but it does not answer the paper's central question: *at
what corruption intensity does a given architecture's accuracy collapse?*
This module reports both:

  1. R_df  -- a normalized, [0, 1]-ish scalar summarizing *how much* accuracy
     is lost on average across the tested severity range, relative to clean
     accuracy. Lower is better (more robust).

  2. Breaking point (s*) -- the *specific* severity level (and its nominal
     corruption percentage) at which accuracy retention first drops below a
     chosen threshold (default 50% of clean accuracy). This is the
     quantity the paper's title refers to as the "degradation boundary".

Definitions
-----------
Let Acc_clean be accuracy on uncorrupted test data, and Acc(s) be accuracy on
the same test set corrupted at severity s in {1, ..., S}.

    Relative accuracy retention:   rho(s) = Acc(s) / Acc_clean            (1)

    Relative degradation:          delta(s) = 1 - rho(s)                  (2)

    Robustness Degradation Factor: R_df = (1/S) * sum_{s=1}^{S} delta(s)  (3)

    Breaking point:  s* = min { s : rho(s) < threshold }, else "not reached" (4)

R_df = 0 means the model is perfectly robust (no accuracy loss at any tested
severity). R_df = 1 means the model degrades to 0 accuracy across the board.
Because delta(s) can slightly exceed 1 in pathological cases (corrupted
accuracy technically above clean, e.g. lucky variance) or be negative, we
clip delta(s) to [0, 1] before averaging for interpretability; the raw,
unclipped curve is retained in the returned per-severity table for full
transparency in the paper's plots.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def relative_accuracy_retention(acc_clean: float, acc_corrupted: float) -> float:
    """rho(s) from Eq. (1). Guards against division by zero clean accuracy."""
    if acc_clean <= 0:
        return 0.0
    return acc_corrupted / acc_clean


def compute_rdf(acc_clean: float, severity_accuracies: Dict[int, float]) -> float:
    """R_df from Eq. (3), averaged over the provided severities."""
    if not severity_accuracies:
        raise ValueError("severity_accuracies must contain at least one entry")

    deltas = []
    for _, acc_s in severity_accuracies.items():
        rho = relative_accuracy_retention(acc_clean, acc_s)
        delta = 1.0 - rho
        deltas.append(float(np.clip(delta, 0.0, 1.0)))
    return float(np.mean(deltas))


def find_breaking_point(
    acc_clean: float,
    severity_accuracies: Dict[int, float],
    threshold: float = 0.5,
    severity_to_percent: Optional[Dict[int, int]] = None,
) -> Dict[str, Optional[float]]:
    """
    s* from Eq. (4): the first (lowest) severity level at which relative
    accuracy retention drops below `threshold`.

    Returns a dict with the breaking severity, its nominal corruption
    percentage (if a mapping is supplied), and the retention value at that
    point -- or None values if the model never dropped below threshold
    across the tested severities.
    """
    for severity in sorted(severity_accuracies.keys()):
        rho = relative_accuracy_retention(acc_clean, severity_accuracies[severity])
        if rho < threshold:
            percent = severity_to_percent.get(severity) if severity_to_percent else None
            return {
                "breaking_severity": severity,
                "breaking_percent": percent,
                "retention_at_break": rho,
            }
    return {"breaking_severity": None, "breaking_percent": None, "retention_at_break": None}


def summarize_robustness(
    acc_clean: float,
    severity_accuracies: Dict[int, float],
    threshold: float = 0.5,
    severity_to_percent: Optional[Dict[int, int]] = None,
) -> Dict:
    """
    Convenience wrapper combining R_df and the breaking point into one
    result dict, plus the full per-severity retention curve (handy for
    plotting accuracy-vs-severity figures in the paper).
    """
    rdf = compute_rdf(acc_clean, severity_accuracies)
    breaking = find_breaking_point(acc_clean, severity_accuracies, threshold, severity_to_percent)

    curve = {}
    for severity, acc_s in sorted(severity_accuracies.items()):
        curve[severity] = {
            "accuracy": acc_s,
            "retention": relative_accuracy_retention(acc_clean, acc_s),
            "percent": severity_to_percent.get(severity) if severity_to_percent else None,
        }

    return {
        "acc_clean": acc_clean,
        "r_df": rdf,
        **breaking,
        "severity_curve": curve,
    }
