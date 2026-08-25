"""
Evaluation utilities:
  - evaluate_accuracy: plain top-1 accuracy on a DataLoader.
  - evaluate_corruption_sweep: runs a model over clean data + every
    (corruption, severity) combination on the *same* underlying raw
    dataset, and packages the result via metrics.summarize_robustness.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.datasets import CorruptedDatasetWrapper
from metrics.robustness_metrics import summarize_robustness


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def evaluate_corruption_sweep(
    model: nn.Module,
    raw_test_dataset: Dataset,
    device: torch.device,
    img_size: int,
    in_channels: int,
    corruptions: List[str],
    severities: List[int],
    batch_size: int = 16,
    num_workers: int = 2,
    breaking_point_threshold: float = 0.5,
    severity_to_percent: Dict[int, int] = None,
) -> Dict[str, Dict]:
    """
    Returns:
        {
          "clean": {"accuracy": float},
          "gaussian_noise": {"acc_clean": ..., "r_df": ..., "breaking_severity": ...,
                              "severity_curve": {1: {...}, 2: {...}, ...}},
          "motion_blur": {...},
          "jpeg_compression": {...},
        }
    """
    # --- clean baseline (corruption_name=None) ---
    clean_ds = CorruptedDatasetWrapper(
        raw_test_dataset, img_size=img_size, corruption_name=None, in_channels=in_channels
    )
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    acc_clean = evaluate_accuracy(model, clean_loader, device)
    print(f"  clean accuracy: {acc_clean:.4f}")

    results = {"clean": {"accuracy": acc_clean}}

    for corruption_name in corruptions:
        severity_accuracies = {}
        for severity in severities:
            corrupt_ds = CorruptedDatasetWrapper(
                raw_test_dataset, img_size=img_size, corruption_name=corruption_name,
                severity=severity, in_channels=in_channels,
            )
            corrupt_loader = DataLoader(corrupt_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            acc_s = evaluate_accuracy(model, corrupt_loader, device)
            severity_accuracies[severity] = acc_s
            print(f"  {corruption_name} severity={severity}: accuracy={acc_s:.4f}")

        summary = summarize_robustness(
            acc_clean, severity_accuracies,
            threshold=breaking_point_threshold, severity_to_percent=severity_to_percent,
        )
        results[corruption_name] = summary

    return results
