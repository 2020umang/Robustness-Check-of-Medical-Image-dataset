"""
Architecture-agnostic training loop. Works identically for resnet18,
mobilenet_v3_small, and vit_tiny since `build_model()` gives them all the
same nn.Module interface (forward(x) -> logits).
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    grad_accum_steps: int = 1,
    amp: bool = True,
    log_every: int = 20,
    epoch: int = 0,
) -> float:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps

        if (step + 1) % log_every == 0:
            print(f"  [epoch {epoch}] step {step + 1}/{len(loader)} - loss: {running_loss / (step + 1):.4f}")

    return running_loss / max(len(loader), 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_accum_steps: int = 1,
    amp: bool = True,
    log_every: int = 20,
) -> nn.Module:
    """Minimal end-to-end training driver used by scripts/train.py."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    model.to(device)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_accum_steps=grad_accum_steps, amp=amp, log_every=log_every, epoch=epoch,
        )
        print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - time: {time.time() - t0:.1f}s")
    return model
