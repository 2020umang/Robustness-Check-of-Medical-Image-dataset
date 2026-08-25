"""
Model factory: builds the three architectures compared in the study,
each adapted for an arbitrary number of input channels (grayscale medical
scans vs. RGB) and output classes, behind a single `build_model()` call.

    - resnet18            (torchvision)  -- standard CNN baseline
    - mobilenet_v3_small  (torchvision)  -- ultra-lightweight CNN baseline
    - vit_tiny            (timm)         -- ultra-lightweight Vision Transformer

`pretrained=True` downloads ImageNet weights on first use (needs internet
access). Default is `False` so the whole benchmark can run fully offline;
robustness *comparisons* are still meaningful when all three models are
trained from scratch under identical conditions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)

AVAILABLE_ARCHITECTURES = ["resnet18", "mobilenet_v3_small", "vit_tiny"]


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Replace a conv layer's input channel count, preserving its other hyperparameters.
    If in_channels==3 (the layer's native config) this is a no-op passthrough."""
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, bias=(conv.bias is not None),
    )
    return new_conv


def _build_resnet18(num_classes: int, in_channels: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = _adapt_first_conv(model.conv1, in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenet_v3_small(num_classes: int, in_channels: int, pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    stem_conv = model.features[0][0]
    model.features[0][0] = _adapt_first_conv(stem_conv, in_channels)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_vit_tiny(num_classes: int, in_channels: int, pretrained: bool, img_size: int) -> nn.Module:
    try:
        import timm
    except ImportError as e:
        raise ImportError("timm is required for vit_tiny. Run `pip install timm`.") from e

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
        img_size=img_size,
    )
    return model


def build_model(
    architecture: str,
    num_classes: int,
    in_channels: int = 3,
    pretrained: bool = False,
    img_size: int = 224,
) -> nn.Module:
    """
    Args:
        architecture: one of AVAILABLE_ARCHITECTURES.
        num_classes: number of output classes.
        in_channels: 1 (grayscale) or 3 (RGB) input images.
        pretrained: load ImageNet-pretrained weights (needs internet access).
        img_size: only used by vit_tiny, must match the resize applied in the
            data pipeline (patch16 architecture requires img_size % 16 == 0).
    """
    if architecture == "resnet18":
        return _build_resnet18(num_classes, in_channels, pretrained)
    if architecture == "mobilenet_v3_small":
        return _build_mobilenet_v3_small(num_classes, in_channels, pretrained)
    if architecture == "vit_tiny":
        return _build_vit_tiny(num_classes, in_channels, pretrained, img_size)
    raise ValueError(f"Unknown architecture '{architecture}'. Available: {AVAILABLE_ARCHITECTURES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
