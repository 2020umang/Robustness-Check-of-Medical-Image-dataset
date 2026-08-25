"""
Synthetic real-world sensor/transmission corruption injection.

This module operationalizes the paper's central experimental variable: a
5-level severity scale (loosely modeled on Hendrycks & Dietterich, 2019,
"Benchmarking Neural Network Robustness to Common Corruptions and
Perturbations") applied to three corruption families that are representative
of real-world medical/IoT imaging degradation:

    - Gaussian noise      -> poor low-light sensor noise
    - Motion blur         -> patient/handheld camera movement during capture
    - JPEG compression    -> lossy artifacts from network transmission / PACS storage

All functions operate on uint8 HxW or HxWxC numpy arrays (OpenCV convention)
and return an array of the same shape/dtype, so they can be dropped into any
dataset's `__getitem__` before the standard torchvision transform pipeline.

Severity level 1 (mildest) to 5 (most severe) is the analysis axis for the
Robustness Degradation Factor (see metrics/robustness_metrics.py). Each
severity is also mapped to a nominal "corruption percentage" purely for
intuitive reporting/plotting (see SEVERITY_TO_PERCENT below).
"""
from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Severity schedules -- tune these once, cite them in the paper's appendix.
# ---------------------------------------------------------------------------

# Standard deviation of additive Gaussian noise, expressed as a fraction of
# the [0, 1] pixel intensity range.
_GAUSSIAN_NOISE_STD = {1: 0.03, 2: 0.06, 3: 0.09, 4: 0.13, 5: 0.18}

# Linear motion-blur kernel length (pixels).
_MOTION_BLUR_KERNEL = {1: 3, 2: 5, 3: 9, 4: 13, 5: 18}

# JPEG quality factor (lower = more compression artifacts / more severe).
_JPEG_QUALITY = {1: 70, 2: 50, 3: 35, 4: 20, 5: 10}

# Purely cosmetic mapping from severity level -> nominal corruption percentage,
# used to phrase results as "accuracy collapses beyond X% corruption intensity".
SEVERITY_TO_PERCENT: Dict[int, int] = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def add_gaussian_noise(image: np.ndarray, severity: int) -> np.ndarray:
    """Simulates sensor noise from poor low-light conditions."""
    assert severity in _GAUSSIAN_NOISE_STD, f"severity must be in 1..5, got {severity}"
    std = _GAUSSIAN_NOISE_STD[severity]

    img_float = image.astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=std, size=img_float.shape).astype(np.float32)
    noisy = np.clip(img_float + noise, 0.0, 1.0) * 255.0
    return _ensure_uint8(noisy)


def add_motion_blur(image: np.ndarray, severity: int) -> np.ndarray:
    """Simulates blur from patient movement or handheld camera shake."""
    assert severity in _MOTION_BLUR_KERNEL, f"severity must be in 1..5, got {severity}"
    k = _MOTION_BLUR_KERNEL[severity]

    # Random-angle linear motion kernel: draw a line through the center of a k x k kernel.
    angle = np.random.uniform(0, 180)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    rot_matrix = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_matrix, (k, k))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        kernel[k // 2, k // 2] = 1.0

    blurred = cv2.filter2D(image, -1, kernel)
    return _ensure_uint8(blurred)


def add_jpeg_compression(image: np.ndarray, severity: int) -> np.ndarray:
    """Simulates lossy compression artifacts from network transmission / storage."""
    assert severity in _JPEG_QUALITY, f"severity must be in 1..5, got {severity}"
    quality = _JPEG_QUALITY[severity]

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", image, encode_params)
    if not success:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    # imdecode may drop a channel dim for grayscale input; restore original shape.
    if decoded.ndim != image.ndim:
        decoded = decoded.reshape(image.shape)
    return _ensure_uint8(decoded)


CORRUPTION_REGISTRY: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise": add_gaussian_noise,
    "motion_blur": add_motion_blur,
    "jpeg_compression": add_jpeg_compression,
}


def apply_corruption(image: np.ndarray, corruption_name: str, severity: int) -> np.ndarray:
    """Dispatch helper: apply a named corruption at a given severity (1-5) to an image."""
    if corruption_name not in CORRUPTION_REGISTRY:
        raise ValueError(
            f"Unknown corruption '{corruption_name}'. Available: {list(CORRUPTION_REGISTRY)}"
        )
    return CORRUPTION_REGISTRY[corruption_name](image, severity)
