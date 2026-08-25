"""
Dataset layer.

Design principle: "base" datasets return a **raw uint8 numpy image** (HxWxC)
and an integer label, with no resizing/normalization applied. Corruption
injection (data/corruptions.py) and the final torchvision transform pipeline
are applied afterwards by `CorruptedDatasetWrapper`. This lets the exact same
underlying image be evaluated both clean and under every corruption/severity
combination, which is essential for a fair robustness comparison.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .corruptions import apply_corruption

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Raw base datasets
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Random uint8 images for pipeline smoke-testing without any real data
    or network access. NOT for reporting actual robustness numbers."""

    def __init__(self, n_samples: int = 32, img_size: int = 224,
                 in_channels: int = 3, num_classes: int = 4):
        self.n = n_samples
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        shape = (self.img_size, self.img_size, self.in_channels)
        img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        label = int(np.random.randint(0, self.num_classes))
        return img, label


class RawImageFolderDataset(Dataset):
    """Generic root/class_name/*.{jpg,png,...} loader returning raw numpy images.
    Suitable for Kaggle-style Retinal OCT / Skin Lesion datasets."""

    def __init__(self, root: str):
        self.root = Path(root)
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: List[Tuple[str, int]] = []
        for c in classes:
            for f in (self.root / c).iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    self.samples.append((str(f), self.class_to_idx[c]))
        if not self.samples:
            raise ValueError(f"No images found under {root} (expected root/class_x/*.jpg layout).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        import cv2
        filepath, label = self.samples[idx]
        img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {filepath}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, label


class RawMedMNISTDataset(Dataset):
    """Wraps a MedMNIST-v2 dataset (https://medmnist.com/) and returns raw
    uint8 numpy images. Requires `pip install medmnist`; downloading the
    actual .npz data the first time requires internet access to Zenodo."""

    def __init__(self, flag: str, split: str, data_root: str):
        try:
            import medmnist
            from medmnist import INFO
        except ImportError as e:
            raise ImportError(
                "medmnist is not installed. Run `pip install medmnist` to use "
                "dataset_name='medmnist'."
            ) from e

        if flag not in INFO:
            raise ValueError(f"Unknown MedMNIST flag '{flag}'. Available: {list(INFO.keys())}")

        os.makedirs(data_root, exist_ok=True)
        info = INFO[flag]
        dataset_class = getattr(medmnist, info["python_class"])
        self.dataset = dataset_class(split=split, download=True, root=data_root)
        self.num_classes = len(info["label"])
        self.in_channels = info["n_channels"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        img, label = self.dataset[idx]  # PIL image, numpy label array
        img = np.array(img)
        if img.ndim == 2:
            img = img[:, :, None]
        label = int(np.array(label).squeeze())
        return img, label


def build_clean_dataset(
    dataset_name: str,
    split: str,
    medmnist_flag: str = "pathmnist",
    data_root: str = "data_cache",
    image_folder_path: Optional[str] = None,
    synthetic_kwargs: Optional[dict] = None,
) -> Dataset:
    """Factory returning a raw (uint8, uncorrupted, untransformed) dataset."""
    if dataset_name == "synthetic":
        kwargs = synthetic_kwargs or {}
        return SyntheticDataset(**kwargs)
    if dataset_name == "medmnist":
        return RawMedMNISTDataset(flag=medmnist_flag, split=split, data_root=data_root)
    if dataset_name == "imagefolder":
        if image_folder_path is None:
            raise ValueError("image_folder_path must be set when dataset_name='imagefolder'")
        return RawImageFolderDataset(image_folder_path)
    raise ValueError(f"Unknown dataset_name '{dataset_name}'")


# ---------------------------------------------------------------------------
# Corruption-injecting wrapper (applied on top of any raw dataset above)
# ---------------------------------------------------------------------------

class CorruptedDatasetWrapper(Dataset):
    """
    Wraps a raw (uint8, HxWxC) dataset. At `__getitem__` time, optionally
    applies a named corruption at a given severity, then resizes/normalizes
    into a model-ready tensor.

    Passing `corruption_name=None` yields the clean (uncorrupted) pipeline,
    so the *same* underlying images can be benchmarked clean vs. corrupted.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        img_size: int = 224,
        corruption_name: Optional[str] = None,
        severity: Optional[int] = None,
        in_channels: int = 3,
    ):
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.in_channels = in_channels

        norm_mean = IMAGENET_MEAN[:in_channels] if in_channels <= 3 else IMAGENET_MEAN
        norm_std = IMAGENET_STD[:in_channels] if in_channels <= 3 else IMAGENET_STD
        if in_channels == 1:
            norm_mean, norm_std = [0.5], [0.5]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        if corruption_name is not None and severity is None:
            raise ValueError("severity must be given when corruption_name is set")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, label = self.base_dataset[idx]  # raw uint8 HxWxC

        # normalize channel count to what the dataset declares (grayscale vs RGB)
        if img.ndim == 2:
            img = img[:, :, None]
        if self.in_channels == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif self.in_channels == 1 and img.shape[2] == 3:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, None]

        if self.corruption_name is not None:
            img = apply_corruption(img, self.corruption_name, self.severity)

        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)
