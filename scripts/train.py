"""
Train a single architecture on the clean training set.

Usage:
    python scripts/train.py --architecture resnet18 --dataset synthetic --debug
    python scripts/train.py --architecture vit_tiny --dataset medmnist --medmnist_flag pathmnist
    python scripts/train.py --architecture mobilenet_v3_small --dataset imagefolder \
        --image_folder_train data/train --epochs 15
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import ExperimentConfig
from data.datasets import build_clean_dataset, CorruptedDatasetWrapper
from models.model_factory import build_model, count_parameters
from engine.trainer import fit
from utils.seed import set_seed


def get_device(preferred: str) -> torch.device:
    return torch.device("cuda") if (preferred == "cuda" and torch.cuda.is_available()) else torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train one architecture on the clean dataset")
    parser.add_argument("--architecture", type=str, required=True,
                         choices=["resnet18", "mobilenet_v3_small", "vit_tiny"])
    parser.add_argument("--dataset", type=str, default="synthetic",
                         choices=["synthetic", "medmnist", "imagefolder"])
    parser.add_argument("--medmnist_flag", type=str, default="pathmnist")
    parser.add_argument("--image_folder_train", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data_cache")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="1 mini-epoch on tiny synthetic data")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device("cuda")
    print(f"Using device: {device}")

    if args.debug:
        print("Running in --debug mode: synthetic data, 1 mini-epoch (pipeline smoke test only).")
        raw_train = build_clean_dataset(
            "synthetic", split="train",
            synthetic_kwargs=dict(n_samples=16, img_size=args.img_size,
                                   in_channels=args.in_channels, num_classes=args.num_classes),
        )
        args.epochs = 1
    else:
        raw_train = build_clean_dataset(
            args.dataset, split="train", medmnist_flag=args.medmnist_flag,
            data_root=args.data_root, image_folder_path=args.image_folder_train,
        )

    train_ds = CorruptedDatasetWrapper(
        raw_train, img_size=args.img_size, corruption_name=None, in_channels=args.in_channels
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    model = build_model(
        args.architecture, num_classes=args.num_classes, in_channels=args.in_channels,
        pretrained=args.pretrained, img_size=args.img_size,
    )
    print(f"{args.architecture}: {count_parameters(model):,} trainable parameters")

    model = fit(
        model, train_loader, device, epochs=args.epochs, lr=args.lr,
        grad_accum_steps=2, amp=True, log_every=10,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.architecture}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": args.architecture,
        "num_classes": args.num_classes,
        "in_channels": args.in_channels,
        "img_size": args.img_size,
    }, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
