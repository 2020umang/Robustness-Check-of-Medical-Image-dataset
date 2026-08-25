"""
Main experiment script: for each trained architecture, evaluate clean vs.
corrupted (every corruption x severity combination) on the *same* held-out
test set, compute R_df + breaking point per (architecture, corruption), and
emit a CSV table + PNG figures ready to drop into the paper.

Usage:
    # full run against checkpoints produced by scripts/train.py
    python scripts/run_corruption_sweep.py --dataset medmnist --medmnist_flag pathmnist \
        --checkpoint_dir checkpoints --results_dir results

    # quick end-to-end smoke test: trains tiny synthetic models on the fly,
    # runs a reduced severity sweep, verifies the full pipeline (no real
    # data or GPU required)
    python scripts/run_corruption_sweep.py --debug
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import ExperimentConfig
from data.datasets import build_clean_dataset, CorruptedDatasetWrapper
from models.model_factory import build_model, AVAILABLE_ARCHITECTURES
from engine.trainer import fit
from engine.evaluator import evaluate_corruption_sweep
from utils.seed import set_seed
from utils.logging_utils import results_to_dataframe, save_results_json
from utils.visualization import plot_severity_curves, plot_rdf_bar_chart


def get_device(preferred: str) -> torch.device:
    return torch.device("cuda") if (preferred == "cuda" and torch.cuda.is_available()) else torch.device("cpu")


def load_or_train_model(
    architecture: str, args, device: torch.device, raw_train_ds=None
) -> torch.nn.Module:
    ckpt_path = os.path.join(args.checkpoint_dir, f"{architecture}.pt")

    model = build_model(
        architecture, num_classes=args.num_classes, in_channels=args.in_channels,
        pretrained=args.pretrained, img_size=args.img_size,
    ).to(device)

    if os.path.exists(ckpt_path) and not args.debug:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[{architecture}] loaded checkpoint from {ckpt_path}")
        return model

    if raw_train_ds is None:
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path} and no training data provided. "
            f"Run scripts/train.py --architecture {architecture} first, or pass --debug."
        )

    print(f"[{architecture}] no checkpoint found -- training now "
          f"({'debug mini-run' if args.debug else f'{args.epochs} epochs'})")
    train_ds = CorruptedDatasetWrapper(
        raw_train_ds, img_size=args.img_size, corruption_name=None, in_channels=args.in_channels
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = fit(model, train_loader, device, epochs=(1 if args.debug else args.epochs),
                lr=args.lr, grad_accum_steps=1, amp=True, log_every=10)
    return model


def main():
    parser = argparse.ArgumentParser(description="Run the full noise-robustness benchmark")
    parser.add_argument("--architectures", type=str, nargs="+", default=AVAILABLE_ARCHITECTURES)
    parser.add_argument("--dataset", type=str, default="synthetic",
                         choices=["synthetic", "medmnist", "imagefolder"])
    parser.add_argument("--medmnist_flag", type=str, default="pathmnist")
    parser.add_argument("--image_folder_train", type=str, default=None)
    parser.add_argument("--image_folder_test", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data_cache")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--corruptions", type=str, nargs="+",
                         default=["gaussian_noise", "motion_blur", "jpeg_compression"])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--breaking_point_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true",
                         help="tiny synthetic end-to-end smoke test (trains + evaluates in seconds)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device("cuda")
    print(f"Using device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = ExperimentConfig()
    severity_to_percent = cfg.corruption.severity_to_percent

    if args.debug:
        print("Running in --debug mode with tiny synthetic data (pipeline smoke test only, "
              "NOT representative of real robustness numbers).")
        args.num_classes, args.in_channels, args.img_size = 4, 3, 64
        raw_train = build_clean_dataset(
            "synthetic", split="train",
            synthetic_kwargs=dict(n_samples=12, img_size=args.img_size,
                                   in_channels=args.in_channels, num_classes=args.num_classes),
        )
        raw_test = build_clean_dataset(
            "synthetic", split="test",
            synthetic_kwargs=dict(n_samples=8, img_size=args.img_size,
                                   in_channels=args.in_channels, num_classes=args.num_classes),
        )
        args.severities = [1, 2]
        args.batch_size = 4
    else:
        raw_train = build_clean_dataset(
            args.dataset, split="train", medmnist_flag=args.medmnist_flag,
            data_root=args.data_root, image_folder_path=args.image_folder_train,
        )
        test_split_source = args.image_folder_test or args.image_folder_train
        raw_test = build_clean_dataset(
            args.dataset, split="test", medmnist_flag=args.medmnist_flag,
            data_root=args.data_root, image_folder_path=test_split_source,
        )

    all_results = {}
    for architecture in args.architectures:
        print(f"\n=== {architecture} ===")
        model = load_or_train_model(architecture, args, device, raw_train_ds=raw_train)
        model.eval()

        sweep = evaluate_corruption_sweep(
            model, raw_test, device,
            img_size=args.img_size, in_channels=args.in_channels,
            corruptions=args.corruptions, severities=args.severities,
            batch_size=args.batch_size, num_workers=2,
            breaking_point_threshold=args.breaking_point_threshold,
            severity_to_percent=severity_to_percent,
        )
        all_results[architecture] = sweep

        if not args.debug:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{architecture}.pt")
            if not os.path.exists(ckpt_path):
                torch.save({
                    "model_state_dict": model.state_dict(), "architecture": architecture,
                    "num_classes": args.num_classes, "in_channels": args.in_channels,
                    "img_size": args.img_size,
                }, ckpt_path)
                print(f"[{architecture}] saved checkpoint -> {ckpt_path}")

    # ---- aggregate results into paper-ready artifacts ----
    df = results_to_dataframe(all_results)
    csv_path = os.path.join(args.results_dir, "robustness_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results table -> {csv_path}")

    json_path = os.path.join(args.results_dir, "robustness_results.json")
    save_results_json(all_results, json_path)
    print(f"Saved raw results -> {json_path}")

    plot_severity_curves(df, args.results_dir)
    plot_rdf_bar_chart(df, args.results_dir)
    print(f"Saved figures -> {args.results_dir}/severity_curve_*.png, {args.results_dir}/rdf_summary.png")

    print("\n=== R_df summary (lower = more robust) ===")
    summary_table = df.drop_duplicates(["architecture", "corruption"])[
        ["architecture", "corruption", "acc_clean", "r_df", "breaking_percent"]
    ]
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
