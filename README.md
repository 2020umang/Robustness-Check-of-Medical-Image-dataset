# Robustness Degradation Benchmark for Lightweight Vision Architectures

Codebase accompanying the study on noise robustness of lightweight CNNs and
Vision Transformers under synthetic real-world sensor/transmission
degradation in medical imaging.

> *Quantifying the Degradation Boundary of Lightweight Vision Architectures
> Under Synthetic Environmental Noise in Medical Imaging*

## Research question

Medical scanners and IoT-connected cameras in real deployments introduce
blur, sensor noise, and lossy compression that break models validated only
on clean benchmark data. This project quantifies, per architecture, **at
what corruption intensity accuracy collapses**, via a proposed
**Robustness Degradation Factor (R_df)** and an explicit **breaking point**
(the corruption severity at which accuracy retention first drops below a
threshold, default 50% of clean accuracy).

## Method summary

1. **Dataset**: any MedMNIST-v2 collection (e.g. PathMNIST, DermaMNIST,
   OCTMNIST), a Kaggle-style `class_x/*.jpg` folder dataset (e.g. Retinal
   OCT, Skin Lesions), or synthetic random tensors for pipeline testing.
2. **Corruption injection** (`data/corruptions.py`): 3 corruption families x
   5 severity levels, applied with OpenCV —
   - Gaussian noise (poor low-light sensors)
   - Motion blur (patient / handheld camera movement)
   - JPEG compression artifacts (lossy network transmission)
3. **Architectures** (`models/model_factory.py`), all built through one
   shared factory function:
   - ResNet-18 (torchvision) — standard CNN baseline
   - MobileNetV3-Small (torchvision) — ultra-lightweight CNN baseline
   - ViT-Tiny/16 (timm) — ultra-lightweight Vision Transformer
4. **Metric** (`metrics/robustness_metrics.py`): for each (architecture,
   corruption type), across severities s=1..5:
   - `rho(s) = Acc(s) / Acc_clean` — relative accuracy retention
   - `R_df = mean_s(1 - rho(s))`, clipped to [0,1] — overall degradation
   - breaking point `s*` — first severity where `rho(s) < threshold`
     (default 0.5), reported both as a severity level and a nominal
     corruption percentage.

Full metric definitions and rationale are documented in the module
docstring of `metrics/robustness_metrics.py` — copy directly into the
paper's methodology section.

## Project layout

```
robustness_pipeline/
  config.py                    # all experiment hyperparameters
  data/
    corruptions.py              # Gaussian noise / motion blur / JPEG injection
    datasets.py                  # MedMNIST / ImageFolder / synthetic loaders + corruption wrapper
  models/
    model_factory.py             # resnet18 / mobilenet_v3_small / vit_tiny, shared interface
  metrics/
    robustness_metrics.py         # R_df + breaking-point computation (core contribution)
  engine/
    trainer.py                    # architecture-agnostic training loop
    evaluator.py                  # clean + corruption-sweep evaluation
  utils/
    seed.py, logging_utils.py, visualization.py
  scripts/
    train.py                      # train one architecture, save checkpoint
    run_corruption_sweep.py        # full benchmark: eval all archs x all corruptions x severities
  results/                         # CSV / JSON / PNG outputs land here
  checkpoints/                     # trained model weights land here
```

## Setup

```bash
pip install -r requirements.txt
```

`medmnist` and internet access to Zenodo (for `--dataset medmnist`) or to
your own Kaggle download (for `--dataset imagefolder`) are only needed for
real experiments — the pipeline can be fully smoke-tested offline first
(see below).

## Reproducing the pipeline (fast sanity check, no data/GPU needed)

```bash
python scripts/run_corruption_sweep.py --debug
```

This trains tiny versions of all three architectures on synthetic random
images for one mini-epoch, runs a reduced corruption sweep, and writes
`results/robustness_results.csv`, `.json`, and the PNG figures — useful to
confirm the full pipeline (data -> corruption -> model -> metric -> plot)
is wired correctly before spending compute on real training.

## Running the real experiment

```bash
# 1) train each architecture on the clean training split
python scripts/train.py --architecture resnet18 --dataset medmnist \
    --medmnist_flag pathmnist --in_channels 3 --num_classes 9 --epochs 15

python scripts/train.py --architecture mobilenet_v3_small --dataset medmnist \
    --medmnist_flag pathmnist --in_channels 3 --num_classes 9 --epochs 15

python scripts/train.py --architecture vit_tiny --dataset medmnist \
    --medmnist_flag pathmnist --in_channels 3 --num_classes 9 --epochs 15

# 2) run the full corruption sweep against the saved checkpoints
python scripts/run_corruption_sweep:.py --dataset medmnist --medmnist_flag pathmnist \
    --in_channels 3 --num_classes 9 \
    --checkpoint_dir checkpoints --results_dir results
```

For a Kaggle-style folder dataset instead:

```bash
python scripts/run_corruption_sweep.py --dataset imagefolder \
    --image_folder_train data/train --image_folder_test data/test \
    --in_channels 3 --num_classes <N> --checkpoint_dir checkpoints --results_dir results
```

If a checkpoint for an architecture is not found, `run_corruption_sweep.py`
will train it on the fly using `--epochs`/`--batch_size`/`--lr`, so scripts
1 and 2 above can also be collapsed into a single call.

## Outputs

- `results/robustness_results.csv` — tidy long-format table (one row per
  architecture x corruption x severity), directly loadable for the paper's
  result tables/statistical tests.
- `results/robustness_results.json` — full nested results, including the
  R_df and breaking-point summary per (architecture, corruption).
- `results/severity_curve_<corruption>.png` — accuracy-vs-severity plot,
  one line per architecture, per corruption type.
- `results/rdf_summary.png` — R_df bar chart comparing all architectures
  across all corruption types (the paper's headline figure).

## Compute-budget notes

- Default training config uses a small batch size (16) with gradient
  accumulation and AMP, and the corruption sweep reuses the same raw
  dataset object across all (corruption, severity) combinations rather than
  duplicating storage.
- `--debug` mode on every script lets you validate correctness with
  synthetic data before running on real (larger, slower) medical imaging
  data or downloading pretrained weights.
- `pretrained=False` is the default for all three architectures so the full
  benchmark can be run without internet access; set `--pretrained` only if
  you have connectivity to fetch ImageNet weights and want to study
  fine-tuning robustness specifically.
