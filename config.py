"""
Central configuration for the robustness degradation study.

    "Quantifying the Degradation Boundary of Lightweight Vision Architectures
    Under Synthetic Environmental Noise in Medical Imaging"

Keep every tunable here so scripts stay declarative and experiments stay
reproducible / citable in the paper's methodology section.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    dataset_name: str = "synthetic"      # "medmnist", "imagefolder", or "synthetic"
    medmnist_flag: str = "pathmnist"     # any MedMNIST-v2 flag, e.g. pathmnist/dermamnist/octmnist
    data_root: str = "data_cache"        # where MedMNIST / downloaded data is cached
    image_folder_train: str = None       # used when dataset_name == "imagefolder"
    image_folder_test: str = None
    in_channels: int = 3
    num_classes: int = 9                 # default for pathmnist; overridden per-dataset
    img_size: int = 224                  # resize target; ViT-Tiny (patch16) expects 224
    num_workers: int = 2


@dataclass
class ModelConfig:
    architectures: List[str] = field(
        default_factory=lambda: ["resnet18", "mobilenet_v3_small", "vit_tiny"]
    )
    pretrained: bool = False             # True requires internet access to fetch ImageNet weights


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_accum_steps: int = 2
    amp: bool = True
    log_every: int = 20
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "cuda"


@dataclass
class CorruptionConfig:
    corruptions: List[str] = field(
        default_factory=lambda: ["gaussian_noise", "motion_blur", "jpeg_compression"]
    )
    severities: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    # nominal "noise percentage" label attached to each severity level, purely for
    # reporting on an intuitive 0-100% axis in plots / the R_df table
    severity_to_percent: dict = field(
        default_factory=lambda: {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
    )
    breaking_point_threshold: float = 0.5  # accuracy retention fraction defining "failure"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    results_dir: str = "results"


default_config = ExperimentConfig()
