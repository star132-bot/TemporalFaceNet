# Cross Temporal Face Recognition
# Configuration Management

import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Dataset configuration"""
    orangutan_root: str = "data/orangutan"
    human_root: str = "data/human"
    cacd_root: str = "data/human/cacd"
    fg_net_root: str = "data/human/fg_net"
    agedb_root: str = "data/human/agedb"

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class ModelConfig:
    """Model configuration"""
    backbone: str = "swin_tiny_patch4_window7_224"
    embed_dim: int = 768
    num_heads: int = 12
    depth: int = 12

    use_temporal_cl: bool = True
    temperature: float = 0.07
    num_negatives: int = 16384

    pretrained: bool = True


@dataclass
class TrainConfig:
    """Training configuration"""
    epochs: int = 100
    start_epoch: int = 0

    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    optimizer: str = "adamw"

    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0

    label_smoothing: float = 0.1

    save_freq: int = 10
    eval_freq: int = 5
    print_freq: int = 50

    use_amp: bool = True
    grad_clip: float = 1.0


@dataclass
class LossConfig:
    """Loss configuration"""
    ce_weight: float = 1.0
    triplet_weight: float = 0.5
    tcl_weight: float = 1.0
    center_weight: float = 0.5

    margin: float = 0.3


@dataclass
class Config:
    """Main configuration"""
    project_name: str = "CrossTemporalFaceRecognition"
    seed: int = 42
    device: str = "cuda"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "outputs/logs"


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = Config()

    # Update data config
    if 'data' in config_dict:
        for key, value in config_dict['data'].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)

    # Update model config
    if 'model' in config_dict:
        for key, value in config_dict['model'].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    # Update train config
    if 'train' in config_dict:
        for key, value in config_dict['train'].items():
            if hasattr(config.train, key):
                setattr(config.train, key, value)

    # Update loss config
    if 'loss' in config_dict:
        for key, value in config_dict['loss'].items():
            if hasattr(config.loss, key):
                setattr(config.loss, key, value)

    # Update main config
    for key in ['project_name', 'seed', 'device', 'output_dir', 'model_dir', 'log_dir']:
        if key in config_dict:
            setattr(config, key, config_dict[key])

    return config


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file"""
    config_dict = {
        'project_name': config.project_name,
        'seed': config.seed,
        'device': config.device,
        'output_dir': config.output_dir,
        'model_dir': config.model_dir,
        'log_dir': config.log_dir,
        'data': vars(config.data),
        'model': vars(config.model),
        'train': vars(config.train),
        'loss': vars(config.loss),
    }

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
