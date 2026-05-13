"""Centralized hyperparameter configuration."""
from dataclasses import dataclass, asdict
from typing import Tuple
import json


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Data / model dims
    image_size: int = 64
    in_channels: int = 3
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16, 8)
    time_embed_dim: int = 256
    dropout: float = 0.1

    # Diffusion
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # "linear" | "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Optimization
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 80
    lr: float = 2e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    warmup_steps: int = 500
    ema_decay: float = 0.9999
    use_amp: bool = True

    # Sampling
    sample_every: int = 5
    n_samples: int = 16
    ddim_steps: int = 50

    # Paths
    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    dataset_id: str = "huggan/smithsonian_butterflies_subset"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
