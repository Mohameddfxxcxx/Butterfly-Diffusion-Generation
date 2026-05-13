"""Checkpoint save/load helpers."""
from dataclasses import asdict
from pathlib import Path
import torch

from ..config import Config


def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, epoch, loss, cfg: Config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "config": asdict(cfg),
        },
        path,
    )


def load_checkpoint(path, model=None, ema=None, optimizer=None, scheduler=None, scaler=None,
                    map_location="cuda"):
    ckpt = torch.load(path, map_location=map_location)
    if model is not None and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
