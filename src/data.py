"""Dataset loader for Smithsonian Butterflies with GPU-ready transforms."""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from .config import Config


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def make_dataloader(cfg: Config):
    ds = load_dataset(cfg.dataset_id, split="train")
    tfm = build_transform(cfg.image_size)

    def collate(batch):
        imgs = torch.stack([tfm(b["image"].convert("RGB")) for b in batch], dim=0)
        return imgs.contiguous(memory_format=torch.channels_last)

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
        collate_fn=collate,
    )
    return loader, ds
