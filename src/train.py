"""CLI training entry point. Mirrors the notebook pipeline."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from .config import Config
from .data import make_dataloader
from .scheduler import Diffusion
from .model import UNet
from .ema import EMA
from .sampler import ddim_sample
from .utils import set_seed, gpu_mem_mb, gpu_info, save_checkpoint
from .utils.visualize import apply_dark_style, show_grid, plot_loss_curve


def parse_args():
    p = argparse.ArgumentParser("butterfly-diffusion-train")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    apply_dark_style()
    args = parse_args()
    cfg = Config()
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.image_size: cfg.image_size = args.image_size
    if args.seed: cfg.seed = args.seed

    assert torch.cuda.is_available(), "GPU required."
    device = torch.device("cuda")
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    print(gpu_info())

    Path(cfg.out_dir, "samples").mkdir(parents=True, exist_ok=True)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    loader, _ = make_dataloader(cfg)
    diff = Diffusion(cfg, device)
    model = UNet(cfg).to(device, memory_format=torch.channels_last)
    ema = EMA(model, cfg.ema_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay
    )
    total_steps = cfg.epochs * len(loader)

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_amp)

    history = {"step_loss": [], "epoch_loss": [], "lr": [], "gpu_mem": [], "epoch_time": []}
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_losses = []
        ep_t0 = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for x0 in pbar:
            x0 = x0.to(device, non_blocking=True)
            t = torch.randint(0, cfg.timesteps, (x0.size(0),), device=device, dtype=torch.long)
            with autocast(enabled=cfg.use_amp, dtype=torch.float16):
                xt, noise = diff.q_sample(x0, t)
                pred = model(xt, t)
                loss = F.mse_loss(pred, noise)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            v = loss.item()
            history["step_loss"].append(v)
            history["lr"].append(scheduler.get_last_lr()[0])
            ep_losses.append(v)
            pbar.set_postfix(loss=f"{v:.4f}", mem=f"{gpu_mem_mb():.0f}MB")

        ep_loss = float(np.mean(ep_losses))
        ep_time = time.time() - ep_t0
        history["epoch_loss"].append(ep_loss)
        history["gpu_mem"].append(gpu_mem_mb())
        history["epoch_time"].append(ep_time)
        print(f"[{epoch:03d}] loss={ep_loss:.4f}  mem={gpu_mem_mb():.0f}MB  time={ep_time:.1f}s")

        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            imgs = ddim_sample(ema.shadow, cfg, diff, cfg.n_samples, device, cfg.ddim_steps)
            show_grid(
                imgs, nrow=4, title=f"EMA samples — epoch {epoch}",
                save=f"{cfg.out_dir}/samples/epoch_{epoch:03d}.png", figsize=(8, 8),
            )

        if ep_loss < best_loss:
            best_loss = ep_loss
            save_checkpoint(f"{cfg.ckpt_dir}/best.pt", model, ema, optimizer, scheduler, scaler,
                            epoch, ep_loss, cfg)

    save_checkpoint(f"{cfg.ckpt_dir}/final.pt", model, ema, optimizer, scheduler, scaler,
                    cfg.epochs, history["epoch_loss"][-1], cfg)
    plot_loss_curve(history, save=f"{cfg.out_dir}/loss_curve.png")
    print(f"Done in {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    main()
