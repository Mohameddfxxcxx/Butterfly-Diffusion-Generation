"""Visualization helpers — grids, trajectories, dashboards, GIFs."""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from torchvision.utils import make_grid


def apply_dark_style():
    sns.set_theme(style="darkgrid", context="talk")
    plt.rcParams["figure.facecolor"] = "#0e1117"
    plt.rcParams["axes.facecolor"] = "#0e1117"
    plt.rcParams["savefig.facecolor"] = "#0e1117"
    plt.rcParams["axes.labelcolor"] = "#e6e6e6"
    plt.rcParams["xtick.color"] = "#cfcfcf"
    plt.rcParams["ytick.color"] = "#cfcfcf"
    plt.rcParams["axes.titlecolor"] = "#ffffff"
    plt.rcParams["text.color"] = "#e6e6e6"


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def show_grid(imgs, nrow=8, title=None, save=None, figsize=(10, 10)):
    grid = make_grid(denorm(imgs.float().cpu()), nrow=nrow, padding=2)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def make_denoising_gif(trajectory, out_path, nrow=4, duration=0.15):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for snap in trajectory:
        grid = make_grid(denorm(snap.float()), nrow=nrow, padding=2)
        arr = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(arr)
    imageio.mimsave(out_path, frames, duration=duration, loop=0)
    return out_path


def plot_loss_curve(history, save=None):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    sl = np.array(history["step_loss"])
    ax[0].plot(sl, color="#ff6ec7", alpha=0.3, label="step")
    if len(sl) > 50:
        k = max(50, len(sl) // 200)
        smooth = np.convolve(sl, np.ones(k) / k, mode="valid")
        ax[0].plot(np.arange(len(smooth)) + k - 1, smooth, color="#ffe066", label=f"EMA({k})")
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("step")
    ax[0].legend()
    ax[1].plot(history["lr"], color="#7df9ff")
    ax[1].set_title("Learning Rate")
    ax[1].set_xlabel("step")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


def plot_schedule(diff, save=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(diff.betas.cpu(), color="#ff6ec7")
    ax[0].set_title(r"$\beta_t$")
    ax[1].plot(diff.alphas_cum.cpu(), color="#7df9ff")
    ax[1].set_title(r"$\bar{\alpha}_t$")
    for a in ax:
        a.set_xlabel("t")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()
