"""Noise schedules (linear, cosine) and forward diffusion utilities."""
import math
import torch
import torch.nn.functional as F

from .config import Config


def linear_beta_schedule(T: int, b0: float, b1: float) -> torch.Tensor:
    return torch.linspace(b0, b1, T, dtype=torch.float32)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cum = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    return betas.clamp(1e-5, 0.999).float()


class Diffusion:
    """Precomputed diffusion buffers and forward sampler."""

    def __init__(self, cfg: Config, device: torch.device):
        self.T = cfg.timesteps
        betas = (
            cosine_beta_schedule(self.T)
            if cfg.beta_schedule == "cosine"
            else linear_beta_schedule(self.T, cfg.beta_start, cfg.beta_end)
        )
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        alphas_cum_prev = F.pad(alphas_cum[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cum = alphas_cum.to(device)
        self.alphas_cum_prev = alphas_cum_prev.to(device)
        self.sqrt_alphas_cum = torch.sqrt(alphas_cum).to(device)
        self.sqrt_one_minus_alphas_cum = torch.sqrt(1 - alphas_cum).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
        self.posterior_var = (betas * (1 - alphas_cum_prev) / (1 - alphas_cum)).to(device)
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alphas_cum.gather(0, t).view(-1, 1, 1, 1)
        sm = self.sqrt_one_minus_alphas_cum.gather(0, t).view(-1, 1, 1, 1)
        return sa * x0 + sm * noise, noise
