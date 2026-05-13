"""DDPM and DDIM samplers.

Both use the x0-clamped posterior formulation (Ho et al.'s reference DDPM)
which is numerically stable on cosine-schedule tails — the algebraically
equivalent eps-formula amplifies prediction error by 1/sqrt(alpha_t) ~ 31x
at high t, which compounds into divergent samples even for well-trained
models.

Sampling intentionally runs in fp32 (no autocast). U-Net activations at
high t can exceed fp16 range (>65504) and produce NaN. fp32 sampling is
~2x slower than fp16 but stable; training still uses fp16 autocast.
"""
import torch
import torch.nn as nn

from .config import Config
from .scheduler import Diffusion


@torch.no_grad()
def ddpm_sample(
    net: nn.Module,
    cfg: Config,
    diff: Diffusion,
    n: int,
    device: torch.device,
    return_trajectory: bool = False,
):
    """Full 1000-step reverse process with x0 clamp + fp32."""
    net.eval()
    x = torch.randn(n, cfg.in_channels, cfg.image_size, cfg.image_size, device=device)
    traj = []
    snap_every = max(1, cfg.timesteps // 20)
    for i in range(cfg.timesteps - 1, -1, -1):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = net(x, t).float()
        sac = diff.sqrt_alphas_cum[i]
        smac = diff.sqrt_one_minus_alphas_cum[i]
        x0 = ((x - smac * eps) / sac).clamp(-1, 1)
        coef_x0 = (torch.sqrt(diff.alphas_cum_prev[i]) * diff.betas[i]) / (1 - diff.alphas_cum[i])
        coef_x = (torch.sqrt(diff.alphas[i]) * (1 - diff.alphas_cum_prev[i])) / (1 - diff.alphas_cum[i])
        mean = coef_x0 * x0 + coef_x * x
        if i > 0:
            x = mean + torch.sqrt(diff.posterior_var[i]) * torch.randn_like(x)
        else:
            x = mean
        if return_trajectory and (i % snap_every == 0 or i == 0):
            traj.append(x.detach().cpu())
    return (x, traj) if return_trajectory else x


@torch.no_grad()
def ddim_sample(
    net: nn.Module,
    cfg: Config,
    diff: Diffusion,
    n: int,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
):
    """Accelerated deterministic sampling with x0 clamp + fp32."""
    net.eval()
    step_seq = torch.linspace(cfg.timesteps - 1, 0, steps, dtype=torch.long).tolist()
    x = torch.randn(n, cfg.in_channels, cfg.image_size, cfg.image_size, device=device)
    for i, t_cur in enumerate(step_seq):
        t = torch.full((n,), t_cur, device=device, dtype=torch.long)
        eps = net(x, t).float()
        a_t = diff.alphas_cum[t_cur]
        a_prev = (
            diff.alphas_cum[step_seq[i + 1]]
            if i + 1 < len(step_seq)
            else torch.tensor(1.0, device=device)
        )
        x0 = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-1, 1)
        # Re-derive eps from clamped x0 to keep the update self-consistent
        eps = (x - torch.sqrt(a_t) * x0) / torch.sqrt(1 - a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt((1 - a_prev - sigma ** 2).clamp(min=0)) * eps
        x = torch.sqrt(a_prev) * x0 + dir_xt + sigma * torch.randn_like(x)
    return x
