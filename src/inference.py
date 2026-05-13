"""Inference pipeline used by Streamlit and CLI tools.

Loads a checkpoint (preferring slim EMA-only format) and runs sampling with
the numerically-stable x0-clamped fp32 path. Auto-falls-back to CPU if CUDA
is unavailable.
"""
from pathlib import Path
from typing import Optional, Tuple, List
import torch

from .config import Config
from .scheduler import Diffusion
from .model import UNet
from .sampler import ddpm_sample, ddim_sample


def auto_device(prefer: Optional[str] = None) -> torch.device:
    if prefer in ("cuda", "cpu"):
        if prefer == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(prefer)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ButterflyDiffusionPipeline:
    """Inference wrapper around a trained Butterfly U-Net checkpoint."""

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.device = auto_device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Validate
        if "config" not in ckpt:
            raise ValueError(f"Checkpoint {ckpt_path} missing 'config'.")
        if "ema" not in ckpt and "model" not in ckpt:
            raise ValueError(f"Checkpoint {ckpt_path} has neither 'ema' nor 'model'.")

        self.cfg = Config.from_dict(ckpt["config"])
        self.diff = Diffusion(self.cfg, self.device)
        self.model = UNet(self.cfg).to(self.device)
        # Prefer EMA weights for inference
        state = ckpt.get("ema") or ckpt["model"]
        self.model.load_state_dict(state)
        self.model.eval()
        self.ckpt_meta = {
            "path": str(ckpt_path),
            "size_mb": round(Path(ckpt_path).stat().st_size / 1e6, 2),
            "weights": "ema" if ckpt.get("ema") is not None else "model",
            "params_M": ckpt.get("params_M", round(
                sum(p.numel() for p in self.model.parameters()) / 1e6, 2)),
            "final_loss": ckpt.get("final_loss"),
        }

    @torch.no_grad()
    def generate(
        self,
        n: int = 16,
        sampler: str = "ddim",
        steps: int = 50,
        seed: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        if seed is not None:
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
        if sampler == "ddim":
            imgs = ddim_sample(self.model, self.cfg, self.diff, n, self.device, steps=steps)
            traj = None
        elif sampler == "ddpm":
            if return_trajectory:
                imgs, traj = ddpm_sample(
                    self.model, self.cfg, self.diff, n, self.device, return_trajectory=True
                )
            else:
                imgs = ddpm_sample(self.model, self.cfg, self.diff, n, self.device)
                traj = None
        else:
            raise ValueError(f"Unknown sampler: {sampler!r}. Use 'ddpm' or 'ddim'.")

        # NaN / Inf guard — should not happen with the fp32 x0-clamped path,
        # but the demo refuses to render unstable tensors.
        if not torch.isfinite(imgs).all():
            raise RuntimeError(
                "Sampler produced non-finite values. This indicates a numerical "
                "instability; verify checkpoint integrity and try fewer steps."
            )
        return imgs, traj

    @torch.no_grad()
    def reconstruct(self, real: torch.Tensor, t_val: int = 400) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-then-reverse round trip for a real image batch."""
        real = real.to(self.device)
        t = torch.full((real.size(0),), t_val, device=self.device, dtype=torch.long)
        noisy, _ = self.diff.q_sample(real, t)
        x = noisy
        for i in range(t_val, -1, -1):
            ti = torch.full((x.size(0),), i, device=self.device, dtype=torch.long)
            eps = self.model(x, ti).float()
            sac = self.diff.sqrt_alphas_cum[i]
            smac = self.diff.sqrt_one_minus_alphas_cum[i]
            x0 = ((x - smac * eps) / sac).clamp(-1, 1)
            coef_x0 = (torch.sqrt(self.diff.alphas_cum_prev[i]) * self.diff.betas[i]) / (1 - self.diff.alphas_cum[i])
            coef_x = (torch.sqrt(self.diff.alphas[i]) * (1 - self.diff.alphas_cum_prev[i])) / (1 - self.diff.alphas_cum[i])
            mean = coef_x0 * x0 + coef_x * x
            if i > 0:
                x = mean + torch.sqrt(self.diff.posterior_var[i]) * torch.randn_like(x)
            else:
                x = mean
        return noisy, x

    @torch.no_grad()
    def interpolate(self, n_pairs: int = 4, steps_per_pair: int = 8, ddim_steps: int = 50) -> torch.Tensor:
        """Spherical-linear interpolation in noise space, then DDIM decode."""
        import numpy as np
        rows = []
        step_seq = torch.linspace(self.cfg.timesteps - 1, 0, ddim_steps, dtype=torch.long).tolist()
        for _ in range(n_pairs):
            z1 = torch.randn(1, self.cfg.in_channels, self.cfg.image_size, self.cfg.image_size,
                             device=self.device)
            z2 = torch.randn(1, self.cfg.in_channels, self.cfg.image_size, self.cfg.image_size,
                             device=self.device)
            zs = []
            for a in np.linspace(0, 1, steps_per_pair):
                t = torch.tensor([[float(a)]], dtype=torch.float32, device=self.device)
                z1f, z2f = z1.flatten(1), z2.flatten(1)
                z1n = z1f / z1f.norm(dim=1, keepdim=True)
                z2n = z2f / z2f.norm(dim=1, keepdim=True)
                omega = torch.acos((z1n * z2n).sum(dim=1, keepdim=True).clamp(-1, 1))
                so = torch.sin(omega)
                z = (torch.sin((1 - t) * omega) / so) * z1f + (torch.sin(t * omega) / so) * z2f
                zs.append(z.view_as(z1))
            x = torch.cat(zs, dim=0)
            for i, t_cur in enumerate(step_seq):
                ti = torch.full((x.size(0),), t_cur, device=self.device, dtype=torch.long)
                eps = self.model(x, ti).float()
                a_t = self.diff.alphas_cum[t_cur]
                a_prev = (self.diff.alphas_cum[step_seq[i + 1]]
                          if i + 1 < len(step_seq) else torch.tensor(1.0, device=self.device))
                x0 = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-1, 1)
                eps = (x - torch.sqrt(a_t) * x0) / torch.sqrt(1 - a_t)
                x = torch.sqrt(a_prev) * x0 + torch.sqrt((1 - a_prev).clamp(min=0)) * eps
            rows.append(x.cpu())
        return torch.cat(rows, dim=0)

    @torch.no_grad()
    def feature_maps(self, t_val: int = 500, n_channels: int = 8) -> torch.Tensor:
        """Pull n_channels mid-block activations at a single noise level."""
        x = torch.randn(1, self.cfg.in_channels, self.cfg.image_size, self.cfg.image_size,
                        device=self.device)
        t = torch.tensor([t_val], device=self.device, dtype=torch.long)
        feats = {}

        def hook(_m, _i, out):
            feats["x"] = out.detach().float().cpu()

        h = self.model.mid[0].register_forward_hook(hook)
        _ = self.model(x, t)
        h.remove()
        return feats["x"][0, :n_channels]

    @staticmethod
    def to_pil_grid(imgs: torch.Tensor, nrow: int = 4):
        from torchvision.utils import make_grid
        from PIL import Image
        from .utils.visualize import denorm
        grid = make_grid(denorm(imgs.float().cpu()), nrow=nrow, padding=2)
        arr = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)
