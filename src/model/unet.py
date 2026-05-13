"""U-Net with optional self-attention at configurable resolutions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .embeddings import SinusoidalTimeEmbedding
from .blocks import ResBlock, AttnBlock, Downsample, Upsample


class UNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        C = cfg.base_channels
        mults = cfg.channel_mult
        nb = cfg.num_res_blocks
        t_dim = cfg.time_embed_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(C),
            nn.Linear(C, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.in_conv = nn.Conv2d(cfg.in_channels, C, 3, padding=1)

        chs = [C]
        cur = C
        res = cfg.image_size
        self.down_blocks = nn.ModuleList()
        for i, m in enumerate(mults):
            out_ch = C * m
            for _ in range(nb):
                block = nn.ModuleList([ResBlock(cur, out_ch, t_dim, cfg.dropout)])
                if res in cfg.attn_resolutions:
                    block.append(AttnBlock(out_ch))
                self.down_blocks.append(block)
                cur = out_ch
                chs.append(cur)
            if i != len(mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(cur)]))
                chs.append(cur)
                res //= 2

        self.mid = nn.ModuleList([
            ResBlock(cur, cur, t_dim, cfg.dropout),
            AttnBlock(cur),
            ResBlock(cur, cur, t_dim, cfg.dropout),
        ])

        self.up_blocks = nn.ModuleList()
        for i, m in enumerate(reversed(mults)):
            out_ch = C * m
            for _ in range(nb + 1):
                skip_ch = chs.pop()
                block = nn.ModuleList([ResBlock(cur + skip_ch, out_ch, t_dim, cfg.dropout)])
                if res in cfg.attn_resolutions:
                    block.append(AttnBlock(out_ch))
                self.up_blocks.append(block)
                cur = out_ch
            if i != len(mults) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample(cur)]))
                res *= 2

        self.out_norm = nn.GroupNorm(min(32, cur), cur)
        self.out_conv = nn.Conv2d(cur, cfg.in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.in_conv(x)
        skips = [h]
        for block in self.down_blocks:
            first = block[0]
            if isinstance(first, Downsample):
                h = first(h)
                skips.append(h)
            else:
                h = first(h, t_emb)
                for layer in list(block)[1:]:
                    h = layer(h)
                skips.append(h)
        for layer in self.mid:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
        for block in self.up_blocks:
            first = block[0]
            if isinstance(first, Upsample):
                h = first(h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = first(h, t_emb)
                for layer in list(block)[1:]:
                    h = layer(h)
        return self.out_conv(F.silu(self.out_norm(h)))
