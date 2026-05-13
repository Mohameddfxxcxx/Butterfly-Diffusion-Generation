# Kaggle Runtime Report — Final

Confirmed execution on Kaggle GPU. This report captures the actual runtime
metadata, validated component matrix, and download manifest.

## Environment (measured on Kaggle, v3)

| Field | Value |
|---|---|
| Device name | **Tesla P100-PCIE-16GB** |
| CUDA capability | 6.0 (Pascal) |
| VRAM total | 17.06 GB |
| SM count | 56 |
| Torch version | **2.4.1+cu121** (auto-installed because sm_60 needs old torch) |
| CUDA build | 12.1 |
| AMP dtype (auto) | `torch.float16` (cap < 8.0) |
| Python | 3.12 (Kaggle default) |
| Internet | enabled — HF dataset pulled successfully |

The conditional install path is now proven: T4 (sm_75) and A100 (sm_80) will run on Kaggle's default PyTorch unchanged; P100 (sm_60) triggers the `torch 2.4.1` fallback.

## Notebook upload status

| Item | State |
|---|---|
| Kernel slug | `mohamedelsadek44/butterfly-diffusion` |
| Final version | v3 |
| Status | `KernelWorkerStatus.COMPLETE` |
| Files produced | 86 |
| Errors in log | 0 |
| GPU enabled | yes |
| Internet enabled | yes |
| Live URL | https://www.kaggle.com/code/mohamedelsadek44/butterfly-diffusion |

## CUDA / AMP compatibility — verified on Kaggle

- `torch.cuda.is_available()` → True
- `torch.backends.cudnn.benchmark = True` — set
- `torch.set_float32_matmul_precision("high")` — set
- Channels-last memory format — applied to model + tensors
- `autocast(dtype=AMP_DTYPE)` — fp16 selected (cap 6.0), GradScaler enabled
- Capability detection `_cap[0] >= 8 → bf16 else fp16` — evaluated correctly

## Pipeline validation matrix (live Kaggle evidence)

| Component | Pass | Evidence |
|---|:---:|---|
| Conditional torch reinstall | yes | log: "Pre-Volta GPU (sm_60); installing torch 2.4.1…" |
| GPU diagnostics cell | yes | full nvidia-smi + props printed |
| HuggingFace dataset load | yes | 1000 images, 15 batches |
| DataLoader (channels-last) | yes | training proceeded without dtype/layout errors |
| Cosine β schedule | yes | `schedule.png` rendered |
| Forward diffusion | yes | `forward_process.png` rendered |
| U-Net build + AMP forward | yes | 63.15 M params, no runtime errors |
| EMA shadow update | yes | tracked through 2 epochs |
| Training loop | yes | 30 steps total, loss 1.087 → 0.615 |
| Per-epoch sample save | yes | `outputs/samples/epoch_001.png`, `epoch_002.png` |
| Loss + LR plot | yes | `loss_curve.png` rendered |
| DDPM sampling (full reverse) | yes | `ddpm_samples.png` rendered |
| DDIM sampling | yes | `ddim_samples.png` rendered |
| Reverse trajectory grid | yes | `trajectories/trajectory_grid.png` |
| Sample grid 8×8 | yes | `samples_grid.png` rendered |
| 64 individual butterfly PNGs | yes | `generated_samples/butterfly_000…063.png` |
| **Latent interpolation** | yes (fixed in v3) | `interpolation.png` |
| Forward-vs-reconstructed | yes | `forward_vs_recon.png` |
| Feature-map hooks | yes | `feature_maps.png` |
| Denoising GIF export | yes | `denoising.gif` (12 KB) |
| Training dashboard | yes | `dashboard.png` — 5 panels populated |
| README banner export | yes | `banner.png` |
| Manifest JSON | yes | `outputs/manifest.json` |
| ZIP packaging | yes | `butterfly_diffusion_deliverables.zip` (2 800 MB) |

## Loss + timing (smoke run)

| Epoch | Loss | LR | GPU Mem | Time |
|:-----:|:----:|:--:|:-------:|:----:|
| 1 | 1.0869 | 6.0e-05 | 1313 MB | 25.9 s |
| 2 | 0.6151 | 1.2e-04 | 1314 MB | 13.0 s |
| **Total** | | | | **0.84 min** |

Loss dropped 43 % in two epochs (LR was still in warmup — convergence will accelerate). VRAM usage 1.3 GB / 17 GB available — plenty of headroom for the full 80-epoch run.

## Generated output paths (on Kaggle)

```
/kaggle/working/
├── butterfly_diffusion_deliverables.zip
├── denoising.gif
├── checkpoints/
│   ├── last.pt   resume point
│   ├── best.pt   lowest loss
│   └── final.pt  end of run
├── assets/                       12 PNGs (README-ready)
├── outputs/
│   ├── samples/epoch_NNN.png    per-epoch EMA grids
│   ├── trajectories/trajectory_grid.png
│   ├── gifs/denoising.gif
│   └── manifest.json
└── generated_samples/
    └── butterfly_000.png … butterfly_063.png
```

## Expected runtime — full mode

Extrapolating from smoke timings on this **P100** allocation:

| Phase | Smoke (run) | Full (estimate) |
|---|---:|---:|
| Diagnostics + install + (P100 torch swap) | ~140 s | ~140 s |
| Dataset download | ~5 s | ~5 s |
| Training (15 batches × epochs) | 50 s @ 2 ep | **~30 min @ 80 ep** |
| DDPM (1000-step sampling) | ~25 s × 8 imgs | ~50 s × 8 imgs |
| DDIM (50 steps) | <5 s | ~10 s × 64 imgs |
| Visualizations + GIF + dashboard | ~30 s | ~3 min |
| ZIP packaging | ~5 s | ~25 s |
| **Total** | **~3 min** | **~35–45 min** |

On a T4 allocation expect ~25 % faster than the above.

## Validation conclusions

End-to-end execution on real Kaggle hardware: confirmed. Every component of the production pipeline — including the parts that aren't exercised locally (P100 path, Kaggle filesystem, full HF dataset hit, papermill notebook execution, per-cell output capture) — has been observed to work.

Outstanding (cosmetic, non-blocking):
- **Checkpoint size**: 1 GB each because optimizer + EMA state are stored in fp32. For inference-only distribution, an `ema_only.pt` (~252 MB) is recommended — easy to add as a final cell.
