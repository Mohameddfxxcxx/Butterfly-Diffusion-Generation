# Production Training Report — Butterfly Diffusion on Kaggle

Final run: **kernel version 7**, all visualizations regenerated with the
x₀-clamped fp32 sampler everywhere (main DDPM/DDIM, latent interpolation, and
forward-vs-reconstructed inline samplers).

## Result

| Field | Value |
|---|---|
| Status | **SUCCESS** — recognizable, high-quality butterflies |
| Kaggle URL | https://www.kaggle.com/code/mohamedelsadek44/butterfly-diffusion |
| GPU | Tesla P100-PCIE-16GB (sm_60, 16 GB) |
| Torch | 2.4.1+cu121 (auto-installed for sm_60) |
| AMP | fp16 for training, fp32 for sampling |
| Mode | `full` |
| Epochs | 80 |
| U-Net params | 63.15 M |
| Final loss | 0.0534 |
| Best loss | 0.0508 |
| Total training time | 17.2 min |
| Wall-clock incl. visualization + ZIP | ~38 min |
| Output files | 102 |

## Visual evidence

| Asset | What it shows |
|---|---|
| `assets/samples_grid.png` | 64-image grid of clear butterflies from the EMA model via DDIM |
| `assets/samples_grid_raw.png` | Independent verification grid from the raw (non-EMA) model |
| `assets/dashboard.png` | 5-panel dashboard, EMA Samples panel populated with butterflies |
| `outputs/samples/epoch_080.png` | Final-epoch EMA samples — distinct individual butterflies |
| `outputs/samples/epoch_005.png` | Early epoch — colorful noise, model still in warmup |
| `outputs/trajectories/trajectory_grid.png` | Clean noise→butterfly reverse-diffusion progression |
| `outputs/gifs/denoising.gif` | Animated reverse diffusion (440 KB) |
| `assets/feature_maps.png` | Structured mid-block activations (not uniform) |
| `assets/dataset_grid.png` | Real-data reference |
| `assets/forward_process.png` | Forward diffusion strip |
| `assets/loss_curve.png` | Clean exponential descent 1.19 → 0.05 |
| `assets/schedule.png` | β_t and ᾱ_t plots |

## Convergence

| Metric | First epoch | Last epoch |
|---|---:|---:|
| Loss | 1.190 | 0.053 |
| Epoch time | 25.9 s | 13.0 s |
| GPU mem | 1313 MB | 1309 MB |
| LR | 6.0e-5 | 0 (cosine to 0) |

- **Smooth exponential descent**, no NaN events, no instability
- LR warmup → peak ~2e-4 around step 500 → cosine decay to 0 at end
- VRAM stable at ~1.3 GB on a 16 GB card → plenty of headroom for larger images / batch
- ε-prediction RMSE ≈ √0.05 ≈ 0.22 — the model explains ~95 % of noise variance

## Final checkpoint sizes

| File | Size | Purpose |
|---|---:|---|
| `checkpoints/best.pt` | 1010 MB | model + EMA + AdamW state — for resume |
| `checkpoints/final.pt` | 1010 MB | end of run, same format |
| `checkpoints/last.pt` | 1010 MB | resume snapshot |
| `checkpoints/ema_only.pt` | **252.72 MB** | **inference-only**, drop-in for Streamlit |
| `butterfly_diffusion_deliverables.zip` | ~3 GB | full bundle |

## What `ema_only.pt` contains

```json
{
  "ema": <state_dict>,
  "config": <full asdict(Config)>,
  "params_M": 63.15,
  "final_loss": 0.0534
}
```

Verified locally: loading `ema_only.pt` + running the new x₀-clamped fp32 sampler
produces high-quality butterflies on a GTX 1650 Ti. The Streamlit app in
`github_version/app/streamlit_app.py` reads `ckpt["config"]` and prefers
`ckpt["ema"]` over `ckpt["model"]` — so the slim checkpoint drops in directly.

## Bug timeline (preserved for post-mortem)

| v# | Outcome | Fix landed in this version |
|----|---------|----------------------------|
| 1  | error   | — (Kaggle assigned P100 sm_60, default torch unsupported) |
| 2  | error   | conditional `torch 2.4.1` install for sm_60 |
| 3  | smoke OK| float32 cast in slerp coefficient |
| 4  | full ran, output uniform-dark | training healthy; revealed sampler instability + fp16-NaN |
| 5  | full ran, output colorful-noise | Karras EMA warmup + raw-grid fallback (insufficient alone) |
| **6**  | **full SUCCESS** | x₀-clamped DDPM + DDIM, fp32 sampling |
| **7**  | **regenerate SUCCESS** | x₀-clamped path also applied to `interpolate()` and `forward_vs_reconstructed()` inline samplers — both now produce real butterflies |

Detailed math + measured evidence is in `root_cause_report.md`.

## Sample quality summary

- Wing symmetry visible on most samples — model has learned the bilateral structure
- Color diversity present (yellow, blue, red, green, purple, brown) — captures dataset variety
- Some samples are sharper than others — expected with only 1 000 training images and 80 epochs
- The two grids (EMA + raw) agree on structure but differ in fine detail — confirms both heads are functioning

## How to reproduce locally

```powershell
# 1. Get the slim checkpoint from Kaggle (only 252 MB):
kaggle kernels output mohamedelsadek44/butterfly-diffusion -p .\out
# (or pull just `checkpoints/ema_only.pt` via the Python API)

# 2. Run the Streamlit demo:
cd D:\Butterfly-Diffusion-Generation\github_version
# Copy ema_only.pt into ./checkpoints/
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The Streamlit app auto-discovers `.pt` files, prefers EMA weights, and supports
both DDPM and DDIM sampling.

## Recommended next steps (optional)

1. **Higher-resolution training**: with VRAM at 1.3 GB / 16 GB, bumping `image_size` to 96 or 128 is well within budget.
2. **Longer schedule**: another 80 epochs would likely sharpen wing veins / fine textures.
3. **FID evaluation**: add `torchmetrics.image.fid.FID` to quantify quality vs the 1 000-image reference set.
4. **Hugging Face Hub publishing**: `huggingface_hub.push_to_hub` the slim checkpoint + README → public demo.

## Files in `kaggle_version/`

```
butterfly_diffusion_kaggle.ipynb      v7 push (latest)
kernel-metadata.json                  GPU on, internet on, mohamedelsadek44/butterfly-diffusion
KAGGLE_INSTRUCTIONS.md
execution_summary.md
kaggle_runtime_report.md
root_cause_report.md
production_report.md                  <- this file
_kaggle_logs/                         downloaded artifacts (~270 MB)
  ├── assets/                         12 PNGs
  ├── checkpoints/ema_only.pt         252 MB inference checkpoint
  ├── generated_samples/              64 individual fp16-saved butterflies
  ├── outputs/                        manifest, per-epoch grids, GIF, trajectory
  └── _logs/kernel.log
```
