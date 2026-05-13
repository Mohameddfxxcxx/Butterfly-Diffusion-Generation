# Running on Kaggle

Single-file, GPU-only training notebook for the Butterfly Diffusion model.

## File

```
butterfly_diffusion_kaggle.ipynb
```

## Option A — Upload via Kaggle UI (recommended)

1. Open https://www.kaggle.com/code → **+ New Notebook**
2. **File → Import Notebook → Upload** → choose `butterfly_diffusion_kaggle.ipynb`
3. In the right sidebar:
   - **Accelerator** → `GPU T4 x2` (or `GPU P100`)
   - **Internet** → `On` *(required — pulls the HF dataset)*
   - **Persistence** → optional, only if you want files between sessions
4. Click **Run All**
5. When done, expand the **Output** panel and download:
   - `butterfly_diffusion_deliverables.zip` — full bundle
   - Or pick individual files: `denoising.gif`, `checkpoints/best.pt`, `assets/*.png`

## Option B — Upload via Kaggle CLI

```bash
pip install kaggle
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

cd kaggle_version
kaggle kernels init -p .
# Edit kernel-metadata.json:
#   "id": "<your-username>/butterfly-diffusion",
#   "code_file": "butterfly_diffusion_kaggle.ipynb",
#   "kernel_type": "notebook",
#   "is_private": false,
#   "enable_gpu": true,
#   "enable_internet": true,
#   "language": "python"
kaggle kernels push -p .
```

The kernel will run in the cloud; check status with:

```bash
kaggle kernels status <your-username>/butterfly-diffusion
kaggle kernels output <your-username>/butterfly-diffusion -p ./downloaded
```

## Modes

The Config cell exposes:

```python
MODE = "full"   # "smoke" | "full"
```

| Mode    | Epochs | DDIM steps | Wall-clock (T4) | Use case                       |
|---------|:------:|:---------:|:---------------:|--------------------------------|
| `smoke` | 2      | 20        | ~2–3 min        | sanity-check before full run   |
| `full`  | 80     | 50        | ~30–50 min      | submission-quality samples     |

Set `MODE = "smoke"` for a fast verification pass, then switch to `"full"`.

## Resume support

Each epoch writes `checkpoints/last.pt`. If the Kaggle session times out (9-hour limit) and you re-run the notebook with `cfg.resume = True` (default), training picks up at the next epoch with optimizer/EMA/LR state intact.

## Outputs (all under `/kaggle/working/`)

```
/kaggle/working/
├── butterfly_diffusion_deliverables.zip     # full bundle (download this)
├── denoising.gif                            # top-level for quick preview
├── checkpoints/
│   ├── last.pt    (resume point)
│   ├── best.pt    (lowest training loss)
│   └── final.pt   (end of training)
├── assets/                                  # README-ready PNGs
│   ├── banner.png, dashboard.png
│   ├── dataset_grid.png, samples_grid.png
│   ├── forward_process.png, schedule.png
│   ├── loss_curve.png, feature_maps.png
│   ├── interpolation.png, forward_vs_recon.png
│   ├── ddpm_samples.png, ddim_samples.png
├── outputs/
│   ├── samples/epoch_XXX.png                # per-epoch EMA grids
│   ├── trajectories/trajectory_grid.png
│   ├── gifs/denoising.gif
│   └── manifest.json                        # run summary
└── generated_samples/
    └── butterfly_000.png … butterfly_063.png
```

## GPU diagnostics

The first runnable cell prints `nvidia-smi`, torch/CUDA versions, VRAM, and SM count. The Imports cell auto-selects **bfloat16** on Ampere+ (capability ≥ 8.0) or **float16** elsewhere — so the notebook behaves correctly whether Kaggle assigns you T4, P100, or A100.

## Troubleshooting

- **"Accelerator: None"** in diagnostics → switch the sidebar accelerator to GPU and click **Run All** again.
- **HF dataset 401** → make sure **Internet** is `On` in the sidebar.
- **OOM mid-training** → lower `cfg.batch_size` to 32 or 16, or drop `cfg.base_channels` to 48.
- **Session timed out** → just re-run; resume picks up from `last.pt`.
