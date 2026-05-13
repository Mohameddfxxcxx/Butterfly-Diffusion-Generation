# Contributing

Thanks for your interest in Butterfly Diffusion. This is a research project, but PRs and issues are welcome.

## Setup

```bash
git clone https://github.com/Mohameddfxxcxx/Butterfly-Diffusion-Generation.git
cd Butterfly-Diffusion-Generation
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .[demo,dev]
```

## Where things live

| Path | What |
|---|---|
| `src/` | The PyTorch package — model, scheduler, sampler, training loop |
| `app/streamlit_app.py` | Interactive demo |
| `notebooks/` | Self-contained Kaggle/academic notebooks |
| `reports/` | Run reports, root-cause analysis, Kaggle timeline |
| `assets/` | Rendered figures used in the README |
| `checkpoints/` | Slim `ema_only.pt` lives here (gitignored due to size) |

## Reporting issues

Please include:

- OS, Python, PyTorch, and CUDA version (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`)
- For training problems: a copy of the loss curve and the last 20 lines of stdout
- For sampling problems: the seed and the checkpoint file size

## Code conventions

- Match existing style: prefer dataclasses for configuration, type hints on public functions, no comments on obvious code
- Don't reformat unrelated files in a PR
- If you change the model architecture, bump `__version__` in `src/__init__.py` and update `reports/production_report.md`

## Numerical fix lineage

If you change anything in the sampler, please read `reports/root_cause_report.md` first — the x₀-clamped fp32 path is load-bearing and was painful to find.
