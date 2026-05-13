# Root-Cause Report — Black Outputs From the Trained Diffusion Model

## TL;DR

Training was perfect (loss 1.19 → 0.05). The outputs were unusable because of
**two compounding bugs in the sampling pipeline**:

1. **eps-formulation sampler with no x₀ clamp.** With the cosine schedule's tail β≈0.999 and √α_t≈0.0316, the eps-based posterior mean has a prefactor 1/√α_t ≈ 31. Any ε-prediction error (RMSE ≈ 0.22 even at loss 0.05) is amplified ~31× per step and compounded over 1000 steps. Final x[std] reached **450** in fp32 (verified locally).
2. **fp16 autocast inside sampling.** Independent of (1), the U-Net's internal activations at the high-t end exceed fp16's ±65 504 dynamic range → NaN → clamp → denorm to 0.5 → **uniform dark gray**.

In v4, both bugs fired and produced uniform-dark output. In v5, the (separately-prepared) Karras EMA fix landed but the same sampling bugs remained, producing **colorful noise** instead (fp32 path was no longer NaN-ed because the trained EMA happened to keep x slightly bounded for the first few steps, but the eps amplification still won out).

The fix is the **x₀-clamped posterior** formulation (Ho et al.'s reference DDPM) plus **fp32 sampling**. Verified locally against the v4-trained checkpoint — produces real butterflies.

## Local verification (decisive evidence)

Same `ema_only.pt` checkpoint, four sampler configurations:

| # | Sampler | Precision | Final x[std] | Visual result |
|---|---|---|---:|---|
| A | eps-formula DDPM | fp32 | **450** | colorful noise (the v5 phenomenon) |
| B | **x₀-clamped DDPM** | **fp32** | **0.56** | **real butterflies** ✓ |
| C | x₀-clamped DDPM | fp16 autocast | NaN | uniform gray |
| D | x₀-clamped DDIM (50) | fp16 autocast | NaN | uniform gray |

Per-step trace of (B):
```
t= 800  x[std]=0.976  x0[mean]=+0.093  x0[std]=0.513
t= 600  x[std]=0.891  x0[mean]=+0.095  x0[std]=0.540
t= 400  x[std]=0.755  x0[mean]=+0.097  x0[std]=0.552
t= 200  x[std]=0.625  x0[mean]=+0.097  x0[std]=0.560
t=   0  x[std]=0.565  x0[mean]=+0.099  x0[std]=0.565
```

Values stay bounded for the full 1000-step reverse. The clamp on the predicted x̂₀ at each step is what keeps things stable.

## Why my initial EMA-decay hypothesis was incomplete

Local debug of the v4 EMA weights showed:
- Whole-model param std = 2.25e-2 (identical to fresh random init)
- Top per-layer diffs ~0.13 — but on a 0.10-magnitude parameter that's roughly two independent random draws.

I concluded EMA was untrained. That was **partially** correct — the EMA was indeed barely tracking the live model — but the v5 verification showed the **raw** model also produced colorful noise. The sampling pipeline was hiding the trained model regardless of which weights it used. The Karras-warmup EMA fix in v5 was a reasonable improvement, but not the root cause.

## Math: why the eps-formula amplifies errors

Two algebraically equivalent forms of the DDPM posterior mean exist:

A. eps-form
```
μ_t = (1/√α_t) · (x_t − β_t/√(1−ᾱ_t) · ε_θ)
```

B. x₀-form (Ho et al. equation 7)
```
x̂₀ = (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t           [optionally clamp(-1, 1)]
μ_t = (√ᾱ_{t-1} · β_t / (1 − ᾱ_t)) · x̂₀
    + (√α_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t)) · x_t
```

When `ε_θ = ε_true`, both forms produce identical `μ_t`. With prediction error `Δ = ε_θ − ε_true`:

- Form A amplifies Δ by `(1/√α_t) · (β_t/√(1−ᾱ_t))`. At t=999 with cosine schedule: ≈ 31 × 1 = **31×**.
- Form B amplifies Δ by `(√ᾱ_{t-1} · β_t / (1 − ᾱ_t)) · (1/√ᾱ_t) · √(1−ᾱ_t)`. At t=999: ≈ **1×**.

Furthermore the clamp on x̂₀ in form B caps the influence of *any* outlier prediction. Form A has no such safety.

Loss 0.05 → RMSE(ε) ≈ 0.22 → amplified to 7 in form A at the first step → compounded over 1000 steps → x[std] reaches O(100s).

## Why fp16 NaNs

In `autocast(fp16)`:
- U-Net intermediate activations (especially mid-block attention) can exceed fp16 max 65 504 at high t
- Some op overflows → inf
- Subsequent `inf - inf` or `inf / inf` → NaN
- NaN propagates through the rest of the forward
- `clamp(-1, 1)` does not normalize NaN — NaN survives
- `denorm = (NaN + 1) / 2 = NaN`
- `imshow` renders NaN as the mid-colormap color → mid gray

bf16 wouldn't have this problem (range ±3.4e38), but Kaggle's P100/T4 are SM 6.0/7.5 (no native bf16). The only safe option for sampling on these cards is fp32.

## v6 fixes (pushed)

1. **Both samplers use the x₀-clamped formulation:**
   ```python
   x0 = ((x - smac * eps) / sac).clamp(-1, 1)
   mean = coef_x0 * x0 + coef_x * x
   ```
2. **Sampling runs in fp32 (no autocast).** Training still uses fp16 autocast — only the reverse process is fp32. Cost: sampling is ~2× slower; on P100 this is still ~3 min for all post-training viz.
3. **The DDIM path also re-derives ε from clamped x̂₀** to keep the update self-consistent.

Patches that landed earlier and are kept:
- Conditional torch 2.4.1 install for sm_60 (v2)
- `float(a)` cast in interpolate slerp (v3)
- Karras-style EMA warmup (v5)
- Raw-model sample-grid fallback (v5)

## Cell-by-cell fix locations

| Notebook section | Change |
|---|---|
| **Samplers** | Replaced `def ddpm_sample` and `def ddim_sample` with x₀-clamped versions; removed `autocast(...)` wrappers around `net(x, t)` |
| **Latent Interpolation** | `with autocast(...): eps = ema.shadow(x, t)` → `eps = ema.shadow(x, t).float()` |
| **Forward vs Reconstructed** | same autocast strip |
| **Training loop** | unchanged — fp16 autocast still used for forward+backward during training |

## What was *not* the cause (with proof)

| Hypothesis | Verdict | Proof |
|---|---|---|
| Denormalization bug | NO | `denorm(x) = (x.clamp(-1,1)+1)/2` — dataset_grid.png renders correctly |
| `.zero_()` accident | NO | weight stats show distribution |
| GroupNorm / SiLU issue | NO | training loss converged cleanly |
| Wrong α_t indexing | NO | manual algebra reproduces Ho et al. equations |
| `np.linspace` dtype mismatch | NO | already fixed in v3; v4 ran end-to-end |
| Time-embedding bug | NO | sin/cos work; predictions vary with t in fp32 traces |
| Channels-last layout | NO | training & forward shapes are correct |
| EMA decay too high | PARTIAL | EMA was indeed undertrained, but raw model also failed → not root cause |

## Verification plan for v6

Locally I produced an 8×8 grid via the new sampler on the v4-trained EMA → real butterflies. v6 retrains with all fixes; expected results:
- `assets/samples_grid.png` — real butterflies (EMA path)
- `assets/samples_grid_raw.png` — real butterflies (raw-model fallback)
- `assets/interpolation.png` — smooth butterfly→butterfly slerp
- `assets/feature_maps.png` — meaningful filter activations
- `outputs/gifs/denoising.gif` — visible noise → butterfly progression
