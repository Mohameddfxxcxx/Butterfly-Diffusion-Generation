"""Butterfly Diffusion - Streamlit demo.

Loads the slim EMA checkpoint and exposes the full inference pipeline
(DDPM, DDIM, trajectory, interpolation, reconstruction, feature maps)
through a dark-themed multi-tab UI with ZIP/GIF/PNG downloads.

Run:
    streamlit run app/streamlit_app.py
"""
import io
import sys
import time
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
import torch
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid

# Make src/ importable when launched from anywhere
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.inference import ButterflyDiffusionPipeline, auto_device  # noqa: E402
from src.utils.visualize import denorm                              # noqa: E402

# ---------------------------------------------------------------------------
# Page / theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Butterfly Diffusion",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp { background: linear-gradient(180deg,#0e1117 0%, #15182b 100%); }
    h1, h2, h3, h4 { color: #ffe066 !important; }
    .stSidebar { background-color: #161a2c; }
    [data-testid="stMetricValue"] { color: #7df9ff; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.6rem; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1c2138; color: #cfcfcf;
        border-radius: 8px 8px 0 0; padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6ec7 !important; color: #0e1117 !important;
    }
    .stButton button {
        background: linear-gradient(90deg,#ff6ec7,#7df9ff);
        color: #0e1117; border: 0; font-weight: 700;
    }
    .stDownloadButton button { background-color: #2a2f4a; color: #ffe066; border: 0; }
    .stProgress > div > div > div > div { background-color: #ff6ec7; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
    "savefig.facecolor": "#0e1117", "axes.labelcolor": "#e6e6e6",
    "xtick.color": "#cfcfcf", "ytick.color": "#cfcfcf",
    "axes.titlecolor": "#ffffff", "text.color": "#e6e6e6",
})

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🦋 Butterfly Diffusion — Live Demo</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#a0a8c0;'>"
    "A DDPM/DDIM playground powered by a U-Net trained on the Smithsonian Butterflies subset. "
    "Numerically stable sampling (x₀-clamped posterior, fp32). CUDA accelerated when available."
    "</p>", unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Pipeline loader (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading checkpoint and warming up the U-Net…")
def load_pipeline(ckpt_path: str, device_pref: str):
    return ButterflyDiffusionPipeline(ckpt_path, device=device_pref)


def discover_checkpoints() -> List[Path]:
    return sorted((REPO / "checkpoints").glob("*.pt"))


# ---------------------------------------------------------------------------
# Sidebar — settings + diagnostics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Configuration")

    ckpt_list = [str(p) for p in discover_checkpoints()]
    if not ckpt_list:
        st.warning(
            "No `.pt` files found in `checkpoints/`. "
            "Drop `ema_only.pt` (252 MB) from the Kaggle output there."
        )
        st.stop()
    ckpt_pretty = [Path(p).name for p in ckpt_list]
    idx = st.selectbox("Checkpoint", range(len(ckpt_list)), format_func=lambda i: ckpt_pretty[i])
    ckpt_path = ckpt_list[idx]

    device_pref = st.radio(
        "Device",
        ["auto", "cuda", "cpu"],
        horizontal=True,
        help="Auto picks CUDA if a GPU is available; CPU works but is ~50× slower for full DDPM.",
    )
    device = auto_device(device_pref if device_pref != "auto" else None)

    st.subheader("🎲 Sampler")
    sampler = st.radio("Algorithm", ["ddim", "ddpm"], horizontal=True)
    if sampler == "ddim":
        steps = st.slider("DDIM steps", 10, 200, 50, step=10)
    else:
        steps = st.slider("DDPM steps (fixed)", 1000, 1000, 1000, disabled=True)

    n_images = st.slider("Number of samples", 1, 64, 16, step=1)
    nrow = st.slider("Grid columns", 1, 8, 4)
    seed_in = st.number_input("Seed (-1 = random each run)", value=42, step=1)
    seed = None if seed_in == -1 else int(seed_in)

    st.divider()
    st.subheader("🩺 Diagnostics")
    try:
        pipe = load_pipeline(ckpt_path, device_pref if device_pref != "auto" else None)
        st.metric("Device", str(pipe.device).upper())
        st.metric("Params", f"{pipe.ckpt_meta['params_M']:.2f} M")
        st.metric("Checkpoint", f"{pipe.ckpt_meta['size_mb']:.0f} MB ({pipe.ckpt_meta['weights']})")
        if pipe.ckpt_meta.get("final_loss") is not None:
            st.metric("Training loss", f"{pipe.ckpt_meta['final_loss']:.4f}")
        if pipe.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            st.caption(
                f"GPU: {props.name}\n"
                f"VRAM total: {props.total_memory/1e9:.1f} GB\n"
                f"CUDA: {torch.version.cuda}  ·  Torch: {torch.__version__}\n"
                f"Compute capability: {props.major}.{props.minor}"
            )
        else:
            st.caption(f"CPU mode  ·  Torch: {torch.__version__}")
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_gen, tab_traj, tab_interp, tab_recon, tab_feat = st.tabs([
    "🦋 Generate",
    "🌫️ Trajectory",
    "🔀 Interpolation",
    "🧩 Reconstruction",
    "🔬 Feature Maps",
])

# ---------------- GENERATE ----------------
with tab_gen:
    col1, col2 = st.columns([3, 2])
    with col1:
        go = st.button("🎨 Generate", type="primary", use_container_width=True)
    with col2:
        st.info(f"Mode: **{sampler.upper()}** · steps: **{steps}** · "
                f"images: **{n_images}** · seed: **{seed if seed is not None else 'random'}**",
                icon="ℹ️")

    if go:
        prog = st.progress(0.0, text=f"Running {sampler.upper()} sampler …")
        t0 = time.time()
        try:
            imgs, traj = pipe.generate(
                n=n_images, sampler=sampler, steps=int(steps), seed=seed,
                return_trajectory=False,
            )
            prog.progress(1.0, text=f"Done in {time.time()-t0:.1f}s")
        except RuntimeError as e:
            prog.empty(); st.error(f"Generation failed: {e}")
            st.stop()

        pil = pipe.to_pil_grid(imgs, nrow=nrow)
        st.subheader("Generated Butterflies")
        st.image(pil, use_container_width=True)

        buf = io.BytesIO(); pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download grid PNG", buf.getvalue(),
            file_name="butterflies_grid.png", mime="image/png",
            use_container_width=True,
        )

        # Per-image downloads + ZIP
        with st.expander(f"Individual images ({n_images})", expanded=False):
            cols = st.columns(min(nrow, 4))
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i in range(imgs.size(0)):
                    arr = (denorm(imgs[i].float().cpu()).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pil_i = Image.fromarray(arr)
                    b = io.BytesIO(); pil_i.save(b, format="PNG")
                    zf.writestr(f"butterfly_{i:03d}.png", b.getvalue())
                    if i < 8:
                        with cols[i % len(cols)]:
                            st.image(pil_i, caption=f"#{i:03d}", use_container_width=True)
            st.download_button(
                "⬇️ Download all (ZIP)", zip_buf.getvalue(),
                file_name=f"butterflies_{n_images}.zip", mime="application/zip",
                use_container_width=True,
            )

# ---------------- TRAJECTORY ----------------
with tab_traj:
    st.markdown(
        "Watch the reverse diffusion process unfold. DDPM only — DDIM doesn't expose "
        "every intermediate state."
    )
    n_traj = st.slider("Images in trajectory", 1, 8, 4, key="ntraj")
    go_traj = st.button("Run full DDPM with trajectory", type="primary")
    if go_traj:
        prog = st.progress(0.0, text="Running 1000-step DDPM (no autocast)…")
        try:
            imgs, traj = pipe.generate(n=n_traj, sampler="ddpm", seed=seed, return_trajectory=True)
            prog.progress(1.0, text="Done")
        except RuntimeError as e:
            prog.empty(); st.error(str(e)); st.stop()

        # GIF
        frames = []
        for snap in traj:
            g = make_grid(denorm(snap.float()), nrow=n_traj, padding=2)
            arr = (g.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            frames.append(arr)
        gif = io.BytesIO()
        imageio.mimsave(gif, frames, duration=0.18, format="GIF", loop=0)
        st.subheader("Denoising trajectory (GIF)")
        st.image(gif.getvalue(), use_container_width=True)
        st.download_button(
            "⬇️ Download GIF", gif.getvalue(),
            file_name="denoising.gif", mime="image/gif",
            use_container_width=True,
        )

        # Strip view: snapshots stacked
        st.subheader("Trajectory snapshots")
        strip = torch.cat([t[:n_traj] for t in traj], dim=0)
        st.image(pipe.to_pil_grid(strip, nrow=n_traj), use_container_width=True)

# ---------------- INTERPOLATION ----------------
with tab_interp:
    st.markdown("Spherical linear interpolation (slerp) between two random latent codes, "
                "decoded with DDIM.")
    n_pairs = st.slider("Number of pairs", 1, 6, 4)
    steps_pair = st.slider("Frames per pair", 4, 16, 8)
    if st.button("Generate interpolation", type="primary"):
        with st.spinner("Slerping in latent space…"):
            grid = pipe.interpolate(n_pairs=n_pairs, steps_per_pair=steps_pair,
                                    ddim_steps=int(steps))
        pil = pipe.to_pil_grid(grid, nrow=steps_pair)
        st.image(pil, use_container_width=True)
        buf = io.BytesIO(); pil.save(buf, format="PNG")
        st.download_button("⬇️ Download interpolation", buf.getvalue(),
                           file_name="interpolation.png", mime="image/png",
                           use_container_width=True)

# ---------------- RECONSTRUCTION ----------------
with tab_recon:
    st.markdown("Upload a butterfly image (or use the provided demo), noise it to a target t, "
                "and let the diffusion model reconstruct it.")
    t_val = st.slider("Noise level (t)", 50, 900, 400, step=50)
    upload = st.file_uploader("Image (PNG/JPG, will be cropped to 64×64)",
                              type=["png", "jpg", "jpeg"])
    if st.button("Reconstruct", type="primary"):
        from torchvision import transforms
        if upload is not None:
            img = Image.open(upload).convert("RGB")
        else:
            # Fallback: use the dataset_grid.png that ships with the repo if present
            sample = REPO / "assets" / "dataset_grid.png"
            if not sample.exists():
                st.error("No upload provided and no fallback image. Please upload one.")
                st.stop()
            img = Image.open(sample).convert("RGB").crop((0, 0, 64, 64))
        tfm = transforms.Compose([
            transforms.Resize(pipe.cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(pipe.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        real = tfm(img).unsqueeze(0)
        with st.spinner(f"Forward+reverse at t={t_val}…"):
            noisy, recon = pipe.reconstruct(real, t_val=t_val)
        panel = torch.cat([real.cpu(), noisy.cpu(), recon.cpu()], dim=0)
        st.image(pipe.to_pil_grid(panel, nrow=1),
                 caption="Top: real · Middle: noisy · Bottom: reconstructed",
                 use_container_width=False, width=240)

# ---------------- FEATURE MAPS ----------------
with tab_feat:
    st.markdown("Mid-block U-Net activations at a single noise level — what the model sees "
                "in its bottleneck.")
    t_val = st.slider("Timestep", 0, pipe.cfg.timesteps - 1, 500, step=10, key="ft_t")
    n_ch = st.slider("Channels to display", 4, 16, 8)
    if st.button("Pull features", type="primary"):
        with st.spinner("Running U-Net forward and capturing mid-block…"):
            fm = pipe.feature_maps(t_val=t_val, n_channels=n_ch)
        fig, axes = plt.subplots(1, n_ch, figsize=(2 * n_ch, 2.4))
        if n_ch == 1: axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(fm[i].numpy(), cmap="magma"); ax.axis("off")
            ax.set_title(f"ch{i}", fontsize=10, color="white")
        fig.suptitle(f"Mid-block feature maps  (t={t_val})", color="white", fontsize=14)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:#666;font-size:0.9rem;'>"
    "Numerically stable sampler · x₀-clamped posterior · fp32 inference · "
    "Karras-style EMA · cosine β-schedule"
    "</p>", unsafe_allow_html=True,
)
