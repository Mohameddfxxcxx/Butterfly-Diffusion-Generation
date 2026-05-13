"""Auto-download the trained checkpoint from the GitHub Release.

Used by the Streamlit Cloud deployment, where the repo is cloned without
the 252 MB checkpoint. On first run the asset is fetched to
``checkpoints/ema_only.pt`` and cached for subsequent boots.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import urllib.request


RELEASE_URL = (
    "https://github.com/Mohameddfxxcxx/Butterfly-Diffusion-Generation"
    "/releases/download/v1.0.0/ema_only.pt"
)
EXPECTED_SIZE = 252_724_580  # bytes


def ensure_checkpoint(target: Optional[Path] = None, progress_callback=None) -> Path:
    """Return the local path to ``ema_only.pt``, downloading if missing."""
    if target is None:
        repo_root = Path(__file__).resolve().parents[2]
        target = repo_root / "checkpoints" / "ema_only.pt"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size >= EXPECTED_SIZE * 0.99:
        return target

    tmp = target.with_suffix(".pt.part")
    if tmp.exists():
        tmp.unlink()

    req = urllib.request.Request(
        RELEASE_URL,
        headers={"User-Agent": "butterfly-diffusion-bootstrap/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", EXPECTED_SIZE))
        downloaded = 0
        chunk = 1 << 16
        with open(tmp, "wb") as out:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                if progress_callback is not None:
                    progress_callback(downloaded / total, downloaded, total)

    tmp.rename(target)
    return target


if __name__ == "__main__":
    def _cli(fraction, dl, total):
        bar = int(fraction * 30)
        print(f"\r[{'#' * bar}{'.' * (30 - bar)}] {dl/1e6:6.1f}/{total/1e6:.1f} MB",
              end="", flush=True)

    p = ensure_checkpoint(progress_callback=_cli)
    print(f"\nReady -> {p}  ({p.stat().st_size/1e6:.2f} MB)")
