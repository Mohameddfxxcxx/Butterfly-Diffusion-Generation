#!/usr/bin/env bash
# Butterfly Diffusion — Streamlit launcher (macOS / Linux)
set -e
cd "$(dirname "$0")"

if [ ! -f "checkpoints/ema_only.pt" ]; then
    echo
    echo "[!] checkpoints/ema_only.pt not found."
    echo "    Download it from the Kaggle release (see reports/production_report.md)"
    echo "    and place it in the checkpoints/ folder, then re-run this script."
    echo
    exit 1
fi

python -m streamlit run app/streamlit_app.py --browser.gatherUsageStats false
