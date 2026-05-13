@echo off
REM Butterfly Diffusion — Streamlit launcher (Windows)
REM Usage: double-click or run from a terminal in the repo root.

setlocal
cd /d "%~dp0"

if not exist "checkpoints\ema_only.pt" (
    echo.
    echo [!] checkpoints\ema_only.pt not found.
    echo     Download it from the Kaggle release ^(see reports\production_report.md^)
    echo     and place it in the checkpoints\ folder, then re-run this script.
    echo.
    pause
    exit /b 1
)

python -m streamlit run app\streamlit_app.py --browser.gatherUsageStats false
endlocal
