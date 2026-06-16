#!/bin/bash

# ==========================================
# PALLETIZATION SYSTEM LAUNCHER (Jetson/Linux)
# ==========================================

# Use a readonly constant for the separator message
readonly SEPARATOR="--------------------------------------------------"

# 1. Navigate to the directory where this script is located
#    This ensures we can find main.py and other files independently of where the script was called from.
cd "$(dirname "$0")"

echo "$SEPARATOR"
echo "Starting Palletization System on $(date)"
echo "Working Directory: $(pwd)"
echo "$SEPARATOR"

# 2. Find the venv Python binary path (do NOT source/activate — sudo doesn't inherit the venv)
#    We invoke the venv Python binary directly so sudo retains access to venv packages.
if [[ -d "../venv" ]]; then
    VENV_PYTHON="../venv/bin/python"
    echo "[INFO] Found virtual environment: ../venv"
elif [[ -d "venv" ]]; then
    VENV_PYTHON="venv/bin/python"
    echo "[INFO] Found virtual environment: venv"
elif [[ -d "../.venv" ]]; then
    VENV_PYTHON="../.venv/bin/python"
    echo "[INFO] Found virtual environment: ../.venv"
else
    VENV_PYTHON=""
    echo "[WARNING] No virtual environment found. Will try system python with sudo." >&2
fi

# 3. Determine the final Python command (always run with sudo for Zebra printer access)
if [[ -n "$VENV_PYTHON" && -f "$VENV_PYTHON" ]]; then
    PY_CMD="$VENV_PYTHON"
    echo "[INFO] Using venv Python: $PY_CMD"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
    echo "[INFO] Venv binary not found, falling back to system: $(which python3)"
elif command -v python &> /dev/null; then
    PY_CMD="python"
    echo "[INFO] Venv binary not found, falling back to system: $(which python)"
else
    echo "[ERROR] Python is not installed or not found in PATH." >&2
    echo "Press Enter to exit..." >&2
    read
    exit 1
fi

$PY_CMD --version

echo ""
echo "[INFO] Launching main.py..."
echo "$SEPARATOR"

# 3b. Expose the user-site YOLO stack to root's Python.
#     torch/torchvision/ultralytics live in /home/icam-540/.local (torchvision is an .egg
#     that only activates via full site processing — PYTHONPATH is NOT enough). The app
#     runs as root via the sudo desktop launcher, so without this YOLO silently falls back
#     to slow full-frame Pyzbar. PYTHONUSERBASE triggers user-site processing without
#     changing HOME (Kivy still writes to /root).
export PYTHONUSERBASE=/home/icam-540/.local

# Preflight: show in the terminal whether YOLO/CUDA will actually be available this launch.
$PY_CMD -c "import torch, ultralytics; print('[PREFLIGHT] torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)" \
    || echo "[PREFLIGHT][WARN] YOLO stack NOT importable — detection will run WITHOUT YOLO" >&2

# 4. Run the application (script is already running with sudo from desktop launcher)
$PY_CMD main.py

# 5. Capture exit code and wait if there was an error
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "$SEPARATOR" >&2
    echo "[ERROR] Application exited with error code: $EXIT_CODE" >&2
    echo "        See output above for details." >&2
    echo "$SEPARATOR" >&2
    echo "Press Enter to close this window..." >&2
    read
else
    echo "$SEPARATOR"
    echo "[INFO] Application closed successfully."
    echo "$SEPARATOR"
    # Optional: Short pause even on success so user sees "Goodbye"
    sleep 2
fi
