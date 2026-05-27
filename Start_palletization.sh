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
