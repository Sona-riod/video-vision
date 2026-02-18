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

# 2. Check for Python Virtual Environment
#    Prioritize the structure: ../venv (User's specific setup)
if [[ -d "../venv" ]]; then
    echo "[INFO] Found virtual environment in parent directory (../venv). Activating..."
    source ../venv/bin/activate
elif [[ -d "venv" ]]; then
    echo "[INFO] Found virtual environment in current directory (venv). Activating..."
    source venv/bin/activate
elif [[ -d "../.venv" ]]; then
    echo "[INFO] Found virtual environment in parent directory (../.venv). Activating..."
    source ../.venv/bin/activate
else
    echo "[WARNING] No virtual environment found in standard locations." >&2
    echo "          Running with system python. This might fail if dependencies are missing." >&2
fi

# 3. Determine Python command
#    Prefer python3, fallback to python
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python is not installed or not found in PATH." >&2
    echo "Press Enter to exit..." >&2
    read
    exit 1
fi

echo "[INFO] Using Python interpreter: $(which $PY_CMD)"
$PY_CMD --version

echo ""
echo "[INFO] Launching main.py..."
echo "$SEPARATOR"

# 4. Run the application
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
