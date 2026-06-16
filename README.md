# Keg Counting & QR Detection System

## Overview

The **Keg Counting & QR Detection System** is a computer-vision HMI for automated keg
counting, QR-code reading, and pallet management in brewery / industrial environments.
It runs on an **NVIDIA Jetson Orin** with an **Advantech ICAM-540 4K camera**, using
**YOLO-based localization**, **multi-method QR decoding** (Pyzbar / QReader / OpenCV
fallback), and **cloud synchronization**. Operators interact through a single dark-themed
Kivy/KivyMD touchscreen panel supporting both automatic and manual capture.

The camera streams at **full 4K (3840×2160)** for decode accuracy while the live preview
is rendered downscaled — see [Architecture](#architecture) below.

## Key Features

- Real-time keg detection and counting at full 4K decode resolution
- QR-code detection via YOLO localization + Pyzbar/QReader/OpenCV decoding
- **Continuous-flow pallet tracking** — kegs already sent are ignored; only new kegs
  count toward the next pallet, which auto-sends and auto-prints when the target is hit
- AUTO and MANUAL capture modes, with automatic fallback to MANUAL when no QR is seen
- Cloud API integration (config sync, beer types, filling-area updates)
- Zebra label printing (USB / PyUSB fallback)
- Background batch sending with retry/backoff and crash recovery
- In-memory session state machine ([modules/session_store.py](modules/session_store.py))
  replacing scattered flags
- Process/run logging and basic reporting

## Architecture

The pipeline is split into **three independent lanes** so the UI never blocks on a 4K
decode (the design that removed the original lag/freeze — see
[ROOT_CAUSE_AND_SOLUTION.md](ROOT_CAUSE_AND_SOLUTION.md)):

| Lane | Where | Job |
|------|-------|-----|
| **1 — Capture** | camera thread | Holds the latest 4K frame |
| **2 — Decode** | background worker (throttled ~10 Hz) | YOLO localizes on 4K → Pyzbar decodes the 4K crop → boxes + text |
| **3 — Display** | Kivy UI tick (30 Hz) | Downscales 4K → 720p preview, draws the most-recent boxes, reuses one texture |

**Principle: 4K is for decoding only; display in 720p; decode in the background; overlay
the last finished result.** Because Lane 3 reuses the previous Lane-2 boxes (kegs barely
move frame to frame), the preview renders in ~10 ms and never waits on the decode.

> **Performance note:** YOLO must be loaded for fast decode. Without PyTorch/Ultralytics,
> detection silently drops to full-frame Pyzbar on 4K (~470 ms/frame). On the Jetson,
> install the **NVIDIA Jetson PyTorch wheel** (not plain `pip install torch`) so YOLO and
> CUDA are available. The launcher prints a `[PREFLIGHT]` line confirming this at startup.

## Project Structure

```
main.py                  # Kivy/KivyMD HMI + 3-lane frame loop (app entry point)
run.py                   # Thin launcher (adds cwd to path, runs SimpleKegApp)
config.py                # Camera, API, GPU, model, preview & detection settings
requirements.txt
Start_palletization.sh   # Jetson/Linux launcher (venv + sudo + YOLO preflight)

modules/
  camera.py              # CameraManager — V4L2 4K capture, latest-frame access
  camera_init.py         # CameraInitializer — 6-step ICAM-540 REST init sequence
  detector.py            # QRDetector — YOLO + Pyzbar/QReader/OpenCV decoding
  advanced.py            # AdvancedQRDetector — tiled/multi-scale decode
  session_store.py       # SessionStore + ScanState — in-memory session state machine
  process_worker.py      # Background batch send (retry/backoff, completed-batch write)
  api_sender.py          # APISender — cloud REST calls
  database.py            # DatabaseManager — SQLite persistence
  pallet_controller.py   # Pallet lifecycle + duplicate-pallet guard
  printer.py             # ZebraPrinter — USB/PyUSB label printing
  gpu_utils.py           # GPU detection / warmup / image ops
  recovery.py            # Startup recovery + DB integrity checks
  reports.py             # ReportGenerator
  splash.py / camera_init_splash.py  # Startup splash screens
  theme.py / utils.py    # Dark color palette + logging/storage helpers

models/model_qr/best.pt  # YOLO QR-localization weights
```

## Supported Platforms

- **Primary:** NVIDIA Jetson Orin (Ubuntu, Python 3.10), CUDA-enabled
- Linux / Windows 10+ for development
- Python 3.8+

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-org>/keg-system.git
cd keg-system
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Install system dependencies (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y libzbar0 ffmpeg libgl1-mesa-glx
```

Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Jetson Orin (important):** do **not** rely on the generic `torch` from PyPI. Install
> the **NVIDIA Jetson PyTorch wheel** matching your JetPack/CUDA, then `pip install
> ultralytics`. Optionally export a TensorRT engine for faster inference:
> `yolo export model=models/model_qr/best.pt format=engine half=True`.

## Configuration

Settings live in [config.py](config.py). Common knobs:

| Setting | Purpose | Default |
|---------|---------|---------|
| `CAMERA_CONFIG` | V4L2 device / 4K resolution / fps | `/dev/video10`, 3840×2160, 30 |
| `CAMERA_INIT_ENABLED` | Run ICAM-540 REST init sequence | `True` |
| `CAMERA_API_BASE_URL` | Camera-init daemon endpoint | `http://localhost:5000` |
| `PREVIEW_WIDTH/HEIGHT` | Downscaled live-preview size | 1280×720 |
| `DETECT_MIN_INTERVAL` | Min seconds between background decodes | 0.10 (~10 Hz) |
| `YOLO_IMGSZ` | YOLO inference size (localizes small QR in 4K) | 1280 |
| `DEFAULT_KEG_COUNT` | Default pallet target | 6 |
| `BASE_URL` / `API_ENDPOINT` | Cloud API base + filling-area update | checkology-cloud.io |
| `GPU_CONFIG.force_cpu` | Force CPU mode | `False` |

## Running the Application

Start the system:

```bash
python main.py
```

This launches the splash screen, ICAM-540 camera init, detection pipeline, operator HMI,
and cloud synchronization.

### One-Click Execution (Jetson/Linux)

```bash
./Start_palletization.sh
```

The launcher locates the venv, runs as root (for Zebra USB access) while exposing the
user-site YOLO stack, prints a torch/CUDA/ultralytics **preflight check**, then starts
`main.py`. Make it executable first: `chmod +x Start_palletization.sh`.

## Operation Modes

- **AUTO** — accumulates detected QR codes; when the target count is reached and stable,
  it auto-captures, sends to the cloud, and prints the label. If no QR is detected for
  ~7 s it shows a banner and falls back to **MANUAL**.
- **MANUAL** — operator positions kegs and presses **CAPTURE**, then **SEND TO SERVER**.

In both modes, kegs already sent in a previous pallet are ignored (continuous flow), so
the line can keep moving without operator resets between pallets.

## Related Documents

- [ROOT_CAUSE_AND_SOLUTION.md](ROOT_CAUSE_AND_SOLUTION.md) — lag/freeze root cause and the 3-lane fix
- [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) — performance optimization plan
- [INDUSTRIAL_UPGRADE_PLAN.md](INDUSTRIAL_UPGRADE_PLAN.md) — VisionEngine extraction & future PyQt6 UI

## Version

- Version: 2.1.0
- Last Updated: June 2026

## License

Proprietary — All rights reserved.
