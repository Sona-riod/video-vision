# Root Cause & Solution — ICAM-540 Pallet System Lag/Freeze

**Audience:** engineering + client review.
**Date:** 2026-06-16.
**Verdict in one line:** the lag/freeze is **not** a Kivy problem and **not** a GIL
problem — it is that **full 4K frames flow through stages that don't need 4K (the UI),
while the QR decoder runs in its slowest fallback because YOLO isn't loaded.**

---

## 1. Target hardware & software (the context every fix must respect)

| Item | Value | Source |
|---|---|---|
| Compute | **NVIDIA Jetson Orin** (Tegra Orin, integrated `nvgpu`), aarch64 | run log: `OpenGL renderer NVIDIA Tegra Orin` |
| OS / Python | Ubuntu, **Python 3.10.12**, GCC 11.4 | run log |
| GPU stack | NVIDIA driver 540.x, **OpenCV CUDA: ENABLED**; CUDA + isolated package env now built | run log + user |
| Camera | **Advantech ICAM-540**, 4K, controlled over **REST API at `localhost:5000`**, presented as **V4L2 `/dev/video10`** | log + [config.py:29-50](config.py#L29) |
| Camera init | 6-step REST sequence sets **3840×2160 (4K)** before streaming | log `Step 3/6 - Setting 4K resolution`, [config.py:32](config.py#L32) |
| Detector | YOLO (`best.pt`) + Pyzbar + QReader; OpenCV `QRCodeDetector` fallback | [detector.py](modules/detector.py) |
| Printer | Zebra, **PyUSB fallback** (no `/dev/usb/lp*`) | log |
| UI | Kivy 2.3.1 + KivyMD 1.2.0 (single panel) | log |
| Launch | `.sh` / `.desktop` → venv Python → `main.py` | log header |

**Hard constraint from the client:** **4K must be kept for decoding** (small/distant QR
codes need the pixels). Therefore "lower the resolution" is **not** an acceptable fix.

---

## 2. Measured evidence (from the 2026-06-13 run log)

```
PyTorch: NOT INSTALLED
[DETECTOR] YOLO: NOT AVAILABLE (No module named 'torch')
QRDetector initialized - YOLO: False, Pyzbar: True, QReader: False
...
Avg detection time: 468.5ms (2.1 FPS potential)        <-- decode is in fallback mode
Camera FPS: 13.5, capture time: 72.0ms                 <-- ~1000/14: 4K is camera-rate-limited
```

---

## 3. Root cause (the "why")

There are two faults, and they share one underlying mistake: **no separation between
capture resolution, decode resolution, and display resolution — and no separation of
work across threads.**

### Root cause A — 4K is rendered on the UI thread every frame
`update_frame` runs on the Kivy thread at 30 Hz and, for every tick, does on a **4K
(~24 MB) frame**: `frame.copy()` → `cv2.flip()` → **`Texture.create()` + full 4K GPU
upload** ([main.py:1336-1453](main.py#L1336)). The display does not need 4K — a 720p/1080p
preview is visually identical on the panel — yet it pays the full 4K cost on the one
thread that must stay responsive. **→ this is the freeze.**

### Root cause B — the QR decoder runs full-frame Pyzbar on 4K because YOLO is off
PyTorch isn't installed, so `best.pt` never loads and detection silently drops to the
fallback path: `_decode_pyzbar(full_4K_frame)` ([detector.py:377](modules/detector.py#L377))
= **468 ms (2.1 FPS)**. YOLO's whole purpose is to localise the QR on a small internally-
resized image, then let Pyzbar decode only the **small 4K crop**
([detector.py:340-355](modules/detector.py#L340)). Without it, every frame is a full 4K scan.
**→ this is the slow/missed detection.**

> Note what is **NOT** the root cause: detection is already off the UI thread
> ([main.py:1370](main.py#L1370)); the 6 s API call is already backgrounded in
> `process_worker`; the camera's ~14 fps is the 4K sensor/link rate, not a code bug.

---

## 4. The solution (the "how") — keep 4K, remove the lag

**Principle:** *4K is for decoding only. Display in 720p. Decode in the background.
Overlay the most-recent boxes.* Three independent lanes:

```
Lane 1  CAPTURE  (camera thread — exists)        → holds latest 4K frame
Lane 2  DECODE   (background worker, throttled)   → YOLO on 4K → Pyzbar on 4K crops → boxes+text
Lane 3  DISPLAY  (UI tick, 30 Hz)                 → 4K→720p, draw latest boxes, upload, list QRs
```

Because Lane 3 draws boxes from the **last finished** Lane-2 result (kegs barely move
between frames), the preview renders in ~10 ms and **never waits on the 4K decode**.

### Step 1 — Restore YOLO on the Jetson (prerequisite, biggest single win)
- Install the **NVIDIA Jetson PyTorch wheel** matching this JetPack/CUDA (NOT plain
  `pip install torch`), then `pip install ultralytics`, into the isolated env.
- Confirm the log shows `YOLO: LOADED` / `YOLO: True`.
- Then compile TensorRT: `yolo export model=models/model_qr/best.pt format=engine half=True`,
  point `QR_MODEL_PATH` at `best.engine` with a `.pt` fallback, enable warmup
  ([config.py:84](config.py#L84)).
- **Effect:** decode 468 ms → ~30-60 ms, full 4K accuracy preserved.

### Step 2 — Decouple display resolution from decode (removes the freeze)
In `config.py`:
```python
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1280, 720
DETECT_MIN_INTERVAL = 0.10          # ~10 Hz decode; each decode still full 4K
```
Rewrite `update_frame` into display-only + a throttled background decode trigger, draw
the latest 4K boxes scaled to 720p, and **reuse one texture** (no per-frame
`Texture.create`/`cv2.flip`/4K `copy`). *(Full drop-in code provided separately and ready
to apply to `main.py`.)*
- **Effect:** UI-thread upload 24 MB → ~2.6 MB; no 4K copy/flip on the UI thread; smooth feed.

### Step 3 — (Optional) 4K at a higher frame rate via hardware decode
Confirm the camera's stream format on the device:
```
v4l2-ctl --device=/dev/video10 --list-formats-ext
```
- If it offers **4K MJPEG/H.264**, switch `CameraManager` to a GStreamer pipeline using
  the Orin hardware decoder (`nvv4l2decoder` + `nvvidconv`) — **not** `nvarguscamerasrc`
  (that's CSI-only) — to lift 4K above ~14 fps while offloading the CPU.
- If it only offers **raw 4K**, ~14 fps of new content is a link-bandwidth ceiling;
  Steps 1-2 still make the feed smooth and the decode fast.

### Step 4 — Durable structure
Move Lanes 1-2 + the dual-resolution handoff into a framework-independent `VisionEngine`
so the UI only displays. This is also what makes the planned PyQt6 multi-page UI a low-risk
view swap. (See [INDUSTRIAL_UPGRADE_PLAN.md](INDUSTRIAL_UPGRADE_PLAN.md).)

---

## 5. Before / After (4K preserved throughout)

| Metric | Now | After Step 1+2 |
|---|---|---|
| QR decode time | 468 ms (full-frame Pyzbar, no YOLO) | ~30-60 ms (YOLO + 4K crop) |
| UI-thread upload / frame | ~24 MB (4K) | ~2.6 MB (720p) |
| Preview smoothness | stutters / freezes | smooth, decoupled from decode |
| Decode resolution | 4K | **4K (unchanged)** |
| Box overlay + QR list | works, but tied to slow frame | live, from latest background result |

---

## 6. Action checklist
- [ ] Install Jetson torch wheel + ultralytics in the env → verify `YOLO: True`.
- [ ] Export `best.engine` (TensorRT) + `.pt` fallback + warmup.
- [ ] Apply Step 2 code to `main.py` (display 720p, decode 4K, reuse texture, throttle).
- [ ] Run `v4l2-ctl --list-formats-ext`; if MJPEG/H.264 → hardware-decode GStreamer pipeline.
- [ ] (Later) extract `VisionEngine`, then PyQt6 multi-page UI.
