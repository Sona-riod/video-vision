# Single-Camera Optimization Plan — Pallet-Creation Camera (ICAM-540)

**Status:** Planning only. No code changes yet.
**Scope:** ONE camera (the pallet-creation ICAM-540). Keep all existing business
logic (sessions, API sender, retry/recovery, Zebra printer, beer-types, cloud sync).
**Explicitly out of scope:** PyQt6 rewrite, `multiprocessing` worker fleet,
shared-memory streaming, multi-camera "Global Dashboard". Those solve a 3-camera
scaling problem this system does not have.

---

## 0. Reality check vs. the original rewrite document

| Document claim | Reality in this repo | Verdict |
|---|---|---|
| 3 cameras; build a dashboard + 3 worker processes | `CAMERA_CONFIG` is a single camera, `device: 10` ([config.py:20](config.py#L20)) | **Not applicable** — drop |
| GIL forces YOLO/pyzbar/capture to serialize on one core | Heavy work (CUDA YOLO, pyzbar C-lib, OpenCV) releases the GIL; only one camera | **Overstated** — not the bottleneck for 1 camera |
| UI freezes because detection runs on the Kivy thread | Detection already runs async via `detector_executor.submit(...)` ([main.py:1370](main.py#L1370)) | **Wrong** — detection is already decoupled |
| Switch to `nvarguscamerasrc` for hardware decode | `nvarguscamerasrc` is for CSI Bayer sensors; ICAM-540 is a USB/networked AI cam init'd over REST ([config.py:29-50](config.py#L29)) | **Likely wrong source** — needs the right pipeline |
| Export `best.pt` → TensorRT `.engine` | `QR_MODEL_PATH` loads `best.pt` ([config.py:73](config.py#L73)) | **Correct & highest value** |
| Texture rebuilt every frame on UI thread | `Texture.create()` + 4.5 MB blit every frame ([main.py:1446-1453](main.py#L1446)) | **Correct — real UI cost** |

**Net:** the two genuinely valuable ideas are TensorRT export and a leaner UI-thread
frame path. The camera pipeline needs investigation, not a blind `nvarguscamerasrc` swap.

---

## Priority 1 — Export the QR model to TensorRT (biggest win, lowest risk)

**Why:** Standard `.pt` runs the PyTorch graph; a TensorRT `.engine` is compiled for
the Jetson's exact GPU + precision and typically cuts inference latency substantially.
Ultralytics loads `.engine` transparently — no detector code change beyond the path.

**Steps**
1. On the Jetson (engines are device- and TensorRT-version-specific — must be built
   on the target, not on a dev box):
   ```
   yolo export model=models/model_qr/best.pt format=engine half=True
   ```
   `half=True` enables FP16 (Orin supports it well); drop it if accuracy regresses.
2. Point the config at the engine, but **keep `.pt` as a fallback** so a missing/stale
   engine doesn't brick the app:
   - `config.py`: set `QR_MODEL_PATH` to `best.engine` if it exists, else `best.pt`.
3. Warm up the model once at init (run one dummy `model(np.zeros(...))`) so the first
   real frame doesn't pay the lazy-build cost. `GPU_CONFIG['warmup_enabled']` is
   currently `False` ([config.py:84](config.py#L84)) — wire it up.

**Risk:** engine rebuild needed after any TensorRT/JetPack upgrade. Fallback covers it.
**Validation:** compare the existing avg-detection-time log ([detector.py:403](modules/detector.py#L403))
before vs. after.

---

## Priority 2 — Lighten the Kivy UI-thread frame path (fix the real lag)

Per-frame UI-thread cost today in `update_frame` ([main.py:1336-1455](main.py#L1336)):
`frame.copy()` → draw overlays → `cv2.flip()` (another full copy) → **`Texture.create()`
+ `blit_buffer()` of ~4.5 MB at 1408×1080, every frame**.

**Steps**
1. **Reuse one `Texture`.** Create the texture once at the camera resolution and call
   `blit_buffer` each frame instead of `Texture.create()` per frame. This is the single
   biggest UI-thread saving.
2. **Drop the `cv2.flip` per frame.** Kivy can flip via texture coordinates
   (`texture.flip_vertical()` once, or UV mapping) instead of copying the whole array
   every frame.
3. **Avoid the redundant copy.** `update_frame` does `frame.copy()` then draws on the
   copy; `get_frame()` ([camera.py:194](modules/camera.py#L194)) *already* returns a copy.
   One of the two copies can go (draw on the returned frame, or use `get_frame_no_copy`
   for the read-only detection submit).
4. **Decouple display rate from detection rate** if needed: the UI tick stays at 30 FPS
   for a smooth feed, but detection can run every Nth frame (it's already single-in-flight
   via `current_detection_task`, so this is a small guard).

**Risk:** low; all within `update_frame`/`_update_preview_texture`. No framework change.
**Validation:** watch for dropped Kivy frames / touch responsiveness; the camera-capture
FPS log ([camera.py:186](modules/camera.py#L186)) isolates capture from UI.

---

## Priority 3 — Fix the camera capture pipeline for the *actual* ICAM-540

The document's `nvarguscamerasrc` advice is for CSI sensors and almost certainly does
not apply. The CSI/GStreamer path already exists in [camera.py:99-113](modules/camera.py#L99)
but is unused; config selects `v4l2` ([config.py:21](config.py#L21)).

**Investigation needed first (cannot be guessed):**
1. On the Jetson, determine what the ICAM-540 exposes:
   `v4l2-ctl --device=/dev/video10 --list-formats-ext`
   — i.e. does it output MJPEG/H.264 (compressed, hardware-decodable) or raw YUYV?
2. Check whether the Advantech REST init layer ([config.py:29-43](config.py#L29),
   `modules/camera_init.py`) already configures the stream format/resolution, and whether
   it conflicts with the `1408×1080` in `CAMERA_CONFIG` vs the `3840×2160` init size
   ([config.py:25](config.py#L25) vs [config.py:32](config.py#L32)). These two resolutions
   disagree today — worth reconciling.

**Then, if compressed output is available**, add a GStreamer pipeline that decodes in
hardware, e.g. (MJPEG example — exact elements depend on step 1):
```
v4l2src device=/dev/video10 ! image/jpeg,width=1408,height=1080,framerate=30/1
  ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx
  ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1
```
Wire it as a new `'gstreamer'`/`'icam'` branch in `CameraManager.start()` so the existing
`v4l2`/`test` paths stay as fallbacks.

**Risk:** medium — depends on camera/driver capabilities and GStreamer plugin availability
on the JetPack image. Keep `v4l2` selectable as a fallback.
**Validation:** capture-time + FPS log ([camera.py:186](modules/camera.py#L186)) and CPU%
(`tegrastats`) before/after.

---

## Priority 4 — Detector hygiene (small, optional)

- Confirm the detector actually runs on GPU: `GPU_CONFIG['force_cpu']` is `False`
  ([config.py:85](config.py#L85)) — verify Ultralytics picks `device=0`, and pass it
  explicitly in the `model(...)` call ([detector.py:330](modules/detector.py#L330)).
- `detect_async` / `_executor` inside `QRDetector` ([detector.py:112](modules/detector.py#L112),
  [detector.py:408](modules/detector.py#L408)) are **dead code** — main.py drives concurrency
  with its own `detector_executor`. Safe to delete to reduce confusion (not a perf issue).
- QReader path is capture-only and lazy — leave as-is.

---

## What we are deliberately NOT doing (and why)

- **No PyQt6 migration.** It would re-implement ~1950 lines of working logic in `main.py`
  to fix a UI cost that Priority 2 fixes in ~20 lines. Revisit only if you add real
  multi-camera screens.
- **No `multiprocessing` / shared memory.** Justified only with multiple parallel camera
  pipelines. One camera + async detection does not benefit.
- **No deletion of `theme.py` / `main.py`.**

---

## Suggested execution order

1. P1 TensorRT export + warmup + `.pt` fallback  → measure.
2. P2 reuse texture / drop redundant copies      → measure UI smoothness.
3. P3 camera-format investigation → hardware-decode pipeline if supported.
4. P4 cleanups.

Each step is independently shippable and reversible. Stop after P1+P2 if the camera
already meets the FPS target — P3 is only worth it if capture time (not inference) is
still the dominant cost in the logs.
