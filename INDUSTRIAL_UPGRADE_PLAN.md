# Industrial Upgrade Plan — ICAM-540 Pallet System

**Goal:** Real-time, stable, live-streaming HMI with an app-like multi-page UI.
**Decisions locked in:**
- Single camera (pallet-creation ICAM-540).
- UI framework: **PyQt6**.
- Sequence: **engine first, then UI** — system stays runnable throughout.
- Keep ALL existing business logic (sessions, API, retry/recovery, printer,
  beer-types, cloud sync). We move it, we don't rewrite it.

> Supersedes the earlier `OPTIMIZATION_PLAN.md` notes.

---

## Measured baseline (from the 2026-06-13 run log)

| Symptom | Log evidence | Real cause |
|---|---|---|
| **YOLO never runs** | `PyTorch: NOT INSTALLED`; `YOLO: NOT AVAILABLE (No module named 'torch')`; `QRDetector ... YOLO: False` | torch/ultralytics not installed → `best.pt` never loads → detection falls back to **full-frame Pyzbar only** |
| **Detection ~2 FPS** | `Avg detection time: 468.5ms (2.1 FPS potential)` | full-frame Pyzbar on a 4K image (no YOLO to localize first) |
| **Feed ~14 FPS, ~75ms reads** | `Camera FPS: 13.5, capture time: 72.0ms` (steady) | camera streams **4K** (`Step 3/6 - Setting 4K resolution → 3840×2160`); `cap.read()` blocks for the next frame ≈ 1000/14 ms |
| **UI freeze** | n/a (architectural) | every `frame.copy()` / `cv2.flip()` / `Texture.create()`+blit pays the **4K tax (~24 MB/frame)** on the UI thread |
| API send slow (non-blocking) | `Response received in 6.0507 seconds` | server-side latency; already off the UI thread in `process_worker` — not a freeze cause |
| Cloud config 404 | `POST /api/current-config ... 404` | endpoint missing; cosmetic |

**Conclusion:** the original "GIL + TensorRT" framing was wrong for this system.
The real chain is **4K capture → everything pays a 4K tax → torch missing so no YOLO →
full-frame Pyzbar at 2 FPS**, plus per-frame heavy work on the UI thread.

## Root cause (why it lags / freezes)

`main.py` is one ~1950-line class that owns the camera, detector, session state,
API/printer logic **and** the Kivy widgets, all pumped from `update_frame` on the UI
thread ([main.py:1336](main.py#L1336)). Heavy work lands on the UI thread:
- a new `Texture.create()` + GPU upload **of a 4K frame** every tick ([main.py:1446-1453](main.py#L1446)),
- full-frame `copy()` + `cv2.flip()` per frame at 4K,
- model is `.pt` ([config.py:73](config.py#L73)) **but doesn't even load — torch is absent.**

Detection itself is already async ([main.py:1370](main.py#L1370)) — so the fix is
**resolution + restoring YOLO + architecture (decoupling)**, not a framework swap.

---

## Target architecture

```
            ┌──────────────────────────────────────────────┐
            │              VisionEngine (no UI)             │
            │  thread: capture → detect (1 in-flight) →     │
            │          annotate → publish latest slot       │
            │  owns: CameraManager, QRDetector, SessionStore│
            │  emits: FrameReady(annotated), StateChanged   │
            └───────────────┬───────────────────────────────┘
                            │  thread-safe latest-frame slot + Qt signals
            ┌───────────────▼───────────────────────────────┐
            │                PyQt6 UI (dumb)                 │
            │  QStackedWidget pages, QTimer/slot redraw only │
            │  NO detection, NO API calls, NO heavy copies   │
            └────────────────────────────────────────────────┘
```

**Rule:** the UI thread never does vision work. It only displays the latest published
frame and reacts to state signals. This is what removes the freeze permanently and makes
the UI swappable.

---

## PHASE 0 — Safety net (before any change)

- Branch off `main` (e.g. `feature/industrial-upgrade`).
- Capture a baseline: record current capture-time + FPS logs ([camera.py:186](modules/camera.py#L186),
  [detector.py:403](modules/detector.py#L403)) and `tegrastats` on the Jetson, so every
  later change is measured against real numbers.
- **Paste a real run log** — the engine + optimization targets should be tuned to the
  actual spikes, not assumed ones.

---

## PHASE 1 — Backend engine + stabilization (ship & verify before UI)

Each item is independently shippable and reversible. The app keeps running on Kivy
during this phase; we're only moving logic underneath it.

### 1.0 P0 quick wins (config + install — do these FIRST, before any refactor)

These come straight out of the run log and need almost no code change:

- **Stop streaming 4K for the live pipeline.** Set the REST init to 1080p@30
  ([config.py:32-33](config.py#L32), `CAMERA_INIT_WIDTH/HEIGHT`) and match
  `CAMERA_CONFIG` ([config.py:25](config.py#L25)). Expect FPS to jump toward 30 and
  capture/copy/flip/texture/Pyzbar costs to all drop together.
  - If small/distant QR codes need more pixels: keep the live + detection path at 1080p
    and grab a **single full-res frame only at capture time** (decouple processing res
    from capture res). Don't run the continuous loop at 4K.
- **Install PyTorch-for-Jetson + ultralytics** so YOLO actually loads. Use the
  **NVIDIA Jetson torch wheel** matching the JetPack/CUDA version (NOT plain
  `pip install torch`), then `pip install ultralytics`. Verify the log shows
  `YOLO: LOADED` and `YOLO: True`. This replaces the 468ms full-frame Pyzbar with
  YOLO-localize + small-crop decode.
- **Only after YOLO loads:** export TensorRT (`yolo export ... format=engine half=True`),
  point `QR_MODEL_PATH` at `best.engine` with a `.pt` fallback, enable warmup
  ([config.py:84](config.py#L84)).
- **Acceptance:** log shows ~30 FPS, capture time well under 75ms, detection time in tens
  of ms, `YOLO: True`.

### 1.1 Extract `VisionEngine` (framework-independent)
- New `modules/vision_engine.py`, **zero** Kivy/Qt imports.
- Owns `CameraManager`, `QRDetector`, `SessionStore`/`ScanState`.
- Runs its own worker thread: get frame → (single-in-flight) detect → draw overlays →
  write `(annotated_frame, state_snapshot)` into a lock-protected **single slot**
  (overwrite, never queue — drop-old keeps latency flat).
- Exposes `get_latest()` and a callback/observer hook for state changes.
- Move the orchestration currently in `update_frame` / `_handle_detection_process` /
  `_process_detection_results` / `_check_auto_trigger` ([main.py:1336-1455](main.py#L1336))
  into the engine. The auto-capture → auto-send → auto-print flow moves here too, calling
  the existing `process_worker.submit_batch` unchanged.
- **Acceptance:** current Kivy UI rewired to just call `engine.get_latest()` on its Clock
  tick and blit it. No behavior change, but UI thread no longer runs detection logic.

### 1.2 Reuse one texture / kill per-frame copies
- Create the display texture once; `blit_buffer` each frame instead of `Texture.create()`.
- Remove per-frame `cv2.flip` (flip via texture coords) and the redundant `frame.copy()`
  (`CameraManager.get_frame()` already returns a copy — [camera.py:194](modules/camera.py#L194)).
- **Acceptance:** UI-thread time per frame drops; touch stays responsive under load.

### 1.3 TensorRT engine + warmup
- On the Jetson: `yolo export model=models/model_qr/best.pt format=engine half=True`.
- `config.py`: point `QR_MODEL_PATH` at `best.engine` **if present, else `best.pt`** fallback.
- Wire `GPU_CONFIG['warmup_enabled']` ([config.py:84](config.py#L84)) → run one dummy
  inference at engine init so the first real frame isn't slow.
- **Acceptance:** avg detection time ([detector.py:403](modules/detector.py#L403)) drops vs baseline.

### 1.4 Camera pipeline (hardware decode) — investigation-gated
- On device: `v4l2-ctl --device=/dev/video10 --list-formats-ext` to learn the real output
  format (MJPEG/H.264 vs raw YUYV).
- If compressed: add a `'gstreamer'` branch in `CameraManager.start()` using `v4l2src !
  ... ! nvv4l2decoder ! nvvidconv ! appsink drop=1 max-buffers=1`
  (**not** `nvarguscamerasrc` — CSI-only). Keep `v4l2` as fallback.
- Reconcile the resolution mismatch: capture `1408×1080` ([config.py:25](config.py#L25))
  vs camera-init `3840×2160` ([config.py:32](config.py#L32)). Pick one deliberately.
- **Acceptance:** capture-time spikes shrink; CPU% (tegrastats) drops.

### 1.5 Cleanup
- Delete dead `QRDetector.detect_async` / internal `_executor` ([detector.py:112](modules/detector.py#L112),
  [detector.py:408](modules/detector.py#L408)) — unused; main drives its own executor.

**End of Phase 1 = stable real-time backend, still on the old UI.** Ship it.

---

## PHASE 2 — PyQt6 multi-page app

The engine already exists and holds the logic, so this is a view layer only.

### 2.1 App shell
- `QApplication` + `QMainWindow` + `QStackedWidget` as the page router.
- A persistent nav (side rail or top bar) to switch pages; full-screen kiosk mode for the HMI.
- Reuse the existing splash flow concept ([modules/splash.py](modules/splash.py),
  [modules/camera_init_splash.py](modules/camera_init_splash.py)) as a Qt startup page.

### 2.2 Engine ↔ Qt bridge
- Wrap the engine in a `QObject` that emits `frameReady(QImage)` and `stateChanged(dict)`
  via Qt signals (engine thread → `pyqtSignal`, delivered on the UI thread).
- **Video widget:** a custom `QWidget`/`QLabel` that paints the latest frame; convert
  `numpy BGR → QImage` once per frame and **reuse the buffer** (mirror the texture-reuse
  rule, scale to widget size with `Qt.SmoothTransformation` only if needed).

### 2.3 Pages
| Page | Source data | Notes |
|---|---|---|
| **Live / Dashboard** | engine `frameReady` + `stateChanged` | annotated stream, keg count, stability, auto/manual, status banner |
| **Pallet detail** | `SessionStore` snapshot | QR list, beer type, batch, filling date, send/print status, manual capture/send |
| **History / Reports** | DB via `reports.py` | `QTableView` over `detection_sessions` + `retry_queue`; filter/export |
| **Settings / Diagnostics** | `config.py`, engine stats | camera config, live FPS/capture-time meters, model/TensorRT status, API health, manual retry (`recovery.py`) |

### 2.4 Reuse, don't rewrite
- `api_sender.py`, `process_worker.py`, `database.py`, `printer.py`, `recovery.py`,
  `reports.py`, `session_store.py` are framework-agnostic already — call them from the
  engine/pages unchanged.
- Port theming intent from `theme.py` to a Qt stylesheet (`.qss`).

### 2.5 Cutover
- Run PyQt6 app against the same engine; verify feature parity page-by-page vs the Kivy app.
- Switch the launcher ([run.py](run.py)) to the PyQt6 entry point; keep Kivy entry around
  one release as rollback.

---

## Risks & mitigations
| Risk | Mitigation |
|---|---|
| TensorRT engine breaks after JetPack/TRT upgrade | `.pt` fallback kept in config; rebuild script documented |
| ICAM-540 doesn't expose compressed format | Keep `v4l2` path; hardware-decode is optional gravy |
| PyQt6 video conversion becomes a new UI bottleneck | Buffer reuse + scale-on-demand, same discipline as texture reuse |
| Long non-working window during UI rewrite | Engine-first sequencing keeps old UI working until parity reached |
| Hidden logic in `main.py` not captured by engine | Phase 1 acceptance test = byte-for-byte behavior parity before UI work |

## Milestones
1. **M1** Engine extracted, Kivy rewired to it, behavior parity. *(stability foundation)*
2. **M2** TensorRT + texture reuse + camera pipeline. *(real-time targets met)*
3. **M3** PyQt6 shell + Live page against the engine. *(new UI visible)*
4. **M4** Pallet / History / Settings pages + cutover. *(feature complete)*
