# modules/camera_init.py
"""
CameraInitializer — configures the Advantech camera via its local REST API
(http://localhost:5000) before the app opens it with cv2.VideoCapture.

This is a Python port of camera_configure.sh. It runs the whole sequence on a
background daemon thread so the Kivy UI (spinner / progress bar) stays responsive
during the ~22 s of hardware "settle" waits.

The sequence (mirrors camera_configure.sh exactly):
    1. Close camera        POST /camera/close              (settle 5 s)
    2. Disable timestamp   POST /camera/timestamp_switch   (mode=0)
    3. Set 4K resolution   POST /camera/resolution         (width, height)
    4. Open camera         POST /camera/open               (settle 5 s)
    5. Play stream         POST /camera/play   -> /dev/video10  (settle 5 s)
    6. Auto-exposure       POST /camera/auto_exposure_op   (auto_exposure_op=1)
    7. Reset focus         POST /camera/reset_focus_position (settle 2 s)
    8. Start auto-focus    POST /camera/autofocus_start

Any non-2xx response or network error aborts the sequence (same as the bash
`post()` helper) and reports the failing step back to the UI.
"""

import time
import threading
import logging

import requests

from config import (
    CAMERA_API_BASE_URL,
    CAMERA_INIT_WIDTH,
    CAMERA_INIT_HEIGHT,
    CAMERA_INIT_CONNECT_TIMEOUT,
    CAMERA_INIT_TIMEOUT,
    CAMERA_INIT_AUTOFOCUS_TIMEOUT,
    CAMERA_INIT_SETTLE_AFTER_CLOSE,
    CAMERA_INIT_SETTLE_AFTER_OPEN,
    CAMERA_INIT_SETTLE_AFTER_PLAY,
    CAMERA_INIT_SETTLE_AFTER_RESET,
)

# -- Logging (same pattern as modules/api_sender.py) ----------
logger = logging.getLogger("CameraInit")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(_h)


def _build_steps():
    """Declarative step list. Built lazily so config values are read at call time."""
    return [
        {"label": "Closing camera",
         "endpoint": "/camera/close", "data": None,
         "settle": CAMERA_INIT_SETTLE_AFTER_CLOSE},

        {"label": "Disabling timestamp overlay",
         "endpoint": "/camera/timestamp_switch", "data": {"mode": "0"},
         "settle": 0},

        {"label": "Setting 4K resolution",
         "endpoint": "/camera/resolution",
         "data": {"width": str(CAMERA_INIT_WIDTH), "height": str(CAMERA_INIT_HEIGHT)},
         "settle": 0},

        {"label": "Opening camera",
         "endpoint": "/camera/open", "data": None,
         "settle": CAMERA_INIT_SETTLE_AFTER_OPEN},

        {"label": "Starting video stream",
         "endpoint": "/camera/play", "data": None,
         "settle": CAMERA_INIT_SETTLE_AFTER_PLAY},

        {"label": "Enabling auto-exposure",
         "endpoint": "/camera/auto_exposure_op", "data": {"auto_exposure_op": "1"},
         "settle": 0},

        {"label": "Resetting focus",
         "endpoint": "/camera/reset_focus_position", "data": None,
         "settle": CAMERA_INIT_SETTLE_AFTER_RESET},

        {"label": "Starting auto-focus",
         "endpoint": "/camera/autofocus_start", "data": None,
         "settle": 0,
         # autofocus_start holds the response open until the lens physically
         # locks focus — needs a much longer read timeout than the other steps.
         "timeout": CAMERA_INIT_AUTOFOCUS_TIMEOUT},
    ]


class CameraInitializer:
    """
    Runs the camera REST configuration sequence on a background thread.

    Callbacks (all invoked from the worker thread — the UI layer is responsible
    for marshalling onto the Kivy main thread, e.g. via Clock.schedule_once):
        status_cb(text: str)              -> human-readable current action
        step_cb(index: int, total: int)   -> 1-based progress
        done_cb(success: bool, error: str)-> final result (error="" on success)
    """

    def __init__(self, base_url=None, status_cb=None, step_cb=None, done_cb=None):
        self.base_url = (base_url or CAMERA_API_BASE_URL).rstrip("/")
        self.status_cb = status_cb or (lambda *_: None)
        self.step_cb = step_cb or (lambda *_: None)
        self.done_cb = done_cb or (lambda *_: None)
        self._thread = None

    # ----------------------------------------------------------
    def run_async(self):
        """Start the sequence on a daemon thread and return immediately."""
        self._thread = threading.Thread(
            target=self._run, name="CameraInit", daemon=True)
        self._thread.start()

    # ----------------------------------------------------------
    def _post(self, endpoint, data, read_timeout):
        """POST one step. Returns (ok: bool, message: str)."""
        url = f"{self.base_url}{endpoint}"
        logger.debug("POST %s | data=%s | read_timeout=%ss", url, data, read_timeout)
        try:
            # camera_configure.sh uses `curl -F` (multipart). `data=` (urlencoded)
            # is accepted by the Advantech API; switch to files={k:(None,v)} if a
            # future firmware insists on multipart.
            # (connect, read) timeout: fail fast if daemon is down, be patient otherwise.
            resp = requests.post(
                url, data=data,
                timeout=(CAMERA_INIT_CONNECT_TIMEOUT, read_timeout))
        except (requests.ConnectionError, requests.ConnectTimeout) as e:
            logger.error("Cannot reach camera service at %s: %s", url, e)
            return False, "cannot reach camera service (is the camera daemon running?)"
        except requests.ReadTimeout as e:
            logger.error("No response from %s within %ss: %s", url, read_timeout, e)
            return False, f"no response within {read_timeout}s — camera may be busy"
        except requests.RequestException as e:
            logger.error("Request failed for %s: %s", url, e)
            return False, f"request failed ({e.__class__.__name__})"

        body = (resp.text or "").strip()
        logger.debug("Response | status=%s | body=%s", resp.status_code, body or "<empty>")
        if 200 <= resp.status_code < 300:
            return True, ""
        return False, f"HTTP {resp.status_code}"

    # ----------------------------------------------------------
    def _run(self):
        steps = _build_steps()
        total = len(steps)
        logger.info("Starting camera initialization (%d steps) at %s",
                    total, self.base_url)

        for idx, step in enumerate(steps, start=1):
            label = step["label"]
            self.step_cb(idx, total)
            self.status_cb(label + "…")
            logger.info("Step %d/%d - %s", idx, total, label)

            read_timeout = step.get("timeout", CAMERA_INIT_TIMEOUT)
            ok, msg = self._post(step["endpoint"], step["data"], read_timeout)
            if not ok:
                error = f"{label} failed ({msg})"
                logger.error("Camera init aborted: %s", error)
                self.done_cb(False, error)
                return

            settle = step.get("settle", 0)
            if settle:
                logger.debug("Settling %ds after '%s'", settle, label)
                time.sleep(settle)

        logger.info("Camera initialization complete. All %d steps succeeded.", total)
        self.done_cb(True, "")
