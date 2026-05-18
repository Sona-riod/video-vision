# modules/process_worker.py
"""
Background worker for batch processing.

Key changes vs original:
  - No module-level APISender singleton (use set_api_sender() from main.py)
  - No db.start_session() inside submit_batch()  → kills session_id race
  - qr_codes accepted as a parameter             → no DB read during processing
  - Single _write_completed_batch() call at end  → kills 5-write-per-batch bug
  - retry_queue still written on API failure     → survives power loss
"""

import os
import time
import logging
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2

from modules.database import DatabaseManager
from config import CAMERA_MAC_ID

log = logging.getLogger(__name__)

# ── Shared resources ─────────────────────────────────────────────────────────
db           = DatabaseManager()
executor     = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ProcessWorker")

# Single APISender instance — injected by main.py after deferred_init()
_api_sender = None


def set_api_sender(instance):
    """
    Called once from main.py deferred_init() to inject the application's
    single APISender.  This prevents a second retry-monitor thread from
    starting at import time (Bug 2 fix).
    """
    global _api_sender
    _api_sender = instance
    log.info("APISender injected into process_worker")


# ── Public entry point ───────────────────────────────────────────────────────

def submit_batch(frame_path: str,
                 image_name: str,
                 session_id: str,
                 qr_codes: List[str],
                 required_count: int = 6,
                 beer_type: str = "",
                 batch: str = "",
                 filling_date: str = None) -> Future:
    """
    Submit a batch for background processing.

    Parameters
    ----------
    frame_path     : absolute path to the saved JPEG
    image_name     : filename portion (for DB record)
    session_id     : unique ID generated ONCE in main.py (timestamp-based)
                     — NOT generated here to prevent race condition (Bug 1 fix)
    qr_codes       : list of QR code strings already decoded by live detector
                     — passed in from SessionStore so we never read the DB here
    required_count : target keg count
    beer_type      : beer type _id returned by cloud beer-types API
    batch          : batch number string from UI
    filling_date   : ISO timestamp of capture
    """
    print("\n" + "=" * 60)
    print("SUBMIT_BATCH CALLED")
    print("=" * 60)
    print(f"  Session ID     : {session_id}")
    print(f"  Frame Path     : {frame_path}")
    print(f"  QR codes in    : {len(qr_codes)} (from SessionStore)")
    print(f"  Required Count : {required_count}")
    print(f"  Beer Type      : {beer_type}")
    print(f"  Batch          : {batch}")
    print(f"  Filling Date   : {filling_date}")

    log.info(f"Submitting batch {session_id} | {len(qr_codes)} QR(s) | beer={beer_type} | batch={batch}")

    if filling_date is None:
        filling_date = datetime.now().isoformat()

    future: Future = executor.submit(
        _process_one,
        frame_path,
        image_name,
        session_id,
        qr_codes,
        required_count,
        beer_type,
        batch,
        filling_date,
    )

    def _callback(f: Future):
        try:
            f.result()
            log.debug(f"[{session_id}] Background processing completed")
        except Exception as e:
            log.exception(f"[{session_id}] Background processing error")

    future.add_done_callback(_callback)
    return future


# ── Internal worker ──────────────────────────────────────────────────────────

def _process_one(frame_path: str,
                 image_name: str,
                 session_id: str,
                 qr_codes: List[str],
                 required_count: int,
                 beer_type: str,
                 batch: str,
                 filling_date: str):
    """
    Runs inside the ThreadPoolExecutor.

    Flow:
    1. Accept QR codes from memory (no DB read needed).
    2. Run advanced detection on the saved frame ONLY if count is short.
    3. Build API payload.
    4. Send to cloud.
    5. ONE DB write at the end — success goes to detection_sessions,
       failure goes to retry_queue.
    """
    start = time.time()
    log.info(f"[{session_id}] Processing started")

    try:
        # ── 1. Use QR codes from SessionStore; run advanced only if short ──
        final_qrs = list(qr_codes)
        adv_used = 0
        adv_found = 0

        if len(final_qrs) < required_count:
            log.info(f"[{session_id}] {len(final_qrs)}/{required_count} QRs — running advanced detection")
            try:
                from modules.detector import detect_qr_advanced
                extra_qrs, adv_found = detect_qr_advanced(frame_path)
                extra_strings = [q['data'] if isinstance(q, dict) else q for q in extra_qrs]
                combined = set(final_qrs) | set(extra_strings)
                final_qrs = list(combined)
                adv_used = 1
                log.info(f"[{session_id}] Advanced detection: {adv_found} found, combined total: {len(final_qrs)}")
            except Exception as e:
                log.exception(f"[{session_id}] Advanced detection failed")

        # ── 2. Build payload ──────────────────────────────────────────────
        payload = {
            "macId":       CAMERA_MAC_ID,
            "kegIds":      final_qrs,
            "kegCount":    len(final_qrs),
            "batch":       batch,
            "beerType":    beer_type,      # always the API _id, never hardcoded
            "fillingDate": filling_date,
            "timestamp":   datetime.now().isoformat(),
        }

        log.debug(f"Payload built: {len(payload.get('kegIds', []))} kegs, batch={payload.get('batch')}")

        # ── 3. Send to API ────────────────────────────────────────────────
        elapsed = time.time() - start
        api_success, pallet_id, error_msg = _send_to_api(session_id, payload)

        # ── 4. ONE DB write at the end ────────────────────────────────────
        elapsed = time.time() - start
        _write_completed_batch({
            "session_id":   session_id,
            "image_name":   image_name,
            "qr_codes":     final_qrs,
            "beer_type":    beer_type,
            "batch":        batch,
            "filling_date": filling_date,
            "required_count": required_count,
            "adv_used":     adv_used,
            "adv_found":    adv_found,
            "elapsed":      elapsed,
            "api_success":  api_success,
            "pallet_id":    pallet_id,
            "error_msg":    error_msg,
            "payload":      payload,
        })

        # Clean up frame on success
        if api_success and os.path.exists(frame_path):
            try:
                os.remove(frame_path)
                log.debug(f"[{session_id}] Frame removed after successful send")
            except Exception as e:
                log.warning(f"[{session_id}] Could not remove frame: {e}")

        status = "SUCCESS" if api_success else "FAILED"
        log.info(f"[{session_id}] {len(final_qrs)}/{required_count} QRs | API:{status} | {elapsed:.2f}s")

        # Return result dict for the caller (main.py future.result())
        return {
            "success":   api_success,
            "pallet_id": pallet_id,
            "error":     error_msg,
            "qr_count":  len(final_qrs),
        }

    except Exception as e:
        elapsed = time.time() - start
        error_msg = f"Unexpected processing error: {str(e)}"
        log.critical(f"[{session_id}] {error_msg}")
        log.exception("Processing error details")

        # Write failure record — frame preserved for manual recovery
        _write_completed_batch({
            "session_id": session_id, "image_name": image_name,
            "qr_codes": list(qr_codes), "beer_type": beer_type, "batch": batch,
            "filling_date": filling_date, "required_count": required_count,
            "adv_used": 0, "adv_found": 0, "elapsed": elapsed,
            "api_success": False, "pallet_id": None,
            "error_msg": error_msg, "payload": None,
        })
        return {"success": False, "pallet_id": None, "error": error_msg, "qr_count": 0}


# ── API send helper ──────────────────────────────────────────────────────────

def _send_to_api(session_id: str, payload: dict):
    """
    Calls the injected APISender.
    Returns (success: bool, pallet_id: str | None, error_msg: str | None).
    """
    if _api_sender is None:
        msg = "APISender not injected — call set_api_sender() first"
        log.error(msg)
        return False, None, msg

    print("\n" + "-" * 40)
    print("CALLING API_SENDER.send_batch()")
    print("-" * 40)

    try:
        success, pallet_id = _api_sender.send_batch(
            batch_id  = session_id,
            qr_codes  = payload["kegIds"],
            payload   = payload,
        )
        print(f"  api_sender returned: success={success}, pallet_id={pallet_id}")
        if success:
            log.info(f"[{session_id}] API send OK — pallet_id={pallet_id}")
        else:
            log.error(f"[{session_id}] API send returned False")
        return success, pallet_id, None

    except Exception as e:
        error_msg = f"API send exception: {str(e)}"
        log.critical(f"[{session_id}] {error_msg}")
        log.exception("API send error details")
        return False, None, error_msg


# ── Single DB write ──────────────────────────────────────────────────────────

def _write_completed_batch(ctx: Dict[str, Any]):
    """
    Write everything to the database in ONE call at the end of processing.
    Uses the existing detection_sessions + retry_queue tables unchanged.
    """
    session_id     = ctx["session_id"]
    qr_codes       = ctx["qr_codes"]
    required_count = ctx["required_count"]
    api_success    = ctx["api_success"]
    payload        = ctx["payload"]
    error_msg      = ctx["error_msg"]

    decoded_cnt = len(qr_codes)
    adimgp      = 1 if decoded_cnt < required_count else 0
    api_status  = "success" if api_success else "failed"
    batch_status = "api_sent" if api_success else "api_failed"

    try:
        # Insert the session record (single write)
        db.start_session_complete({
            "session_id"   : session_id,
            "source_image" : ctx["image_name"],
            "qr_list"      : qr_codes,
            "beer_type"    : ctx["beer_type"],
            "batch"        : ctx["batch"],
            "filling_date" : ctx["filling_date"],
            "target_count" : required_count,
            "decoded_cnt"  : decoded_cnt,
            "adv_used"     : ctx["adv_used"],
            "adv_found"    : ctx["adv_found"],
            "adimgp"       : adimgp,
            "elapsed"      : ctx["elapsed"],
            "api_status"   : api_status,
            "batch_status" : batch_status,
            "pallet_id"    : ctx["pallet_id"],
            "error_msg"    : error_msg,
            "payload"      : payload,
        })
        log.debug(f"[{session_id}] DB write complete | status={batch_status}")
    except Exception as e:
        log.exception(f"[{session_id}] DB write failed")

    # On failure → add to retry_queue so it survives power loss
    if not api_success and payload:
        try:
            db.add_to_retry_queue(session_id, payload, error_msg or "Send failed")
            log.info(f"[{session_id}] Added to retry_queue")
        except Exception as e:
            log.exception(f"[{session_id}] Failed to add to retry_queue")


# ── Retry helper (called by recovery.py or manually) ─────────────────────────

def retry_failed_batch(session_id: str) -> bool:
    """Retry a batch that's currently in the retry_queue."""
    if _api_sender is None:
        log.error("APISender not injected")
        return False
    try:
        retry_items = db.get_retry_queue(limit=100)
        item = next((r for r in retry_items if r["session_id"] == session_id), None)
        if not item:
            log.error(f"[{session_id}] Not found in retry_queue")
            return False

        payload = item["payload"]
        success, _ = _api_sender.send_batch(
            batch_id = session_id,
            qr_codes = payload.get("kegIds", []),
            payload  = payload,
        )
        if success:
            db.remove_from_retry_queue(session_id)
            db.resolve_attention(session_id)
            log.info(f"[{session_id}] Retry successful")
        else:
            log.warning(f"[{session_id}] Retry failed")
        return success
    except Exception as e:
        log.exception(f"[{session_id}] Retry error")
        return False


# ── Utilities ────────────────────────────────────────────────────────────────

def get_active_tasks() -> int:
    return executor._work_queue.qsize()


def shutdown():
    """Graceful shutdown — called by App.on_stop()"""
    log.info("Shutting down process_worker...")
    if _api_sender:
        _api_sender.stop_retry_monitor()
    executor.shutdown(wait=True)
    if _api_sender:
        _api_sender.close()
    log.info("process_worker shutdown complete")


# ── Logging bootstrap ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)