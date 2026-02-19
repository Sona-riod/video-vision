# modules/process_worker.py
import os
import time
import logging
import traceback
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import List, Dict, Any
import cv2
from modules.detector import detect_qr_standard, detect_qr_advanced
from modules.database import DatabaseManager
from modules.api_sender import APISender
from config import CAMERA_MAC_ID

log = logging.getLogger(__name__)
db = DatabaseManager()
api_sender = APISender()
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ProcessWorker")

def _process_one(frame_path: str, image_name: str, session_id: str, 
                 required_count: int, composition=None, beer_type="", 
                 batch: str = None, keg_type: str = None, filling_date: str = None):
    """Process a single batch with enhanced error handling and tracking"""
    _log_start(session_id, frame_path, image_name, required_count, beer_type, batch, keg_type)
    
    start = time.time()
    db.update_batch_status(session_id, 'processing', require_attention=False, attention_reason="Processing started")
    
    try:
        frame = _load_frame(frame_path, session_id)
        if frame is None:
            return

        qr_strings, std_dets, method, adv_used, adv_found = _run_detection(frame, frame_path, required_count, session_id)
        
        if not filling_date:
            effective_filling_date = datetime.now().isoformat()
        else:
            effective_filling_date = filling_date
            
        keg_types = [db.get_keg_type(qr_data) for qr_data in qr_strings]
        
        already = _check_and_log_duplicates(qr_strings, required_count, session_id)
        
        _, decoded_cnt = db.store_qr_codes(session_id, qr_strings, method, std_dets, keg_types)
        
        _check_batch_miss(decoded_cnt, required_count, session_id)
        
        if composition is None:
            composition = {'Unknown': decoded_cnt}
            
        db.mark_pallet_processed(qr_strings, session_id, required_count)
        
        payload, qr_list_db, ts_db = _prepare_api_payload(session_id, decoded_cnt, required_count, batch, beer_type, effective_filling_date, qr_strings)
        
        api_success = False
        if payload:
            api_success = _send_api_request(session_id, qr_list_db, payload, image_name, ts_db, required_count)

        _finalize_session(session_id, start, adv_used, adv_found, decoded_cnt, std_dets, api_success, frame_path, already, method, required_count)

    except Exception as e:
        _handle_process_exception(session_id, start, frame_path, e)

def _log_start(session_id, frame_path, image_name, required_count, beer_type, batch, keg_type):
    print("\n" + "="*60)
    print("_PROCESS_ONE STARTED (Background Thread)")
    print("="*60)
    print(f"  Session ID: {session_id}")
    print(f"  Frame Path: {frame_path}")
    print(f"  Image Name: {image_name}")
    print(f"  Required Count: {required_count}")
    print(f"  Beer Type: {beer_type}")
    print(f"  Batch: {batch}")
    print(f"  Keg Type: {keg_type}")
    log.info(f"[{session_id}] Pallet processing started - Target: {required_count} kegs, Beer: {beer_type}, Batch: {batch}")

def _load_frame(frame_path, session_id):
    print("  -> Loading frame from disk...")
    frame = cv2.imread(frame_path)
    if frame is None:
        error_msg = f"Cannot read frame: {frame_path}"
        print(f"  -> ERROR: {error_msg}")
        log.error(f"[{session_id}] {error_msg}")
        db.finish_session(session_id, 0.01, 0, 0, 0, 0, False)
        db.update_batch_status(session_id, 'api_failed', error_msg, 
                             require_attention=True, 
                             attention_reason="Frame read error")
        db.mark_for_attention(session_id, "Frame read error")
        return None
    print(f"  -> Frame loaded: {frame.shape}")
    return frame

def _run_detection(frame, frame_path, required_count, session_id):
    print("  -> Running standard QR detection...")
    std_qrs, std_dets = detect_qr_standard(frame)
    print(f"  -> Standard QR detection: found {len(std_qrs)} QRs")
    
    method = "normal"
    adv_used = 0
    adv_found = 0
    
    if len(std_qrs) < required_count:
        print(f"\n[PROCESS] Standard detection found {len(std_qrs)}/{required_count} QRs - Triggering ADVANCED DETECTION...")
        log.info(f"[{session_id}] Standard → {len(std_qrs)} QR(s), using advanced detection")
        try:
            std_qrs, adv_found = detect_qr_advanced(frame_path)
            print(f"[PROCESS] Standard + Advanced Combined: {adv_found} QRs")
            method = "advanced"
            adv_used = 1
            std_dets = adv_found
            log.info(f"[{session_id}] Advanced detection found {adv_found} QR(s)")
        except Exception as e:
            error_msg = f"Advanced detection failed: {str(e)}"
            log.error(f"[{session_id}] {error_msg}")
            db.update_batch_status(session_id, 'api_failed', error_msg,
                                 require_attention=True,
                                 attention_reason="Advanced detection error")
            db.mark_for_attention(session_id, "Advanced detection error")
    
    qr_strings = [qr['data'] if isinstance(qr, dict) else qr for qr in std_qrs]
    print(f"  -> QR Strings for DB: {qr_strings}")
    return qr_strings, std_dets, method, adv_used, adv_found

def _check_and_log_duplicates(qr_strings, required_count, session_id):
    print(f"  -> Checking for duplicates with: {qr_strings}")
    already, old_session_id = db.is_pallet_processed(qr_strings, required_count)
    print(f"  -> Duplicate check result: already={already}, old_session_id={old_session_id}")
    
    if already:
        try:
            old_info = db.get_batch_status(old_session_id)
            old_user_batch = old_info.get('batch', '')
            log.warning(f"[{session_id}] Duplicate QRs detected (previous: {old_session_id} / Batch: {old_user_batch})")
            log.warning(f"[{session_id}] Proceeding anyway per user request (Reference Mode)")
            
            db.update_batch_status(session_id, 'processing', 
                                  f"Warning: Duplicate of {old_session_id}",
                                  require_attention=False)
        except Exception as e:
            log.error(f"[{session_id}] Duplicate check error: {e}")
    return already

def _check_batch_miss(decoded_cnt, required_count, session_id):
    if decoded_cnt < required_count:
        error_msg = f"Batch miss: Decoded {decoded_cnt}/{required_count} QRs"
        log.warning(f"[{session_id}] {error_msg}")
        db.update_batch_status(session_id, 'api_failed', error_msg,
                             require_attention=True,
                             attention_reason="Batch miss - incomplete QRs")
        db.mark_for_attention(session_id, error_msg)

def _prepare_api_payload(session_id, decoded_cnt, required_count, batch, beer_type, filling_date, qr_strings):
    print("\n" + "-"*40)
    print("STEP 5: PREPARING API PAYLOAD")
    print("-"*40)
    log.info(f"[{session_id}] Ready for API - {decoded_cnt}/{required_count} QR codes")
    
    print("  -> Getting session data from database...")
    qr_list_db, ts_db = db.get_session_data(session_id)
    if not qr_list_db or not isinstance(qr_list_db, list):
        print("  -> WARNING: DB QR list invalid, using local list")
        log.warning(f"[{session_id}] DB QR list invalid, using local list")
        qr_list_db = qr_strings
        ts_db = datetime.now()
    
    if not qr_list_db:
        print("  -> ERROR: No QR codes detected - skipping API call")
        log.warning(f"[{session_id}] Skipping API: No QR codes detected (Empty Batch)")
        db.update_batch_status(session_id, 'api_failed', "Empty batch - not sent")
        return None, None, None

    print("  -> Constructing payload...")
    payload = {
        "macId": CAMERA_MAC_ID,
        "kegIds": qr_list_db,
        "kegCount": decoded_cnt,
        "batch": batch,
        "beerType": beer_type,
        "fillingDate": filling_date,
        "timestamp": ts_db.isoformat() if hasattr(ts_db, 'isoformat') else datetime.now().isoformat()
    }
    
    import json as _json
    print(f"  -> Payload constructed:\n{_json.dumps(payload, indent=4, default=str)}")
    
    print("  -> Storing payload for retry capability...")
    db.store_api_payload(session_id, payload)
    
    return payload, qr_list_db, ts_db

def _send_api_request(session_id, qr_list_db, payload, image_name, ts_db, required_count):
    print("\n" + "-"*40)
    print("STEP 6: CALLING API_SENDER.SEND_BATCH")
    print("-"*40)
    try:
        print("  -> Calling api_sender.send_batch()...")
        api_success = api_sender.send_batch(
            batch_id=session_id,
            qr_codes=qr_list_db,
            payload=payload,
            image_name=image_name,
            timestamp=ts_db,
            required_count=required_count
        )
        print(f"  -> api_sender.send_batch returned: {api_success}")
        
        if api_success:
            print("  -> SUCCESS: Batch sent to API")
            log.info(f"[{session_id}] Batch sent - awaiting pallet QR from cloud")
            db.remove_from_retry_queue(session_id)
        else:
            print("  -> FAILED: API send returned False")
            log.error(f"[{session_id}] API send failed")
            db.mark_for_attention(session_id, "API send failure")
            db.add_to_retry_queue(session_id, payload, "API send failed")
        return api_success
            
    except Exception as e:
        error_msg = f"API send error: {str(e)}"
        print(f"  -> EXCEPTION in send_batch: {e}")
        log.critical(f"[{session_id}] {error_msg}")
        traceback.print_exc()
        db.update_batch_status(session_id, 'api_failed', error_msg,
                             require_attention=True,
                             attention_reason="API send exception")
        db.mark_for_attention(session_id, error_msg)
        db.add_to_retry_queue(session_id, payload, error_msg)
        return False

def _finalize_session(session_id, start, adv_used, adv_found, decoded_cnt, std_dets, api_success, frame_path, already, method, required_count):
    elapsed = time.time() - start
    db.finish_session(session_id, elapsed, adv_used, adv_found, decoded_cnt, std_dets, api_success)
    
    if api_success:
        db.update_batch_status(session_id, 'api_sent', require_attention=False)
        if os.path.exists(frame_path):
            try:
                os.remove(frame_path)
                log.debug(f"[{session_id}] Removed processed frame")
            except Exception as e:
                log.warning(f"[{session_id}] Failed to remove frame: {e}")
    elif not already:
        db.update_batch_status(session_id, 'api_failed', "API send failed",
                             require_attention=True, attention_reason="API failure")
        db.mark_for_attention(session_id, "API failure")

    status = "COMPLETE" if decoded_cnt >= required_count else "INCOMPLETE"
    api_status = "SUCCESS" if api_success else "FAILED"
    log.info(f"[{session_id}] {method.upper()} | {decoded_cnt}/{required_count} | API:{api_status} | {elapsed:.2f}s | {status}")

def _handle_process_exception(session_id, start, frame_path, e):
    error_msg = f"Unexpected processing error: {str(e)}"
    log.critical(f"[{session_id}] {error_msg}")
    traceback.print_exc()
    
    elapsed = time.time() - start
    db.finish_session(session_id, elapsed, 0, 0, 0, 0, False)
    db.update_batch_status(session_id, 'api_failed', error_msg,
                         require_attention=True,
                         attention_reason="Processing crashed")
    db.mark_for_attention(session_id, error_msg)
    
    # Don't delete frame on crash for recovery
    log.warning(f"[{session_id}] Frame preserved for recovery: {frame_path}")

def submit_batch(frame_path: str, image_name: str, session_id: str, 
                 required_count: int = 6, keg_type: str = None, 
                 beer_type: str = None, batch: str = None, 
                 filling_date: str = None) -> Future:
    """Submit a batch for processing in background thread"""
    print("\n" + "="*60)
    print("SUBMIT_BATCH CALLED")
    print("="*60)
    print(f"  Frame Path: {frame_path}")
    print(f"  Image Name: {image_name}")
    print(f"  Session ID: {session_id}")
    print(f"  Required Count: {required_count}")
    print(f"  Keg Type: {keg_type}")
    print(f"  Beer Type: {beer_type}")
    print(f"  Batch: {batch}")
    print(f"  Filling Date: {filling_date}")
    
    log.info(f"Submitting batch {session_id} for processing")
    
    # Start session in database with all parameters
    print("  -> Starting session in database...")
    db.start_session(
        source_image=image_name, 
        target_keg_count=required_count, 
        beer_type=beer_type or "Lager",
        batch=batch,
        filling_date=filling_date
    )
    print("  -> Session started in database")
    
    # Create composition dict for local tracking
    composition = {keg_type: required_count} if keg_type else None
    
    # Submit to thread pool
    print("  -> Submitting to thread pool executor...")
    future: Future = executor.submit(
        _process_one, 
        frame_path, 
        image_name, 
        session_id, 
        required_count, 
        composition, 
        beer_type or "Lager",
        batch,
        keg_type,
        filling_date
    )
    print("  -> Submitted to thread pool")
    
    # Add callback for completion/failure
    def callback(f: Future):
        try:
            f.result()
            print(f"  -> [CALLBACK] {session_id} processing completed successfully")
            log.debug(f"[{session_id}] Background processing completed")
        except Exception as e:
            print(f"  -> [CALLBACK] {session_id} processing error: {e}")
            log.error(f"Background processing error for {session_id}: {e}")
            # The error is already handled in _process_one, so just log
    
    future.add_done_callback(callback)
    
    print("  -> Callback added, returning future")
    return future

def get_active_tasks() -> int:
    """Get number of active processing tasks"""
    return executor._work_queue.qsize()

def get_pending_tasks() -> List[Dict[str, Any]]:
    """Get information about pending tasks"""
    pending = []
    # Note: This is a simplified implementation
    # In production, you might want to track tasks more explicitly
    return pending

def shutdown():
    """Graceful shutdown of worker threads"""
    log.info("Shutting down process worker...")
    
    # Stop API sender retry monitor
    api_sender.stop_retry_monitor()
    
    # Shutdown executor
    executor.shutdown(wait=True, timeout=30)
    
    # Close API sender
    api_sender.close()
    
    log.info("Process worker shutdown complete")

def get_processing_stats() -> Dict[str, Any]:
    """Get processing statistics"""
    return {
        "active_tasks": get_active_tasks(),
        "max_workers": executor._max_workers,
        "thread_name_prefix": executor._thread_name_prefix
    }

def retry_failed_batch(session_id: str) -> bool:
    """Retry a failed batch"""
    try:
        # Get session data
        qr_list, _ = db.get_session_data(session_id)
        if not qr_list:
            log.error(f"[{session_id}] No QR data found for retry")
            return False
        
        # Get batch info from database
        conn = sqlite3.connect(db.db_path, timeout=60)
        cur = conn.cursor()
        cur.execute('''
            SELECT beer_type, batch, filling_date, target_keg_count
            FROM detection_sessions WHERE session_id = ?
        ''', (session_id,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            log.error(f"[{session_id}] Session not found in database")
            return False
        
        beer_type, batch_num, filling_date, target_count = row
        
        # Prepare payload
        payload = {
            "macId": CAMERA_MAC_ID,
            "kegIds": qr_list,
            "beerType": beer_type or "Lager",
            "batch": batch_num,
            "fillingDate": filling_date or datetime.now().isoformat(),
            "kegCount": len(qr_list),
            "targetCount": target_count or 6
        }
        
        # Send retry
        success = api_sender.send_batch(
            batch_id=session_id,
            qr_codes=qr_list,
            payload=payload,
            is_retry=True
        )
        
        if success:
            log.info(f"[{session_id}] Retry successful")
            db.update_batch_status(session_id, 'api_sent', 
                                 require_attention=False)
            db.remove_from_retry_queue(session_id)
            db.resolve_attention(session_id)
        else:
            log.error(f"[{session_id}] Retry failed")
            db.add_to_retry_queue(session_id, payload, "Manual retry failed")
        
        return success
        
    except Exception as e:
        log.error(f"[{session_id}] Retry error: {e}")
        return False

def get_batch_status(session_id: str) -> Dict[str, Any]:
    """Get detailed status of a batch"""
    try:
        # Get basic session data
        qr_list, timestamp = db.get_session_data(session_id)
        
        # Get detailed info from database
        conn = sqlite3.connect(db.db_path, timeout=60)
        cur = conn.cursor()
        cur.execute('''
            SELECT batch_status, require_attention, attention_reason,
                   decodedqrcodes, target_keg_count, beer_type, batch,
                   filling_date, api_attempts, processing_time
            FROM detection_sessions WHERE session_id = ?
        ''', (session_id,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "status": row[0],
            "requires_attention": bool(row[1]),
            "attention_reason": row[2],
            "qr_count": row[3] or 0,
            "target_count": row[4] or 6,
            "beer_type": row[5] or "Unknown",
            "batch": row[6],
            "filling_date": row[7],
            "api_attempts": row[8] or 0,
            "processing_time": row[9] or 0.0,
            "qr_codes": qr_list,
            "timestamp": timestamp
        }
        
    except Exception as e:
        log.error(f"Error getting batch status for {session_id}: {e}")
        return {"error": str(e)}

def cleanup_old_frames(days_old: int = 7):
    """Clean up old frame files"""
    try:
        # This would need to be implemented based on your file storage structure
        log.info(f"Cleanup of frames older than {days_old} days not implemented")
    except Exception as e:
        log.error(f"Frame cleanup error: {e}")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)