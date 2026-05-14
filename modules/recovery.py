#!/usr/bin/env python3
# modules/recovery.py - System recovery on startup
import sqlite3
import logging
from datetime import datetime, timedelta
from config import DB_PATH

logger = logging.getLogger(__name__)

def recover_system():
    """Recover system state on startup after crash or power loss"""
    logger.info("=== SYSTEM RECOVERY STARTED ===")
    
    db_path = str(DB_PATH)
    recovered_count = 0
    stuck_count = 0
    
    try:
        conn = sqlite3.connect(db_path, timeout=60)
        cur = conn.cursor()
        
        # 1. Find batches stuck in processing state
        timeout_time = datetime.now() - timedelta(minutes=10)
        cur.execute('''
            SELECT session_id, source_image, batch_status, session_timestamp
            FROM detection_sessions 
            WHERE batch_status IN ('processing', 'api_pending')
            AND session_timestamp < ?
        ''', (timeout_time,))
        
        stuck_batches = cur.fetchall()
        stuck_count = len(stuck_batches)
        
        if stuck_batches:
            logger.warning(f"Found {stuck_count} stuck batches:")
            
            for session_id, image, status, timestamp in stuck_batches:
                logger.warning(f"  - {session_id} ({status}) since {timestamp}")
                
                # Mark as failed for operator attention
                cur.execute('''
                    UPDATE detection_sessions 
                    SET batch_status = 'api_failed', 
                        require_attention = 1,
                        attention_reason = 'Stuck during system restart',
                        last_error = 'System crash/recovery'
                    WHERE session_id = ?
                ''', (session_id,))
                
                # Log system event
                cur.execute('''
                    INSERT INTO system_events (event_type, details)
                    VALUES (?, ?)
                ''', ('batch_recovery', f'{session_id} marked as failed after system restart'))
        
        # 2. Check for batches with payload but no API attempt
        cur.execute('''
            SELECT session_id, api_payload 
            FROM detection_sessions 
            WHERE batch_status = 'api_pending'
            AND api_attempts = 0
            AND api_payload IS NOT NULL
            AND session_timestamp > datetime('now', '-24 hours')
        ''')
        
        pending_batches = cur.fetchall()
        
        if pending_batches:
            logger.info(f"Found {len(pending_batches)} pending batches to retry")
            
            for session_id, payload_json in pending_batches:
                if payload_json:
                    # Add to retry queue
                    cur.execute('''
                        INSERT OR REPLACE INTO retry_queue 
                        (session_id, payload, created_at, next_retry, error_message)
                        VALUES (?, ?, datetime('now'), datetime('now'), 'System recovery')
                    ''', (session_id, payload_json))
                    
                    recovered_count += 1
                    logger.info(f"  - Added {session_id} to retry queue")
        
        # 3. Clean up old retry queue entries (older than 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        cur.execute('''
            DELETE FROM retry_queue 
            WHERE created_at < ? AND attempts >= max_attempts
        ''', (week_ago,))
        
        cleaned = cur.rowcount
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} old retry queue entries")
        
        # 4. Reset network status alerts
        cur.execute('''
            UPDATE system_alerts 
            SET resolved = 1, resolved_at = datetime('now')
            WHERE alert_type = 'network_offline' 
            AND resolved = 0
            AND created_at < datetime('now', '-1 hour')
        ''')
        
        # New: Check for incomplete batches (misses)
        cur.execute('''
            SELECT session_id FROM detection_sessions 
            WHERE decodedqrcodes < target_keg_count AND batch_status != 'api_sent'
        ''')
        miss_batches = cur.fetchall()
        for (session_id,) in miss_batches:
            cur.execute('UPDATE detection_sessions SET require_attention = 1, attention_reason = "Batch miss - incomplete QRs" WHERE session_id = ?', (session_id,))
        
        # Commit all changes
        conn.commit()
        conn.close()
        
        # Log summary
        logger.info("=== SYSTEM RECOVERY COMPLETE ===")
        logger.info(f"Recovered batches: {recovered_count}")
        logger.info(f"Stuck batches marked for attention: {stuck_count}")
        
        return {
            'success': True,
            'recovered': recovered_count,
            'stuck': stuck_count,
            'message': f"Recovery complete: {recovered_count} recovered, {stuck_count} stuck"
        }
        
    except Exception as e:
        logger.error(f"System recovery failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': "System recovery failed"
        }

def check_database_integrity():
    """Check database integrity and fix issues"""
    logger.info("Checking database integrity...")
    
    db_path = str(DB_PATH)
    
    try:
        conn = sqlite3.connect(db_path, timeout=60)
        cur = conn.cursor()
        
        # Check for orphaned retry queue entries
        cur.execute('''
            SELECT rq.session_id 
            FROM retry_queue rq
            LEFT JOIN detection_sessions ds ON rq.session_id = ds.session_id
            WHERE ds.session_id IS NULL
        ''')
        
        orphaned = cur.fetchall()
        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned retry queue entries")
            for (session_id,) in orphaned:
                cur.execute('DELETE FROM retry_queue WHERE session_id = ?', (session_id,))
            logger.info("Cleaned orphaned retry queue entries")
        
        # Check for inconsistent batch status
        cur.execute('''
            SELECT session_id, batch_status, api_status
            FROM detection_sessions 
            WHERE (batch_status = 'api_sent' AND api_status != 'success')
            OR (batch_status = 'api_failed' AND api_status = 'success')
        ''')
        
        inconsistent = cur.fetchall()
        if inconsistent:
            logger.warning(f"Found {len(inconsistent)} inconsistent batch statuses")
            for session_id, batch_status, api_status in inconsistent:
                # Fix based on batch_status (more reliable)
                if batch_status == 'api_sent':
                    cur.execute('UPDATE detection_sessions SET api_status = "success" WHERE session_id = ?', (session_id,))
                elif batch_status == 'api_failed':
                    cur.execute('UPDATE detection_sessions SET api_status = "failed" WHERE session_id = ?', (session_id,))
            logger.info("Fixed inconsistent statuses")
        
        conn.commit()
        conn.close()
        logger.info("Database integrity check complete")
        return True
        
    except Exception as e:
        logger.error(f"Database integrity check failed: {e}")
        return False

if __name__ == "__main__":
    # Run recovery when executed directly
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--integrity':
        check_database_integrity()
    else:
        recover_system()