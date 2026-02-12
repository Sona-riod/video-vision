# modules/database.py
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import threading
from config import DB_PATH

# Thread-safe DB access
db_lock = threading.RLock()

class DatabaseManager:
    def __init__(self):
        self.db_path = str(DB_PATH)
        self._init_db()
        self._migrate_schema()

    def _init_db(self):
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('PRAGMA journal_mode=WAL;')
            cur.execute('PRAGMA foreign_keys = ON;')
            
            # Enhanced detection_sessions table with batch tracking
            cur.execute('''
                CREATE TABLE IF NOT EXISTS detection_sessions (
                    slno INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    source_image TEXT,
                    target_keg_count INTEGER DEFAULT 6,
                    detectionstart INTEGER DEFAULT 1,
                    adimgp INTEGER DEFAULT 0,
                    totaldetection INTEGER,
                    decodedqrcodes INTEGER,
                    processing_time REAL,
                    session_timestamp DATETIME,
                    advanced_detection_used INTEGER DEFAULT 0,
                    advanced_qr_found INTEGER DEFAULT 0,
                    qr_list TEXT DEFAULT '[]',
                    beer_type TEXT DEFAULT 'Lager',
                    batch TEXT,
                    filling_date TEXT,
                    batch_status TEXT DEFAULT 'captured',
                    api_status TEXT DEFAULT 'pending',
                    api_attempts INTEGER DEFAULT 0,
                    last_api_attempt DATETIME,
                    last_error TEXT,
                    api_payload TEXT,
                    require_attention INTEGER DEFAULT 0,
                    attention_reason TEXT,
                    CONSTRAINT valid_status CHECK (
                        batch_status IN ('captured', 'processing', 'api_pending', 'api_sent', 'api_failed', 'duplicate', 'manual_resolved')
                    )
                )
            ''')
            
            # Decoded QR data table with keg_type
            cur.execute('''
                CREATE TABLE IF NOT EXISTS decoded_data (
                    slno INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_image TEXT,
                    qr_data TEXT UNIQUE,
                    first_detected_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    detection_method TEXT DEFAULT 'normal',
                    keg_type TEXT DEFAULT 'Unknown'
                )
            ''')
            
            # Processed pallets table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS processed_pallets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT UNIQUE,
                    batch_id TEXT,
                    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Retry queue for failed API calls
            cur.execute('''
                CREATE TABLE IF NOT EXISTS retry_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    payload TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_attempt DATETIME,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    next_retry DATETIME,
                    error_message TEXT,
                    FOREIGN KEY (session_id) REFERENCES detection_sessions(session_id) ON DELETE CASCADE
                )
            ''')
            
            # System alerts/notifications
            cur.execute('''
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    session_id TEXT,
                    message TEXT NOT NULL,
                    severity TEXT DEFAULT 'medium',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved INTEGER DEFAULT 0,
                    resolved_at DATETIME,
                    auto_resolve INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES detection_sessions(session_id) ON DELETE SET NULL
                )
            ''')
            
            # System events log
            cur.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Pallet lifecycle table (NEW)
            cur.execute('''
                CREATE TABLE IF NOT EXISTS pallet_lifecycle (
                    pallet_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    keg_type TEXT,
                    keg_count INTEGER,
                    qr_codes TEXT,
                    status TEXT DEFAULT 'CREATED',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    shipped_at DATETIME,
                    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
                    qr_generated INTEGER DEFAULT 0,
                    qr_data TEXT,
                    FOREIGN KEY (session_id) REFERENCES detection_sessions(session_id)
                )
            ''')
            
            # Create indexes for performance
            cur.execute('CREATE INDEX IF NOT EXISTS idx_session_ts ON detection_sessions(session_timestamp)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_qr_data ON decoded_data(qr_data)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_batch_status ON detection_sessions(batch_status)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_require_attention ON detection_sessions(require_attention)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_next_retry ON retry_queue(next_retry)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON system_alerts(resolved)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_alerts_type ON system_alerts(alert_type)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_keg_type ON decoded_data(keg_type)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_pallet_status ON pallet_lifecycle(status)')
            
            conn.commit()
            conn.close()
            print(f"[DB] Initialized {self.db_path}")

    def _migrate_schema(self):
        """Migrate existing database to new schema"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            
            try:
                self._migrate_detection_sessions(cur)
                self._migrate_decoded_data(cur)
                self._migrate_system_events(cur)
                self._migrate_batch_status(cur)
                self._create_pallet_lifecycle_table(cur)
                
                conn.commit()
            finally:
                conn.close()

    def _migrate_detection_sessions(self, cur):
        # Get current columns for detection_sessions
        cur.execute("PRAGMA table_info(detection_sessions)")
        columns = [row[1] for row in cur.fetchall()]
        
        # Add missing columns to detection_sessions
        new_columns_detection = [
            ('batch_status', "ALTER TABLE detection_sessions ADD COLUMN batch_status TEXT DEFAULT 'captured'"),
            ('api_attempts', "ALTER TABLE detection_sessions ADD COLUMN api_attempts INTEGER DEFAULT 0"),
            ('last_api_attempt', "ALTER TABLE detection_sessions ADD COLUMN last_api_attempt DATETIME"),
            ('last_error', "ALTER TABLE detection_sessions ADD COLUMN last_error TEXT"),
            ('api_payload', "ALTER TABLE detection_sessions ADD COLUMN api_payload TEXT"),
            ('api_response', "ALTER TABLE detection_sessions ADD COLUMN api_response TEXT"),
            ('require_attention', "ALTER TABLE detection_sessions ADD COLUMN require_attention INTEGER DEFAULT 0"),
            ('attention_reason', "ALTER TABLE detection_sessions ADD COLUMN attention_reason TEXT"),
            ('target_keg_count', "ALTER TABLE detection_sessions ADD COLUMN target_keg_count INTEGER DEFAULT 6"),
            ('beer_type', "ALTER TABLE detection_sessions ADD COLUMN beer_type TEXT DEFAULT 'Lager'"),
            ('api_status', "ALTER TABLE detection_sessions ADD COLUMN api_status TEXT DEFAULT 'pending'"),
            ('batch', "ALTER TABLE detection_sessions ADD COLUMN batch TEXT"),
            ('filling_date', "ALTER TABLE detection_sessions ADD COLUMN filling_date TEXT"),
            ('pallet_id', "ALTER TABLE detection_sessions ADD COLUMN pallet_id TEXT"),
        ]
        
        for col_name, sql in new_columns_detection:
            if col_name not in columns:
                try:
                    cur.execute(sql)
                    print(f"[DB] Added column to detection_sessions: {col_name}")
                except Exception as e:
                    print(f"[DB] Error adding {col_name} to detection_sessions: {e}")

    def _migrate_decoded_data(self, cur):
        # Get current columns for decoded_data
        cur.execute("PRAGMA table_info(decoded_data)")
        columns = [row[1] for row in cur.fetchall()]
        
        # Add keg_type column to decoded_data if not exists
        if 'keg_type' not in columns:
            try:
                cur.execute("ALTER TABLE decoded_data ADD COLUMN keg_type TEXT DEFAULT 'Unknown'")
                print(f"[DB] Added column to decoded_data: keg_type")
            except Exception as e:
                print(f"[DB] Error adding keg_type to decoded_data: {e}")

    def _migrate_system_events(self, cur):
        # Check and rename system_events column if needed
        cur.execute("PRAGMA table_info(system_events)")
        columns = [row[1] for row in cur.fetchall()]
        
        # If old column exists and new one doesn't, rename
        if 'created_at' in columns and 'timestamp' not in columns:
            try:
                # SQLite doesn't support direct column rename, so we need to recreate
                cur.execute('''
                    CREATE TABLE system_events_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        details TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cur.execute('''
                    INSERT INTO system_events_new (id, event_type, details, timestamp)
                    SELECT id, event_type, details, created_at FROM system_events
                ''')
                cur.execute('DROP TABLE system_events')
                cur.execute('ALTER TABLE system_events_new RENAME TO system_events')
                print("[DB] Migrated system_events table")
            except Exception as e:
                print(f"[DB] Error migrating system_events: {e}")

    def _migrate_batch_status(self, cur):
        # Migrate existing data for status
        try:
            cur.execute("UPDATE detection_sessions SET batch_status = 'api_sent' WHERE api_status = 'success'")
            cur.execute("UPDATE detection_sessions SET batch_status = 'api_failed' WHERE api_status = 'failed'")
            cur.execute("UPDATE detection_sessions SET batch_status = 'api_pending' WHERE api_status = 'pending'")
        except Exception as e:
            print(f"[DB] Migration error: {e}")

    def _create_pallet_lifecycle_table(self, cur):
        # Create pallet_lifecycle table if not exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pallet_lifecycle'")
        if not cur.fetchone():
            try:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS pallet_lifecycle (
                        pallet_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        keg_type TEXT,
                        keg_count INTEGER,
                        qr_codes TEXT,
                        status TEXT DEFAULT 'CREATED',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        shipped_at DATETIME,
                        last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
                        qr_generated INTEGER DEFAULT 0,
                        qr_data TEXT,
                        FOREIGN KEY (session_id) REFERENCES detection_sessions(session_id)
                    )
                ''')
                print("[DB] Created pallet_lifecycle table")
            except Exception as e:
                print(f"[DB] Error creating pallet_lifecycle table: {e}")

    def _next_batch_number(self) -> int:
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('SELECT MAX(slno) FROM detection_sessions')
            mx = cur.fetchone()[0]
            conn.close()
            return 1 if mx is None else mx + 1

    def start_session(self, source_image: str, target_keg_count: int = 6, 
                     beer_type: str = "Lager", batch: str = None, 
                     filling_date: str = None) -> Tuple[int, str]:
        batch_no = self._next_batch_number()
        session_id = f"BATCH_{batch_no:04d}"
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO detection_sessions 
                (session_id, source_image, session_timestamp, detectionstart, 
                 target_keg_count, beer_type, batch, filling_date, batch_status) 
                VALUES (?, ?, ?, 1, ?, ?, ?, ?, 'captured')
            ''', (session_id, source_image, datetime.now(), target_keg_count, 
                  beer_type, batch, filling_date))
            
            # Log system event
            cur.execute('''
                INSERT INTO system_events (event_type, details)
                VALUES (?, ?)
            ''', ('session_started', 
                  f'{session_id} - Beer: {beer_type}, Target: {target_keg_count}, Batch: {batch}'))
            
            conn.commit()
            conn.close()
        
        print(f"[DB] STARTED {session_id} | Target: {target_keg_count} | Beer: {beer_type} | Batch: {batch} | Image: {source_image}")
        return batch_no, session_id

    def _insert_global_qr(self, qr: str, source_image: str, method: str, keg_type: str = 'Unknown') -> bool:
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            try:
                cur.execute('''
                    INSERT INTO decoded_data (qr_data, source_image, detection_method, keg_type) 
                    VALUES (?, ?, ?, ?)
                ''', (qr, source_image, method, keg_type))
                conn.commit()
                conn.close()
                return True
            except sqlite3.IntegrityError:
                # Update keg_type if QR already exists
                cur.execute('''
                    UPDATE decoded_data 
                    SET keg_type = COALESCE(?, keg_type)
                    WHERE qr_data = ? AND (keg_type IS NULL OR keg_type = 'Unknown')
                ''', (keg_type, qr))
                conn.commit()
                conn.close()
                return False

    def store_registered_keg(self, qr_data: str, keg_type: str):
        """Store or update keg type for a QR code"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                INSERT OR REPLACE INTO decoded_data (qr_data, keg_type) 
                VALUES (?, ?)
            ''', (qr_data, keg_type))
            conn.commit()
            conn.close()

    def get_keg_type(self, qr_data: str) -> str:
        """Get keg type for a QR code"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute("SELECT keg_type FROM decoded_data WHERE qr_data = ?", (qr_data,))
            row = cur.fetchone()
            conn.close()
            return row[0] if row else 'Unknown'

    def store_qr_codes(self, session_id: str, qr_list: List[str], method: str, 
                       total_detections: int, keg_types: List[str] = None):
        """Store QR codes with optional keg types"""
        if keg_types is None:
            keg_types = ['Unknown'] * len(qr_list)
        
        unique_data = list({(q.strip(), k) for q, k in zip(qr_list, keg_types) if q.strip()})
        unique_qrs = [d[0] for d in unique_data]
        unique_types = [d[1] for d in unique_data]
        
        new_global = sum(self._insert_global_qr(q, session_id, method, k) 
                         for q, k in zip(unique_qrs, unique_types))

        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                UPDATE detection_sessions 
                SET qr_list = ?, decodedqrcodes = ?, totaldetection = ?, batch_status = 'processing'
                WHERE session_id = ?
            ''', (json.dumps(unique_qrs), len(unique_qrs), total_detections, session_id))
            conn.commit()
            conn.close()
        
        print(f"[DB] {session_id} → {len(unique_qrs)} QR(s) stored ({new_global} new)")
        return new_global, len(unique_qrs)

    def update_batch_status(self, session_id: str, status: str, error_msg: str = None, 
                          require_attention: bool = False, attention_reason: str = None):
        """Update batch status with detailed tracking"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            
            update_fields = ['batch_status = ?']
            params = [status]
            
            if error_msg:
                update_fields.append('last_error = ?')
                params.append(error_msg)
            
            if status in ['api_failed', 'api_pending']:
                update_fields.append('last_api_attempt = ?')
                params.append(datetime.now())
                if status == 'api_failed':
                    update_fields.append('api_attempts = api_attempts + 1')
            
            if require_attention:
                update_fields.append('require_attention = 1')
                if attention_reason:
                    update_fields.append('attention_reason = ?')
                    params.append(attention_reason)
            else:
                update_fields.append('require_attention = 0')
                update_fields.append('attention_reason = NULL')
            
            params.append(session_id)
            
            sql = f'UPDATE detection_sessions SET {", ".join(update_fields)} WHERE session_id = ?'
            cur.execute(sql, params)
            
            # Create alert for failures
            if status == 'api_failed' and error_msg:
                cur.execute('''
                    INSERT INTO system_alerts (alert_type, session_id, message, severity)
                    VALUES (?, ?, ?, ?)
                ''', ('api_failure', session_id, f'API failed: {error_msg}', 'high'))
            
            conn.commit()
            conn.close()
        
        print(f"[DB] {session_id} → {status}" + (f" | Error: {error_msg}" if error_msg else ""))

    def store_api_payload(self, session_id: str, payload: dict):
        """Store API payload for retry capability"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                UPDATE detection_sessions 
                SET api_payload = ?, batch_status = 'api_pending'
                WHERE session_id = ?
            ''', (json.dumps(payload, default=str), session_id))
            conn.commit()
            conn.close()

    def finish_session(self, session_id: str, processing_time: float, advanced_used: int, 
                      advanced_found: int, decoded_count: int, total_detection: int, 
                      api_success: bool = None):
        adimgp = 1 if (decoded_count < 6 or total_detection < 6) else 0
        status = "success" if api_success else "failed" if api_success is not None else "pending"
        
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                UPDATE detection_sessions 
                SET processing_time = ?, advanced_detection_used = ?, advanced_qr_found = ?, 
                    adimgp = ?, detectionstart = 0, api_status = ?
                WHERE session_id = ?
            ''', (processing_time, advanced_used, advanced_found, adimgp, status, session_id))
            conn.commit()
            conn.close()
        
        print(f"[DB] FINISHED {session_id} | adimgp={adimgp} | API:{status}")

    def get_batch_status(self, session_id: str) -> str:
        """Get batch status for a session"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute("SELECT batch_status FROM detection_sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            conn.close()
            return row[0] if row else None

    def get_batch_response(self, session_id: str) -> str:
        """Get API response for a session"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute("SELECT api_response FROM detection_sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            conn.close()
            return row[0] if row else None

    def get_decoded_count(self, session_id: str) -> int:
        """Get number of decoded QR codes for a session"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute("SELECT decodedqrcodes FROM detection_sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            conn.close()
            return row[0] if row else 0

    def mark_for_attention(self, session_id: str, reason: str):
        """Mark a batch as requiring attention"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                UPDATE detection_sessions 
                SET require_attention = 1, attention_reason = ? 
                WHERE session_id = ?
            ''', (reason, session_id))
            conn.commit()
            conn.close()

    def resolve_attention(self, session_id: str):
        """Mark a batch as no longer requiring attention"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                UPDATE detection_sessions 
                SET require_attention = 0 
                WHERE session_id = ?
            ''', (session_id,))
            conn.commit()
            conn.close()

    def get_batches_requiring_attention(self) -> List[Dict[str, Any]]:
        """Get all batches that need operator attention"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                SELECT session_id, batch_status, last_error, decodedqrcodes, 
                       session_timestamp, require_attention, attention_reason,
                       beer_type, target_keg_count, api_attempts, batch, filling_date
                FROM detection_sessions 
                WHERE require_attention = 1 
                ORDER BY session_timestamp DESC
            ''')
            results = cur.fetchall()
            conn.close()
        
        batches = []
        for row in results:
            batches.append({
                'session_id': row[0],
                'status': row[1],
                'error': row[2],
                'qr_count': row[3] or 0,
                'timestamp': row[4],
                'require_attention': bool(row[5]),
                'reason': row[6] or 'Unknown',
                'beer_type': row[7] or 'Unknown',
                'target_count': row[8] or 6,
                'api_attempts': row[9] or 0,
                'batch': row[10],
                'filling_date': row[11]
            })
        
        return batches

    def get_attention_batches(self) -> List[Dict[str, Any]]:
        """Get batches requiring attention (simplified format)"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                SELECT session_id, batch_status, attention_reason, decodedqrcodes, target_keg_count 
                FROM detection_sessions 
                WHERE require_attention = 1
                ORDER BY session_timestamp DESC
            ''')
            rows = cur.fetchall()
            conn.close()
        
        return [{
            'id': r[0], 
            'status': r[1], 
            'reason': r[2], 
            'qr_count': r[3], 
            'target': r[4]
        } for r in rows]

    def get_attention_count(self) -> int:
        """Count batches needing attention"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM detection_sessions WHERE require_attention = 1')
            count = cur.fetchone()[0]
            conn.close()
        return count

    def add_to_retry_queue(self, session_id: str, payload: dict, error_msg: str = None):
        """Add batch to retry queue with exponential backoff"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            
            # Get current attempt count
            cur.execute('SELECT attempts FROM retry_queue WHERE session_id = ?', (session_id,))
            result = cur.fetchone()
            attempts = result[0] + 1 if result else 1
            
            # Calculate next retry time (exponential backoff)
            backoff_minutes = 2 ** (attempts - 1)  # 1, 2, 4, 8 minutes
            next_retry = datetime.now() + timedelta(minutes=backoff_minutes)
            
            cur.execute('''
                INSERT OR REPLACE INTO retry_queue 
                (session_id, payload, last_attempt, attempts, next_retry, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, json.dumps(payload, default=str), datetime.now(), attempts, next_retry, error_msg))
            
            conn.commit()
            conn.close()

    def get_retry_queue(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get batches ready for retry"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                SELECT session_id, payload, attempts, next_retry, error_message
                FROM retry_queue 
                WHERE next_retry <= ? AND attempts < max_attempts
                ORDER BY next_retry ASC
                LIMIT ?
            ''', (datetime.now(), limit))
            results = cur.fetchall()
            conn.close()
        
        retry_items = []
        for row in results:
            try:
                payload = json.loads(row[1])
            except Exception:
                payload = {}
            
            retry_items.append({
                'session_id': row[0],
                'payload': payload,
                'attempts': row[2],
                'next_retry': row[3],
                'error_message': row[4]
            })
        
        return retry_items

    def remove_from_retry_queue(self, session_id: str):
        """Remove batch from retry queue after successful send"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('DELETE FROM retry_queue WHERE session_id = ?', (session_id,))
            conn.commit()
            conn.close()

    def mark_batch_resolved(self, session_id: str, reason: str = "Manually resolved"):
        """Mark batch as resolved (no longer requires attention)"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                UPDATE detection_sessions 
                SET require_attention = 0, attention_reason = ?, batch_status = 'manual_resolved'
                WHERE session_id = ?
            ''', (reason, session_id))
            conn.commit()
            conn.close()

    def get_stuck_batches(self, timeout_minutes: int = 10) -> List[str]:
        """Find batches stuck in processing state"""
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            timeout_time = datetime.now() - timedelta(minutes=timeout_minutes)
            cur.execute('''
                SELECT session_id 
                FROM detection_sessions 
                WHERE batch_status IN ('processing', 'api_pending')
                AND session_timestamp < ?
            ''', (timeout_time,))
            results = cur.fetchall()
            conn.close()
        
        return [row[0] for row in results]

    def get_session_data(self, session_id: str) -> Tuple[List[str], Optional[datetime]]:
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            try:
                cur.execute("SELECT qr_list, session_timestamp FROM detection_sessions WHERE session_id = ?", (session_id,))
                row = cur.fetchone()
                conn.close()
                if row and row[0]:
                    try:
                        return json.loads(row[0]), row[1]
                    except json.JSONDecodeError:
                        print(f"[DB] Corrupted qr_list for {session_id}")
                        return [], None
                return [], None
            except Exception as e:
                conn.close()
                print(f"[DB] Error reading {session_id}: {e}")
                return [], None

    def is_pallet_processed(self, qr_codes: List[str], required_count: int = 6) -> Tuple[bool, str]:
        if len(qr_codes) != required_count:
            return False, ""
        fingerprint = json.dumps(tuple(sorted(qr_codes)))
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute("SELECT batch_id FROM processed_pallets WHERE fingerprint = ?", (fingerprint,))
            row = cur.fetchone()
            conn.close()
            return (row is not None, row[0] if row else "")

    def mark_pallet_processed(self, qr_codes: List[str], batch_id: str, required_count: int = 6):
        if len(qr_codes) != required_count:
            return
        fingerprint = json.dumps(tuple(sorted(qr_codes)))
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            try:
                cur.execute("INSERT INTO processed_pallets (fingerprint, batch_id) VALUES (?, ?)",
                           (fingerprint, batch_id))
                conn.commit()
            except sqlite3.IntegrityError:
                pass
            conn.close()

    # NEW METHODS FOR PALLET CONTROLLER
    def check_pallet_duplicate(self, qr_codes: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Check if pallet with same QR codes already exists"""
        fingerprint = json.dumps(tuple(sorted(qr_codes)))
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                SELECT pallet_id, status FROM pallet_lifecycle 
                WHERE qr_codes = ? OR json_extract(qr_codes, '$') = ?
            ''', (fingerprint, fingerprint))
            row = cur.fetchone()
            conn.close()
            return (row[0], row[1]) if row else (None, None)

    def create_pallet_record(self, session_id: str, keg_type: str, keg_count: int, qr_codes: List[str]) -> str:
        """Create a new pallet record in database"""
        pallet_id = f"PALLET_{int(datetime.now().timestamp())}"
        fingerprint = json.dumps(tuple(sorted(qr_codes)))
        
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO pallet_lifecycle 
                (pallet_id, session_id, keg_type, keg_count, qr_codes, status)
                VALUES (?, ?, ?, ?, ?, 'CREATED')
            ''', (pallet_id, session_id, keg_type, keg_count, fingerprint))
            conn.commit()
            conn.close()
        
        print(f"[DB] Created pallet {pallet_id} for session {session_id}")
        return pallet_id

    def print_summary(self):
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM detection_sessions'); batches = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM decoded_data'); unique_qr = cur.fetchone()[0]
            cur.execute('SELECT SUM(decodedqrcodes) FROM detection_sessions'); total_mappings = cur.fetchone()[0] or 0
            cur.execute('SELECT COUNT(*) FROM detection_sessions WHERE advanced_detection_used = 1'); adv_batches = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM processed_pallets'); pallets = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM detection_sessions WHERE batch_status = "api_failed"'); failed = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM detection_sessions WHERE require_attention = 1'); attention = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM retry_queue'); retry_queue = cur.fetchone()[0]
            cur.execute('SELECT COUNT(DISTINCT keg_type) FROM decoded_data WHERE keg_type != "Unknown"'); keg_types = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM pallet_lifecycle'); total_pallets = cur.fetchone()[0]
            
            print("\n" + "="*70 + "\nDATABASE SUMMARY\n" + "="*70)
            print(f"Total Batches Processed      : {batches}")
            print(f"Total Unique QR Codes        : {unique_qr}")
            print(f"Total QR Mappings            : {total_mappings}")
            print(f"Batches using Advanced Method: {adv_batches}")
            print(f"Unique Pallets Processed     : {pallets}")
            print(f"Pallets in Lifecycle         : {total_pallets}")
            print(f"Failed API Sends             : {failed}")
            print(f"Batches Needing Attention    : {attention}")
            print(f"Batches in Retry Queue       : {retry_queue}")
            print(f"Distinct Keg Types Tracked   : {keg_types}")
            print("="*70 + "\n")
            conn.close()

    def get_next_batch_number(self) -> int:
        return self._next_batch_number()

    def is_batch_number_sent(self, batch_number: str) -> Tuple[bool, Optional[str]]:
        """Check if a batch number has already been sent to API successfully.
        
        Returns:
            Tuple[bool, str]: (is_duplicate, session_id if duplicate)
        """
        if not batch_number:
            return False, None
        
        with db_lock:
            conn = sqlite3.connect(self.db_path, timeout=60)
            cur = conn.cursor()
            cur.execute('''
                SELECT session_id FROM detection_sessions 
                WHERE batch = ? AND batch_status = 'api_sent'
                ORDER BY session_timestamp DESC
                LIMIT 1
            ''', (batch_number,))
            row = cur.fetchone()
            conn.close()
            
            if row:
                return True, row[0]
            return False, None