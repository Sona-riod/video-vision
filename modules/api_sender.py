# modules/api_sender.py
import requests
import logging
import time
import threading
import json
from datetime import datetime, timedelta
from config import API_TIMEOUT, API_MAX_RETRIES, DB_PATH, CAMERA_NAME, API_ENDPOINT, CAMERA_MAC_ID, ENABLE_PAYLOAD_HASH, BEER_TYPES_ENDPOINT
import sqlite3
import hashlib
import ssl
import urllib3

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HTTP_PREFIX = "https://"
HTTPS_PREFIX = "https://"
CONTENT_TYPE_JSON = "application/json"

# Thread lock for database access
db_lock = threading.RLock()

class APISender:
    def __init__(self, api_url=None, timeout=None, max_retries=None, camera_name=None):
        """
        Initialize API Sender for cloud communication
        """
        # Use the ORIGINAL cloud API endpoint
        self.api_url = api_url or API_ENDPOINT
        self.beer_types_url = BEER_TYPES_ENDPOINT
        self.timeout = timeout or API_TIMEOUT
        self.max_retries = max_retries or API_MAX_RETRIES
        self.camera_name = camera_name or CAMERA_NAME
        
        # Create session with custom SSL context
        self.session = requests.Session()
        
        # Disable SSL verification for development (ENABLE in production!)
        self.session.verify = False
        
        # Set retry configuration
        from requests.adapters import HTTPAdapter, Retry
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount(HTTP_PREFIX, adapter)
        self.session.mount(HTTPS_PREFIX, adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'KegDetectionSystem/1.0',
            'Accept': CONTENT_TYPE_JSON,
            'Content-Type': CONTENT_TYPE_JSON
        })
        
        self.logger = logging.getLogger(__name__)
        self.db_path = str(DB_PATH)
        
        # Network monitoring
        self.network_online = True
        self.last_network_check = None
        self.network_check_interval = 30  # seconds
        
        # Retry thread
        self.retry_thread = None
        self._stop_event = threading.Event()
        self.retry_interval = 60  # seconds
        
        # Start monitoring
        self.start_retry_monitor()
        self.logger.info(f"API Sender initialized. Endpoint: {self.api_url}")

    def get_beer_types(self):
        """
        Fetch beer types from cloud API using the configured endpoint
        """
        headers = {'Content-Type': CONTENT_TYPE_JSON}
        payload = {"macId": CAMERA_MAC_ID}  # Exactly as cloud team specified
        
        try:
            self.logger.info(f"Fetching beer types from: {self.beer_types_url}")
            
            # User requested POST for beer types
            response = self.session.post(self.beer_types_url, json=payload, headers=headers, timeout=5)
            
            self.logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.logger.info(f"Raw response: {data}")
                    
                    beer_types = self._process_beer_types_response(data)
                    
                    if beer_types and len(beer_types) > 0:
                        self.logger.info(f"Successfully fetched {len(beer_types)} beer types")
                        return beer_types
                    else:
                        self.logger.warning(f"Endpoint returned empty beer types list")
                        
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON response: {response.text[:100]}")
                except Exception as e:
                    self.logger.warning(f"Error parsing response: {e}")
            else:
                self.logger.warning(f"Unexpected status {response.status_code}: {response.text[:200]}")
                
        except requests.exceptions.SSLError as e:
            self.logger.warning(f"SSL error: {e}")
        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout connecting to beer types endpoint")
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Network error: {e}")
        except Exception as e:
            self.logger.warning(f"Unexpected error: {e}")
        
        # If the configured endpoint fails, use fallback
        self.logger.error("Beer types endpoint failed.")
        return []

    def _process_beer_types_response(self, data):
        """Helper to process beer types response data"""
        if isinstance(data, list):
            # check if it is a list of dictionaries
            if data and isinstance(data[0], dict):
                 # Keep full object
                 return data
            else:
                 # Convert strings to objects if needed (fallback)
                 return [{"name": x, "id": x} for x in data]
        elif isinstance(data, dict):
            # Try different possible keys
            raw_list = data.get("beer_types", data.get("types", data.get("beerTypes", [])))
            if raw_list and isinstance(raw_list[0], dict):
                 return raw_list
            else:
                 return [{"name": x, "id": x} for x in raw_list]
        return []

    def send_batch(self, batch_id: str, qr_codes: list, payload: dict = None, **kwargs) -> bool:
        """
        Send batch to cloud API
        
        Args:
            batch_id: Unique batch/pallet identifier
            qr_codes: List of decoded QR code strings
            payload: Optional pre-constructed payload (extended)
            **kwargs: Optional parameters
        """
        if not qr_codes:
            self.logger.warning(f"Batch {batch_id}: No QR codes to send")
            return False
        
        self.logger.info(f"Sending batch {batch_id} with {len(qr_codes)} kegs")
        
        # Use provided payload (extended) or create default
        if not payload:
            payload = {
                "macId": CAMERA_MAC_ID,
                "kegIds": qr_codes,
                "kegCount": kwargs.get('keg_count', len(qr_codes)),
                "batch": batch_id,
                "beerType": kwargs.get('beer_type', 'Unknown'),
                "timestamp": kwargs.get('timestamp', datetime.now().isoformat())
            }

        # Store payload in database for retry capability
        try:
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                cur.execute('''
                    UPDATE detection_sessions 
                    SET api_payload = ?, batch_status = 'api_pending', last_api_attempt = ?
                    WHERE session_id = ?
                ''', (json.dumps(payload, default=str), datetime.now(), batch_id))
                conn.commit()
                conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store payload: {e}")
        
        # Send with retry logic
        success = self._send_with_retry(batch_id, payload)
        
        if not success:
            # Add to retry queue
            self._add_to_retry_queue(batch_id, payload, "Initial send failed")
            # Mark for attention
            self._mark_for_attention(batch_id, "API send failure")
        
        return success

    def _send_with_retry(self, batch_id: str, payload: dict, start_attempt: int = 0) -> bool:
        """
        Send request with retry logic
        """
        self._print_start_banner(batch_id, start_attempt)
        
        last_error = None
        
        self._add_payload_hash(payload, batch_id)
        
        print("\nSTEP 3: Preparing HTTP headers...")
        headers = {'Content-Type': CONTENT_TYPE_JSON}
        
        self._log_request_details(payload, headers)
        
        for attempt in range(start_attempt + 1, start_attempt + self.max_retries + 1):
            success, error_msg, should_break = self._process_single_attempt(batch_id, payload, headers, attempt, start_attempt)
            
            if success:
                return True
            
            if error_msg:
                last_error = error_msg
                
            if should_break:
                break
            
            # Wait before retry (exponential backoff)
            if attempt < start_attempt + self.max_retries:
                self._wait_before_retry(attempt, start_attempt)
        
        # Update error in database
        self._log_error_to_db(batch_id, last_error)
        
        self._print_failure_banner()
        return False

    def _process_single_attempt(self, batch_id, payload, headers, attempt, start_attempt):
        self._print_attempt_banner(attempt, start_attempt)
        
        try:
            self.logger.info(f"Sending {batch_id} to cloud (attempt {attempt})")
            
            import time as _time
            start_time = _time.time()
            
            # Try the primary endpoint first
            response = self.session.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                verify=False
            )
            
            elapsed = _time.time() - start_time
            
            self._print_response_details(response, elapsed)
            
            # Check response
            if response.status_code in [200, 201]:
                return self._handle_success_response(response, batch_id), None, False
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"\nSTEP 8: FAILURE - Status {response.status_code}")
                print(f"  Error: {last_error}")
                self.logger.warning(f"Batch {batch_id}: {last_error} (attempt {attempt})")
                return False, last_error, False
                
        except requests.exceptions.SSLError as e:
            # SSL error handling
            if self._attempt_http_fallback(e, batch_id, payload, headers):
                 return True, None, False
            last_error = f"SSL verification failed: {str(e)}"
            self.logger.error(f"Batch {batch_id}: {last_error} (attempt {attempt})")
            return False, last_error, False
            
        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            print(f"\nSTEP 6: TIMEOUT ERROR")
            print(f"  Request timed out after {self.timeout} seconds")
            self.logger.warning(f"Batch {batch_id}: {last_error} (attempt {attempt})")
            return False, last_error, False
        
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {str(e)}"
            print(f"\nSTEP 6: NETWORK ERROR")
            print(f"  Error: {e}")
            self.logger.warning(f"Batch {batch_id}: {last_error} (attempt {attempt})")
            return False, last_error, False
        
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            print(f"\nSTEP 6: UNEXPECTED ERROR")
            print(f"  Error: {e}")
            self.logger.error(f"Batch {batch_id}: {last_error}")
            return False, last_error, True  # Break on unexpected errors

    def _print_start_banner(self, batch_id, start_attempt):
        print("\n" + "="*60)
        print("STEP 1: Starting API Send Process")
        print("="*60)
        print(f"  Batch ID: {batch_id}")
        print(f"  API Endpoint: {self.api_url}")
        print(f"  Start Attempt: {start_attempt}")

    def _print_attempt_banner(self, attempt, start_attempt):
        print(f"\n{'='*60}")
        print(f"STEP 5: Sending request (Attempt {attempt}/{start_attempt + self.max_retries})")
        print("="*60)
        print(f"  URL: {self.api_url}")
        print(f"  Method: POST")
        print(f"  Timeout: {self.timeout} seconds")
        print(f"  SSL Verify: False")

    def _print_response_details(self, response, elapsed):
        print(f"\nSTEP 6: Response received in {elapsed:.4f} seconds")
        print(f"  Status Code: {response.status_code}")
        print(f"  Reason: {response.reason}")
        print(f"  Response Headers: {dict(response.headers)}")
        print(f"\nSTEP 7: Response Body:")
        print("-"*40)
        print(response.text)
        print("-"*40)

    def _wait_before_retry(self, attempt, start_attempt):
        wait_time = 2 ** (attempt - start_attempt - 1)  # 1, 2, 4 seconds
        print(f"\nWaiting {wait_time} seconds before retry...")
        time.sleep(wait_time)

    def _print_failure_banner(self):
        print("\n" + "="*60)
        print("API SEND FAILED")
        print("="*60 + "\n")

    def _add_payload_hash(self, payload, batch_id):
        # Add payload hash for integrity if enabled
        print("\nSTEP 2: Checking payload hash configuration...")
        print(f"  ENABLE_PAYLOAD_HASH: {ENABLE_PAYLOAD_HASH}")
        if ENABLE_PAYLOAD_HASH:
            try:
                data_string = json.dumps(payload, sort_keys=True)
                payload['hash'] = hashlib.sha256(data_string.encode()).hexdigest()
                print(f"  Hash added: {payload['hash'][:20]}...")
                self.logger.debug(f"Added hash to payload for batch {batch_id}")
            except Exception as e:
                print(f"  ERROR adding hash: {e}")
                self.logger.warning(f"Failed to add hash to payload: {e}")
        else:
            print("  Payload hash disabled, skipping...")

    def _log_request_details(self, payload, headers):
        print(f"  Headers: {headers}")
        
        print("\nSTEP 4: Preparing payload...")
        print(f"  Payload keys: {list(payload.keys())}")
        print(f"  Full Payload:")
        print(json.dumps(payload, indent=4, default=str))

    def _handle_success_response(self, response, batch_id):
        print(f"\nSTEP 8: SUCCESS - Status {response.status_code}")
        try:
            resp_data = response.json()
            pallet_id = resp_data.get('paletteId') or resp_data.get('palletId') or resp_data.get('id') or "Unknown"
            print(f"  Parsed JSON successfully")
            print(f"  Pallet ID: {pallet_id}")
            self.logger.info(f"Batch {batch_id} sent successfully. Pallet ID: {pallet_id}")
        except Exception as parse_err:
            print(f"  Could not parse JSON: {parse_err}")
            self.logger.info(f"Batch {batch_id} sent successfully (Status: {response.status_code})")
        
        self._update_db_success(response, batch_id)
        
        print("\n" + "="*60)
        print("API SEND COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        return True

    def _update_db_success(self, response, batch_id):
        # Update database status
        print("\nSTEP 9: Updating database status...")
        try:
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                cur.execute('''
                    UPDATE detection_sessions 
                    SET batch_status = 'api_sent', 
                        api_response = ?,
                        last_api_attempt = ?
                    WHERE session_id = ?
                ''', (response.text, datetime.now(), batch_id))
                conn.commit()
                conn.close()
            print("  Database updated successfully")
        except Exception as e:
            print(f"  ERROR updating database: {e}")
            self.logger.error(f"Failed to update success status: {e}")

    def _log_error_to_db(self, batch_id, last_error):
        print("\nSTEP FINAL: All retries exhausted, updating error in database...")
        if last_error:
            print(f"  Last Error: {last_error}")
            try:
                with db_lock:
                    conn = sqlite3.connect(self.db_path, timeout=30)
                    cur = conn.cursor()
                    cur.execute('''
                        UPDATE detection_sessions 
                        SET last_error = ?, batch_status = 'api_failed',
                            require_attention = 1, attention_reason = ?
                        WHERE session_id = ?
                    ''', (last_error, f"Send failure: {last_error}", batch_id))
                    conn.commit()
                    conn.close()
                print("  Database updated with error status")
            except Exception as e:
                print(f"  ERROR updating database: {e}")
                self.logger.error(f"Failed to update error for {batch_id}: {e}")
    
    def _attempt_http_fallback(self, e, batch_id, payload, headers):
        print(f"\nSTEP 6: SSL ERROR")
        print(f"  Error: {e}")
        # Try HTTP fallback if HTTPS fails
        if self.api_url.startswith(HTTPS_PREFIX):
            http_url = self.api_url.replace(HTTPS_PREFIX, HTTP_PREFIX)
            print(f"  Trying HTTP fallback: {http_url}")
            self.logger.info(f"SSL error, trying HTTP fallback: {http_url}")
            try:
                response = self.session.post(
                    http_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                if response.status_code in [200, 201]:
                    print(f"  HTTP fallback SUCCESS!")
                    self.logger.info(f"Batch {batch_id} sent via HTTP fallback")
                    return True
            except Exception as fallback_err:
                print(f"  HTTP fallback failed: {fallback_err}")
        
        return False

    def _check_network_status(self):
        """Check if API server is reachable"""
        try:
            # Check using the configured API endpoint base
            # Ensure we use the same protocol (http/https) as the actual API
            from urllib.parse import urlparse
            parsed = urlparse(self.api_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            # Use the base URL or the API URL itself (checking for 404/405/200)
            test_url = base_url
            
            self.logger.debug(f"Checking network status via: {test_url}")
            response = self.session.get(test_url, timeout=5, verify=False)
            was_online = self.network_online
            self.network_online = (response.status_code < 500)  # 2xx, 3xx, 4xx considered online
            
            if was_online != self.network_online:
                if self.network_online:
                    self.logger.info("Network: ONLINE - Server reachable")
                else:
                    self.logger.warning(f"Network: OFFLINE - Server returned {response.status_code}")
                    
            self.last_network_check = datetime.now()
            return self.network_online
            
        except requests.exceptions.RequestException as e:
            was_online = self.network_online
            self.network_online = False
            
            if was_online:
                self.logger.error(f"Network: OFFLINE - Connection error: {e}")
            
            self.last_network_check = datetime.now()
            return False

    def start_retry_monitor(self):
        """Start background thread to monitor retry queue"""
        if self.retry_thread and self.retry_thread.is_alive():
            return
            
        self._stop_event.clear()
        self.retry_thread = threading.Thread(
            target=self._retry_monitor_loop,
            name="API_Retry_Monitor",
            daemon=True
        )
        self.retry_thread.start()
        self.logger.info("Retry monitor started")

    def stop_retry_monitor(self):
        """Stop retry monitor thread"""
        self._stop_event.set()
        if self.retry_thread:
            self.retry_thread.join(timeout=2)
            self.logger.info("Retry monitor stopped")

    def _retry_monitor_loop(self):
        """Main loop for retry monitoring"""
        self.logger.info("Retry monitor loop started")
        
        while not self._stop_event.is_set():
            try:
                # Check network every interval
                current_time = datetime.now()
                if (not self.last_network_check or 
                    (current_time - self.last_network_check).seconds >= self.network_check_interval):
                    self._check_network_status()
                
                # Process retry queue if network is online
                if self.network_online:
                    self._process_retry_queue()
                
                # Sleep before next check (interruptible)
                self._stop_event.wait(self.retry_interval)
                
            except Exception as e:
                self.logger.error(f"Retry monitor error: {e}")
                self._stop_event.wait(60)  # Longer sleep on error

    def _process_retry_queue(self):
        """Process batches in retry queue"""
        try:
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                cur.execute('''
                    SELECT session_id, payload, attempts, error_message
                    FROM retry_queue 
                    WHERE next_retry <= ? AND attempts < max_attempts
                    ORDER BY next_retry ASC
                    LIMIT 5
                ''', (datetime.now(),))
                batches = cur.fetchall()
                conn.close()
            
            if not batches:
                return
                
            self.logger.info(f"Processing {len(batches)} batches from retry queue")
            
            for session_id, payload_json, attempts, error_msg in batches:
                try:
                    payload = json.loads(payload_json)
                    self.logger.info(f"Retrying {session_id} (attempt {attempts + 1})")
                    
                    success = self._send_with_retry(session_id, payload, attempts)
                    
                    if success:
                        self._remove_from_retry_queue(session_id)
                    else:
                        self._update_retry_attempts(session_id, attempts + 1)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {session_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in retry queue processing: {e}")

    def _update_retry_attempts(self, session_id: str, attempts: int):
        """Update retry attempt count"""
        try:
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                
                # Calculate next retry time with exponential backoff
                backoff_minutes = 2 ** attempts
                next_retry = datetime.now() + timedelta(minutes=backoff_minutes)
                
                cur.execute('''
                    UPDATE retry_queue 
                    SET attempts = ?, next_retry = ?, last_attempt = ?
                    WHERE session_id = ?
                ''', (attempts, next_retry, datetime.now(), session_id))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            self.logger.error(f"Failed to update retry attempts: {e}")

    def _add_to_retry_queue(self, session_id: str, payload: dict, error_msg: str = None):
        """Add batch to retry queue"""
        try:
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                
                # Get current attempt count
                cur.execute('SELECT attempts FROM retry_queue WHERE session_id = ?', (session_id,))
                result = cur.fetchone()
                attempts = result[0] + 1 if result else 1
                
                # Calculate next retry time
                backoff_minutes = 2 ** (attempts - 1)
                next_retry = datetime.now() + timedelta(minutes=backoff_minutes)
                
                cur.execute('''
                    INSERT OR REPLACE INTO retry_queue 
                    (session_id, payload, last_attempt, attempts, next_retry, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_id, json.dumps(payload, default=str), datetime.now(), 
                     attempts, next_retry, error_msg))
                
                conn.commit()
                conn.close()
            
            self.logger.info(f"Added {session_id} to retry queue (next retry: {next_retry.strftime('%H:%M')})")
            
        except Exception as e:
            self.logger.error(f"Failed to add {session_id} to retry queue: {e}")

    def _remove_from_retry_queue(self, session_id: str):
        """Remove batch from retry queue"""
        try:
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                cur.execute('DELETE FROM retry_queue WHERE session_id = ?', (session_id,))
                conn.commit()
                conn.close()
                self.logger.info(f"Removed {session_id} from retry queue")
        except Exception as e:
            self.logger.error(f"Failed to remove {session_id} from retry queue: {e}")

    def _mark_for_attention(self, session_id: str, reason: str):
        """Mark batch for attention in database"""
        try:
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                cur.execute('''
                    UPDATE detection_sessions 
                    SET require_attention = 1, attention_reason = ?
                    WHERE session_id = ?
                ''', (reason, session_id))
                conn.commit()
                conn.close()
                self.logger.info(f"Marked {session_id} for attention: {reason}")
        except Exception as e:
            self.logger.error(f"Failed to mark {session_id} for attention: {e}")

    def retry_single_batch(self, session_id: str) -> bool:
        """Manually retry a single batch"""
        self.logger.info(f"Manual retry requested for {session_id}")
        
        try:
            # Get payload from database
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                cur.execute('SELECT api_payload FROM detection_sessions WHERE session_id = ?', (session_id,))
                result = cur.fetchone()
                conn.close()
            
            if not result or not result[0]:
                self.logger.error(f"No payload found for {session_id}")
                return False
            
            payload = json.loads(result[0])
            
            # Update status
            with db_lock:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cur = conn.cursor()
                cur.execute('''
                    UPDATE detection_sessions 
                    SET batch_status = 'api_pending', last_api_attempt = ?
                    WHERE session_id = ?
                ''', (datetime.now(), session_id))
                conn.commit()
                conn.close()
            
            # Send with retry
            success = self._send_with_retry(session_id, payload)
            
            if success:
                self.logger.info(f"Manual retry successful for {session_id}")
                self._remove_from_retry_queue(session_id)
                return True
            else:
                self.logger.warning(f"Manual retry failed for {session_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Manual retry error for {session_id}: {e}")
            return False

    def get_network_status(self) -> bool:
        """Get current network status"""
        return self.network_online

    def close(self):
        """Clean shutdown"""
        self.stop_retry_monitor()
        self.session.close()
        self.logger.info("API sender closed")