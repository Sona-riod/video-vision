# modules/camera.py - Optimized Camera Manager with Background Capture
"""
Camera Manager with non-blocking background capture thread.
Optimized for Jetson Orin with:
- Dedicated capture thread (no UI blocking)
- Frame lock for thread-safe access
- Performance timing logs
"""
import cv2
import logging
import numpy as np
import threading
import time
from config import CAMERA_CONFIG

# Setup debug logger
logger = logging.getLogger("CameraManager")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)

log = logging.getLogger(__name__)


class CameraManager:
    """
    Manages the camera lifecycle with background capture thread.
    
    Key features:
    - Non-blocking frame capture via dedicated thread
    - Thread-safe frame access via lock
    - Performance metrics logging
    """
    
    def __init__(self, config=None):
        # Use provided config or default to the module-level CAMERA_CONFIG
        self.config = config if config is not None else CAMERA_CONFIG
        self.cap = None
        self.is_running = False
        
        # Background capture thread
        self._capture_thread = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_ret = False
        self._stop_event = threading.Event()
        
        # Performance tracking
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0
        
        logger.info(f"CameraManager initialized with config: {self.config.get('type', 'unknown')}")

    def start(self):
        """Initializes and opens the camera based on the current configuration."""
        if self.is_running and self.cap is not None:
            logger.warning("Camera is already running.")
            return True

        cam_type = self.config.get('type', 'v4l2').lower()
        logger.info(f"Starting camera type: {cam_type}")

        try:
            # 1. V4L2 (USB Webcams)
            if cam_type == 'v4l2':
                device = self.config.get('device', 0)
                logger.debug(f"Opening V4L2 device: {device}")
                self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('width', 1920))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('height', 1080))
                self.cap.set(cv2.CAP_PROP_FPS, self.config.get('fps', 30))
                # Optimize buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(f"[CAMERA] Started V4L2: {device}")

            # 1b. Generic Webcam (Windows/Mac/Linux)
            elif cam_type == 'webcam':
                device = self.config.get('device', 0)
                logger.debug(f"Opening Webcam device: {device}")
                # On Windows, cv2.CAP_DSHOW can be faster/more compatible, but default is usually fine too.
                # Using default backend for broadest compatibility.
                self.cap = cv2.VideoCapture(device)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('width', 640))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('height', 480))
                self.cap.set(cv2.CAP_PROP_FPS, self.config.get('fps', 30))
                logger.info(f"[CAMERA] Started Webcam: {device}")

            # 2. RTSP (IP Cameras)
            elif cam_type == 'rtsp':
                url = self.config.get('rtsp_url') 
                logger.debug(f"Opening RTSP stream: {url}")
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                logger.info(f"[CAMERA] Started RTSP: {url}")

            # 3. CSI (Jetson/Pi Camera Module)
            elif cam_type == 'csi':
                sensor_id = self.config.get('sensor_id', 0)
                width = self.config.get('width', 1920)
                height = self.config.get('height', 1080)
                fps = self.config.get('fps', 30)

                gst_pipeline = (
                    f"nvarguscamerasrc sensor-id={sensor_id} ! "
                    f"video/x-raw(memory:NVMM), width={width}, height={height}, "
                    f"format=NV12, framerate={fps}/1 ! "
                    f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
                )
                logger.debug(f"Opening CSI with pipeline: {gst_pipeline}")
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                logger.info(f"[CAMERA] Started CSI: Sensor {sensor_id}")

            # 4. File (Video testing)
            elif cam_type == 'file':
                file_path = self.config.get('file_path', 'sample.mp4')
                logger.debug(f"Opening video file: {file_path}")
                self.cap = cv2.VideoCapture(file_path)
                logger.info(f"[CAMERA] Started FILE: {file_path}")

            # 5. Test (Dummy black frame)
            elif cam_type == 'test':
                self.cap = self._create_dummy_cap()
                logger.info("[CAMERA] Started TEST mode")

            else:
                raise ValueError(f"Unknown Camera Type: {cam_type}")

            # Final Check
            if not self.cap.isOpened():
                raise RuntimeError("Camera failed to open (isOpened=False)")
            
            self.is_running = True
            
            # Start background capture thread
            self._stop_event.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_loop, 
                name="CameraCapture",
                daemon=True
            )
            self._capture_thread.start()
            logger.info("[CAMERA] Background capture thread started")
            
            return True

        except Exception as e:
            logger.error(f"Critical Camera Error during start: {e}")
            self.cap = None
            self.is_running = False
            return False

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        logger.info("[CAMERA] Capture loop started")
        
        while not self._stop_event.is_set():
            if self.cap is None:
                time.sleep(0.01)
                continue
            
            try:
                start = time.perf_counter()
                ret, frame = self.cap.read()
                capture_time = (time.perf_counter() - start) * 1000
                
                # Auto-loop for video files
                if not ret and self.config.get('type') == 'file':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                
                # Update shared frame with lock
                with self._frame_lock:
                    self._latest_ret = ret
                    self._latest_frame = frame
                
                # Update FPS counter
                self._frame_count += 1
                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed >= 1.0:
                    self._current_fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._last_fps_time = now
                    logger.debug(f"Camera FPS: {self._current_fps:.1f}, capture time: {capture_time:.1f}ms")
                    
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)
        
        logger.info("[CAMERA] Capture loop stopped")

    def get_frame(self):
        """
        Gets the latest frame from background capture thread.
        NON-BLOCKING - returns immediately with the most recent frame.
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                # Return a copy to prevent threading issues
                return self._latest_ret, self._latest_frame.copy()
            return False, None

    def get_frame_no_copy(self):
        """
        Gets the latest frame WITHOUT copying (faster but not thread-safe for modification).
        Use only if you won't modify the frame.
        """
        with self._frame_lock:
            return self._latest_ret, self._latest_frame

    def get_fps(self):
        """Get current camera FPS."""
        return self._current_fps

    def stop(self):
        """Releases the camera resource and stops capture thread."""
        logger.info("[CAMERA] Stopping camera...")
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            logger.debug("[CAMERA] Capture thread joined")
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        self.cap = None
        self.is_running = False
        self._latest_frame = None
        logger.info("[CAMERA] Stopped")

    def _create_dummy_cap(self):
        """Helper to create a fake camera object for testing."""
        config = self.config
        
        class TestCap:
            def __init__(self):
                self._config = config
                
            def read(self): 
                # Create a test frame with some content
                h = self._config.get('height', 1080)
                w = self._config.get('width', 1920)
                frame = np.zeros((h, w, 3), np.uint8)
                # Add timestamp text
                cv2.putText(frame, f"TEST MODE: {time.strftime('%H:%M:%S')}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return True, frame
                
            def isOpened(self): 
                return True
                
            def release(self): 
                """Dummy method: no resource to release for test camera"""
                pass
                
            def set(self, prop, val): 
                """Dummy method: property setting not supported in test mode"""
                pass
                
            def get(self, prop): 
                """Dummy method: property getting not supported in test mode"""
                return 0
                
        return TestCap()

    def get_stats(self):
        """Get camera statistics."""
        return {
            'is_running': self.is_running,
            'fps': self._current_fps,
            'has_frame': self._latest_frame is not None,
            'type': self.config.get('type', 'unknown')
        }