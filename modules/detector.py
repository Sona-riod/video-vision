# modules/detector.py - Optimized QR Code Detection with Performance Logging
"""
QR Detector using Pyzbar (fast) + QReader (robust) fallback chain.
Optimized for Jetson Orin with:
- Early exit on successful detection
- Optional QReader for final capture only
- Performance timing logs
"""
import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
import time
import logging

# Setup debug logger
logger = logging.getLogger("QRDetector")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)

# Performance timing helper
def log_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed:.1f}ms")
        return result
    return wrapper

# Suppress warnings during imports
warnings.filterwarnings('ignore')

# ===== INITIALIZATION =====
print("\n" + "="*60)
print("INITIALIZING QR DETECTOR (OPTIMIZED)")
print("="*60)

# Check CUDA availability
CUDA_AVAILABLE = False
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    print(f"[DETECTOR] PyTorch: {torch.__version__}")
    print(f"[DETECTOR] CUDA Available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"[DETECTOR] GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("[DETECTOR] PyTorch: NOT AVAILABLE")

# Import Pyzbar (primary decoder - fast)
PYZBAR_AVAILABLE = False
pyzbar_decode = None
ZBarSymbol = None
try:
    from pyzbar.pyzbar import decode as _pyzbar_decode, ZBarSymbol as _ZBarSymbol
    pyzbar_decode = _pyzbar_decode
    ZBarSymbol = _ZBarSymbol
    PYZBAR_AVAILABLE = True
    print("[DETECTOR] Pyzbar: LOADED (Primary - Fast)")
except ImportError as e:
    print(f"[DETECTOR] Pyzbar: NOT AVAILABLE ({e})")

# Import QReader (fallback decoder - more robust but slower)
QREADER_AVAILABLE = False
QReaderClass = None
try:
    from qreader import QReader as _QReaderClass
    QReaderClass = _QReaderClass
    QREADER_AVAILABLE = True
    print("[DETECTOR] QReader: LOADED (Fallback - Slow)")
except Exception as e:
    print(f"[DETECTOR] QReader: NOT AVAILABLE ({e})")

# Import config
try:
    from config import QR_CONF_THRESHOLD, GPU_CONFIG
except ImportError:
    QR_CONF_THRESHOLD = 0.5
    GPU_CONFIG = {'force_cpu': False}

# Apply Force CPU override
if TORCH_AVAILABLE and CUDA_AVAILABLE and GPU_CONFIG.get('force_cpu', False):
    print("[DETECTOR] GPU detected but FORCE_CPU is enabled. Switching to CPU.")
    CUDA_AVAILABLE = False
    if torch.cuda.is_available():
         torch.cuda.is_available = lambda : False # Monkey patch to be sure? No, just rely on CUDA_AVAILABLE flag in class


# Determine detection mode
if PYZBAR_AVAILABLE and QREADER_AVAILABLE:
    DETECTION_MODE = "Pyzbar + QReader"
elif PYZBAR_AVAILABLE:
    DETECTION_MODE = "Pyzbar Only"
elif QREADER_AVAILABLE:
    DETECTION_MODE = "QReader Only"
else:
    DETECTION_MODE = "OpenCV Fallback"

print("="*60)
print(f"DETECTOR STATUS (OPTIMIZED):")
print(f"   Pyzbar: {'OK' if PYZBAR_AVAILABLE else 'FAILED'}")
print(f"   QReader: {'OK' if QREADER_AVAILABLE else 'FAILED'}")
print(f"   Mode: {DETECTION_MODE}")
print(f"   GPU: {'ENABLED' if CUDA_AVAILABLE else 'CPU'}")
print("="*60 + "\n")


class QRDetector:
    """
    Optimized QR Detector with fallback chain: Pyzbar -> QReader -> OpenCV
    
    Performance optimizations:
    - Early exit on first successful detection
    - Skip enhanced image if gray works
    - Optional QReader (disabled for live preview)
    - Single-scale first, multi-scale only if needed
    """
    
    def __init__(self):
        self.pyzbar_available = PYZBAR_AVAILABLE
        self.qreader_available = QREADER_AVAILABLE
        
        # Initialize GPU Processor
        from modules.gpu_utils import get_gpu_processor
        self.gpu_processor = get_gpu_processor()
        self.gpu_enabled = self.gpu_processor.gpu_available
        
        # Initialize QReader instance if available (lazy - only on first use)
        self.qreader = None
        self._qreader_initialized = False
        
        # OpenCV QR detector as last resort
        self.cv_detector = cv2.QRCodeDetector()
        
        # Detection settings - MULTI-SCALE (Preserved for Distance)
        self.scale_factors = [1.0, 0.75, 0.5]
        self.use_multiscale = True  # ENABLED for distance
        
        # Threading for async detection
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="QRDetect")
        self._lock = threading.Lock()
        self._last_results = []
        self._detection_running = False
        
        # Performance stats
        self._frame_count = 0
        self._total_time = 0
        
        mode = "GPU ACCELERATED" if self.gpu_enabled else "CPU"
        logger.info(f"QRDetector initialized in {mode} mode")
        logger.info(f"Pyzbar: {'Yes' if self.pyzbar_available else 'No'}")
        logger.info(f"Multi-scale: Enabled (1.0, 0.75, 0.5) - using {mode} resizing")

    def _init_qreader(self):
        """Lazy initialize QReader only when needed"""
        if self._qreader_initialized:
            return
        
        if self.qreader_available and QReaderClass is not None:
            try:
                logger.info("Initializing QReader model (one-time)...")
                start = time.perf_counter()
                self.qreader = QReaderClass(model_size='s', min_confidence=0.5)
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(f"QReader initialized in {elapsed:.0f}ms")
            except Exception as e:
                logger.error(f"QReader init failed: {e}")
                self.qreader_available = False
        self._qreader_initialized = True

    @log_time
    def _preprocess_frame_gpu(self, frame, scale=1.0):
        """
        Preprocess frame on GPU:
        1. Upload to GPU (if needed)
        2. Resize (if scale != 1.0)
        3. Convert to Gray
        4. Download Gray for pyzbar
        """
        # 1. Upload
        gpu_frame = self.gpu_processor.upload_to_gpu(frame)
        
        # 2. Resize on GPU
        if scale != 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            gpu_frame = self.gpu_processor.resize_gpu(gpu_frame, (new_w, new_h))
            
        # 3. Convert to Gray on GPU
        gpu_gray = self.gpu_processor.cvt_color_gpu(gpu_frame, cv2.COLOR_BGR2GRAY)
        
        # 4. Download (Pyzbar needs CPU numpy array)
        gray = self.gpu_processor.download_from_gpu(gpu_gray)
        
        return gray

    def _decode_pyzbar(self, image):
        """Decode QR codes using Pyzbar (fast)."""
        if not self.pyzbar_available or pyzbar_decode is None:
            return []
        
        results = []
        try:
            # start = time.perf_counter()
            decoded = pyzbar_decode(image, symbols=[ZBarSymbol.QRCODE])
            # elapsed = (time.perf_counter() - start) * 1000
            
            for obj in decoded:
                text = obj.data.decode("utf-8")
                if text:
                    rect = obj.rect
                    bbox = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)
                    results.append({'data': text, 'bbox': bbox, 'source': 'pyzbar'})
            
            # if results:
            #     logger.debug(f"Pyzbar found {len(results)} QR codes in {elapsed:.1f}ms")
        except Exception as e:
            logger.warning(f"Pyzbar error: {e}")
        
        return results

    def _decode_qreader(self, image):
        """Decode QR codes using QReader (robust, uses ML) - SLOW."""
        if self.qreader is None:
            return []
        
        # print("[DETECTOR] Running QReader model scan...")
        results = []
        try:
            start = time.perf_counter()
            texts = self.qreader.detect_and_decode(image=image)
            elapsed = (time.perf_counter() - start) * 1000
            
            if texts:
                # print(f"[DETECTOR] QReader Raw Output: {texts}")
                for i, text in enumerate(texts):
                    if text:
                        h, w = image.shape[:2]
                        results.append({
                            'data': text, 
                            'bbox': (0, 0, w, h),
                            'source': 'qreader'
                        })
            
            if len(results) > 0:
                logger.debug(f"QReader found {len(results)} QR codes in {elapsed:.1f}ms")
            
        except Exception as e:
            logger.warning(f"QReader error: {e}")
        
        return results

    def _decode_opencv(self, image):
        """Fallback: Decode using OpenCV's QRCodeDetector."""
        results = []
        try:
            # start = time.perf_counter()
            data, points, _ = self.cv_detector.detectAndDecode(image)
            # elapsed = (time.perf_counter() - start) * 1000
            
            if data and points is not None:
                pts = points[0]
                x_min = int(min(pts[:, 0]))
                y_min = int(min(pts[:, 1]))
                x_max = int(max(pts[:, 0]))
                y_max = int(max(pts[:, 1]))
                results.append({
                    'data': data, 
                    'bbox': (x_min, y_min, x_max, y_max),
                    'source': 'opencv'
                })
        except Exception:
            pass
        
        return results

    def detect_and_decode(self, frame, use_qreader=False):
        """
        Detect and decode QR codes in the frame.
        Use GPU for valid resizing and color conversion.
        
        Args:
            frame: Input BGR frame
            use_qreader: Enable QReader fallback (slow, use only for final capture)
        
        Returns: (list of decoded objects, total count)
        """
        if frame is None:
            return [], 0
        
        start_total = time.perf_counter()
        
        all_results = []
        seen_texts = set()
        
        # Upload frame ONCE if using GPU, to avoid repeated uploads (Future opt)
        # For now, simplest path is to let _preprocess_frame_gpu handle it per scale
        
        # Multi-scale detection loop
        scales = self.scale_factors if self.use_multiscale else [1.0]
        
        for scale in scales:
            # === GPU ACCELERATED PREPROCESSING ===
            # Resizes and converts to Gray on GPU, then downloads small gray image
            gray = self._preprocess_frame_gpu(frame, scale)
            
            # --- 1. Pyzbar (Fast CPU) ---
            results = self._decode_pyzbar(gray)
            
            # --- 2. QReader (Slow, Optional) ---
            if not results and use_qreader and self.qreader_available:
                self._init_qreader()
                if self.qreader:
                     results.extend(self._decode_qreader(gray))
            
            # --- 3. OpenCV (Fallback) ---
            if not results and len(scales) == 1: # Only try opencv if strictly single scale
                 results.extend(self._decode_opencv(gray))

            # Process results for this scale
            scale_results = []
            for result in results:
                if result['data'] not in seen_texts:
                    seen_texts.add(result['data'])
                    
                    # Rescale bbox back to original size
                    if scale != 1.0:
                        x1, y1, x2, y2 = result['bbox']
                        result['bbox'] = (
                            int(x1 / scale), int(y1 / scale),
                            int(x2 / scale), int(y2 / scale)
                        )
                    scale_results.append(result)
            
            all_results.extend(scale_results)
            
            # Early exit: If we found results, don't keep searching other scales
            # (Unless we want to find ALL, but typically we want ANY valid QR quickly)
            # For distance, if we found it at 1.0, great. If not, try 0.75, etc.
            if all_results:
                break
        
        elapsed_total = (time.perf_counter() - start_total) * 1000
        
        # Update stats
        self._frame_count += 1
        self._total_time += elapsed_total
        
        if self._frame_count % 30 == 0:  # Log every 30 frames
            avg_time = self._total_time / self._frame_count
            logger.info(f"Avg detection time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS potential)")
            # self.gpu_processor.logger.info(f"GPU Active: {self.gpu_enabled}")
        
        return all_results, len(all_results)

    def detect_async(self, frame, callback=None, use_qreader=False):
        """Non-blocking detection."""
        if self._detection_running:
            return False
        
        self._detection_running = True
        
        def _detect_task():
            try:
                results, count = self.detect_and_decode(frame.copy(), use_qreader=use_qreader)
                with self._lock:
                    self._last_results = results
                if callback:
                    callback(results, count)
            finally:
                self._detection_running = False
        
        self._executor.submit(_detect_task)
        return True

    def get_latest_results(self):
        """Get results from last async detection."""
        with self._lock:
            return self._last_results.copy()

    def get_stats(self):
        """Get performance statistics."""
        if self._frame_count == 0:
            return {'avg_time_ms': 0, 'frame_count': 0}
        return {
            'avg_time_ms': self._total_time / self._frame_count,
            'frame_count': self._frame_count,
            'potential_fps': 1000 / (self._total_time / self._frame_count) if self._total_time > 0 else 0
        }

    def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down QRDetector")
        self._executor.shutdown(wait=False)


# Convenience functions
def detect_qr_standard(frame, use_qreader=False):
    """Detect QR codes in full frame."""
    detector = QRDetector()
    return detector.detect_and_decode(frame, use_qreader=use_qreader)


def detect_qr_advanced(image_path):
    """
    Advanced QR detection from file path.
    Uses QReader for thorough detection.
    """
    print(f"\n[ADVANCED] --------------------------------------------------")
    print(f"[ADVANCED] Starting Advanced Detection on: {image_path}")
    print(f"[ADVANCED] --------------------------------------------------")
    
    try:
        import cv2
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[ADVANCED] ERROR: Could not read image file")
            return [], 0
            
        print(f"[ADVANCED] Image loaded successfully: {frame.shape}")
        
        detector = QRDetector()
        print(f"[ADVANCED] calling detector.detect_and_decode(use_qreader=True)...")
        results, count = detector.detect_and_decode(frame, use_qreader=True)  # Enable QReader for saved images
        
        print(f"[ADVANCED] Detection finished. Total QRs found: {count}")
        if count > 0:
            print(f"[ADVANCED] Detected QR Data: {[r['data'] for r in results]}")
        else:
            print(f"[ADVANCED] No QR codes found in advanced mode.")
            
        print(f"[ADVANCED] --------------------------------------------------\n")
        return results, count
    except Exception as e:
        logger.error(f"detect_qr_advanced Error: {e}")
        print(f"[ADVANCED] EXCEPTION: {e}")
        return [], 0


def detect_composition(frame):
    """Placeholder for composition detection."""
    return {'Unknown': 0}


def get_gpu_status():
    """Return current GPU status."""
    return {
        'gpu_enabled': CUDA_AVAILABLE,
        'detection_mode': DETECTION_MODE,
        'pyzbar': PYZBAR_AVAILABLE,
        'qreader': QREADER_AVAILABLE
    }
