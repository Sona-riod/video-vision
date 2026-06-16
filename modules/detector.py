# modules/detector.py - Optimized QR Code Detection with YOLO + Enhanced Crop Decoding
"""
Fast QR Detector using YOLO for localization + Pyzbar for decoding.
Optimized for:
- Long-distance QR detection via YOLO model
- Fast crop-only decoding (no full-frame scanning)
- Enhanced preprocessing fallbacks (upscale + CLAHE + sharpening)
"""
import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from pathlib import Path

# Setup debug logger
logger = logging.getLogger("QRDetector")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)

# ===== INITIALIZATION =====
print("\n" + "="*60)
print("INITIALIZING QR DETECTOR (YOLO + ENHANCED DECODE)")
print("="*60)

# Import Pyzbar (primary decoder - fast)
PYZBAR_AVAILABLE = False
pyzbar_decode = None
ZBarSymbol = None
try:
    from pyzbar.pyzbar import decode as _pyzbar_decode, ZBarSymbol as _ZBarSymbol
    pyzbar_decode = _pyzbar_decode
    ZBarSymbol = _ZBarSymbol
    PYZBAR_AVAILABLE = True
    print("[DETECTOR] Pyzbar: LOADED")
except ImportError as e:
    print(f"[DETECTOR] Pyzbar: NOT AVAILABLE ({e})")

# Import YOLO
YOLO_AVAILABLE = False
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
    YOLO_AVAILABLE = True
    print("[DETECTOR] YOLO: LOADED")
except Exception as e:
    print(f"[DETECTOR] YOLO: NOT AVAILABLE ({e})")
    print("[DETECTOR] Warning: Running without YOLO (Deep Learning) support.")

# QReader (lazy loaded - only when needed for deep scan)
QREADER_AVAILABLE = False
QReaderClass = None
try:
    from qreader import QReader as _QReaderClass
    QReaderClass = _QReaderClass
    QREADER_AVAILABLE = True
    print("[DETECTOR] QReader: AVAILABLE (lazy-loaded on capture)")
except ImportError as e:
    print(f"[DETECTOR] QReader: NOT AVAILABLE ({e})")

# Import config
try:
    from config import QR_MODEL_PATH
except ImportError:
    from pathlib import Path
    QR_MODEL_PATH = Path(__file__).parent.parent / "models" / "model_qr" / "best.pt"

# YOLO inference size — large enough to localize small/distant QR codes in a 4K frame.
try:
    from config import YOLO_IMGSZ
except ImportError:
    YOLO_IMGSZ = 1280

print("="*60)
print("DETECTOR STATUS:")
print(f"   Pyzbar: {'OK' if PYZBAR_AVAILABLE else 'FAILED'}")
print(f"   YOLO: {'OK' if YOLO_AVAILABLE else 'FAILED'}")
print(f"   QReader: {'OK (lazy)' if QREADER_AVAILABLE else 'NOT AVAILABLE'}")
print(f"   Model Path: {QR_MODEL_PATH}")
print("="*60 + "\n")


class QRDetector:
    """
    Fast QR Detector using YOLO for localization + Pyzbar for decoding.
    
    Pipeline:
    1. YOLO detects QR code bounding boxes (works at distance)
    2. Crop detected regions with padding
    3. Enhanced decode: Pyzbar -> Upscale+Pyzbar -> CLAHE+Pyzbar -> Sharpen+Pyzbar
    """
    
    def __init__(self):
        self.pyzbar_available = PYZBAR_AVAILABLE
        self.yolo_available = YOLO_AVAILABLE
        
        # Initialize YOLO model for QR detection
        self.model = None
        if self.yolo_available:
            try:
                model_path = str(QR_MODEL_PATH)
                logger.info(f"Loading YOLO model from: {model_path}")
                self.model = YOLO(model_path)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.exception("Failed to load YOLO model")
                self.model = None

        # Inference device for YOLO: GPU if CUDA is available, else CPU.
        self.device = 0
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else 'cpu'
        except Exception:
            self.device = 'cpu'
        logger.info(f"YOLO inference device={self.device}, imgsz={YOLO_IMGSZ}")

        # OpenCV QR detector as last resort fallback
        self.cv_detector = cv2.QRCodeDetector()
        
        # Threading for async detection
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="QRDetect")
        self._lock = threading.Lock()
        self._last_results = []
        self._detection_running = False
        
        # QReader (lazy initialized - only on first capture)
        self.qreader = None
        self._qreader_initialized = False
        self.qreader_available = QREADER_AVAILABLE
        
        # Performance stats
        self._frame_count = 0
        self._total_time = 0
        
        # Configuration
        self.crop_padding = 15  # Pixels to add around detected QR
        self.min_crop_size = 100  # Minimum crop size for upscaling
        self.upscale_factor = 2  # How much to upscale small crops
        self.confidence_threshold = 0.5  # YOLO confidence threshold
        
        logger.info(f"QRDetector initialized - YOLO: {self.model is not None}, Pyzbar: {self.pyzbar_available}, QReader: {self.qreader_available}")
    
    def _init_qreader(self):
        """Lazy initialize QReader only when needed (first capture)"""
        if self._qreader_initialized:
            return self.qreader
        
        if self.qreader_available and QReaderClass is not None:
            try:
                logger.info("Initializing QReader model (one-time, may take a moment)...")
                start = time.perf_counter()
                self.qreader = QReaderClass(model_size='s', min_confidence=0.5)
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(f"QReader initialized in {elapsed:.0f}ms")
            except Exception as e:
                logger.exception("QReader init failed")
                self.qreader_available = False
                self.qreader = None
        
        self._qreader_initialized = True
        return self.qreader
    
    def _decode_qreader(self, image):
        """Decode QR codes using QReader (ML-based, slower but more robust)."""
        if self.qreader is None:
            return []
        
        results = []
        try:
            start = time.perf_counter()
            texts = self.qreader.detect_and_decode(image=image)
            elapsed = (time.perf_counter() - start) * 1000
            
            if texts:
                for text in texts:
                    if text:
                        h, w = image.shape[:2]
                        results.append({
                            'data': text, 
                            'bbox': (0, 0, w, h),
                            'source': 'qreader'
                        })
            
            if len(results) > 0:
                logger.info(f"QReader found {len(results)} QR codes in {elapsed:.1f}ms")
                
        except Exception as e:
            logger.warning(f"QReader error: {e}")
        
        return results

    def _decode_pyzbar(self, image):
        """Decode QR codes using Pyzbar (fast)."""
        if not self.pyzbar_available or pyzbar_decode is None:
            return []
        
        results = []
        try:
            # Ensure grayscale for Pyzbar
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            decoded = pyzbar_decode(gray, symbols=[ZBarSymbol.QRCODE])
            
            for obj in decoded:
                text = obj.data.decode("utf-8")
                if text:
                    rect = obj.rect
                    bbox = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)
                    results.append({'data': text, 'bbox': bbox, 'source': 'pyzbar'})
        except Exception as e:
            logger.warning(f"Pyzbar error: {e}")
        
        return results
    
    def _enhanced_decode(self, crop):
        """
        Enhanced decode with multiple preprocessing attempts.
        Tries: Raw -> Upscale -> CLAHE -> Sharpen
        """
        # 1. Try original crop first (fastest)
        results = self._decode_pyzbar(crop)
        if results:
            return results
        
        # Convert to grayscale for preprocessing
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        h, w = gray.shape[:2]
        
        # 2. Upscale if crop is small (for distance detection)
        if w < self.min_crop_size or h < self.min_crop_size:
            upscaled = cv2.resize(gray, (w * self.upscale_factor, h * self.upscale_factor), 
                                   interpolation=cv2.INTER_CUBIC)
            results = self._decode_pyzbar(upscaled)
            if results:
                return results
            gray = upscaled  # Use upscaled for further processing
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results = self._decode_pyzbar(enhanced)
        if results:
            return results
        
        # 4. Sharpening kernel
        kernel = np.array([[-1, -1, -1], 
                          [-1,  9, -1], 
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        results = self._decode_pyzbar(sharpened)
        if results:
            return results
        
        # 5. Binary threshold as last resort
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = self._decode_pyzbar(binary)
        
        return results
    
    def _decode_opencv(self, image):
        """Fallback: Decode using OpenCV's QRCodeDetector."""
        results = []
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            data, points, _ = self.cv_detector.detectAndDecode(gray)
            
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
        Uses YOLO for localization, then Pyzbar for decoding crops.
        
        Args:
            frame: Input BGR frame
            use_qreader: Enable QReader deep scan (slower, use only for capture)
        
        Returns: (list of decoded objects, total count)
        """
        if frame is None:
            return [], 0
            
        start_total = time.perf_counter()
        
        # Initialize QReader if needed for capture mode
        self._init_qreader_if_needed(use_qreader)
        
        all_results = []
        seen_texts = set()
        
        # 1. YOLO-based detection
        if self.model is not None:
             self._run_yolo_detection(frame, use_qreader, all_results, seen_texts)
        
        # 2. Fallbacks (Pyzbar -> OpenCV -> QReader)
        if not all_results:
             self._run_fallbacks(frame, use_qreader, all_results, seen_texts)
        
        elapsed_total = (time.perf_counter() - start_total) * 1000
        
        # Update stats
        self._update_stats(elapsed_total, len(all_results), use_qreader)
        
        return all_results, len(all_results)

    def _init_qreader_if_needed(self, use_qreader):
        if use_qreader and self.qreader_available and not self._qreader_initialized:
            logger.info("[CAPTURE MODE] Initializing QReader for deep scan...")
            self._init_qreader()

    def _run_yolo_detection(self, frame, use_qreader, all_results, seen_texts):
        try:
            # Run YOLO inference. imgsz is large so small QRs survive in 4K; device pins GPU.
            yolo_results = self.model(frame, verbose=False, conf=self.confidence_threshold,
                                      imgsz=YOLO_IMGSZ, device=self.device)
            
            h, w = frame.shape[:2]
            
            for result in yolo_results:
                for box in result.boxes:
                    self._process_yolo_box(box, frame, h, w, use_qreader, all_results, seen_texts)
        except Exception as e:
            logger.exception("YOLO detection error")

    def _process_yolo_box(self, box, frame, h, w, use_qreader, all_results, seen_texts):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Add padding for better decoding
        pad = self.crop_padding
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w, x2 + pad)
        crop_y2 = min(h, y2 + pad)
        
        # Crop the region
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if crop.size > 0:
            self._decode_crop(crop, use_qreader, (x1, y1, x2, y2), all_results, seen_texts)

    def _decode_crop(self, crop, use_qreader, bbox, all_results, seen_texts):
        # Try enhanced decode on the crop
        decoded = self._enhanced_decode(crop)
        
        # If capture mode and Pyzbar failed, try QReader on crop
        if not decoded and use_qreader and self.qreader is not None:
             logger.info("[CAPTURE] Pyzbar failed, trying QReader on crop...")
             decoded = self._decode_qreader(crop)
        
        for result_item in decoded:
            text = result_item['data']
            if text and text not in seen_texts:
                seen_texts.add(text)
                # Map bbox back to original frame coordinates but keep source info
                all_results.append({
                    'data': text,
                    'bbox': bbox,
                    'source': result_item.get('source', '')
                })

    def _run_fallbacks(self, frame, use_qreader, all_results, seen_texts):
        # Try direct Pyzbar decode (fast, works for large visible QRs)
        if not all_results:
            self._collect_results(self._decode_pyzbar(frame), all_results, seen_texts)
            
        # Try OpenCV QR detector
        if not all_results:
            self._collect_results(self._decode_opencv(frame), all_results, seen_texts)
            
        # Final Fallback: QReader on full frame (only in capture mode)
        if not all_results and use_qreader and self.qreader is not None:
            logger.info("[CAPTURE] All methods failed, trying QReader on full frame...")
            self._collect_results(self._decode_qreader(frame), all_results, seen_texts)

    def _collect_results(self, source_results, all_results, seen_texts):
        for result in source_results:
            if result['data'] not in seen_texts:
                seen_texts.add(result['data'])
                all_results.append(result)

    def _update_stats(self, elapsed_total, count, use_qreader):
        self._frame_count += 1
        self._total_time += elapsed_total
        
        if self._frame_count % 30 == 0:  # Log every 30 frames
            avg_time = self._total_time / self._frame_count
            logger.info(f"Avg detection time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS potential)")
        
        if use_qreader:
            logger.info(f"[CAPTURE] Detection complete: {count} QRs in {elapsed_total:.0f}ms")

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
        """Cleanup resources and free GPU memory."""
        logger.info("Shutting down QRDetector")
        self._executor.shutdown(wait=False)
        if self.model is not None:
            del self.model
            self.model = None
        if self.qreader is not None:
            del self.qreader
            self.qreader = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except Exception:
            pass


# Convenience functions (kept for compatibility)
def detect_qr_standard(frame, use_qreader=False):
    """Detect QR codes in full frame."""
    detector = QRDetector()
    return detector.detect_and_decode(frame, use_qreader=use_qreader)


def detect_qr_advanced(image_path):
    """
    Advanced QR detection from file path.
    """
    print("\n[ADVANCED] --------------------------------------------------")
    print(f"[ADVANCED] Starting Advanced Detection on: {image_path}")
    print("[ADVANCED] --------------------------------------------------")
    
    try:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print("[ADVANCED] ERROR: Could not read image file")
            return [], 0
            
        print(f"[ADVANCED] Image loaded successfully: {frame.shape}")
        
        detector = QRDetector()
        results, count = detector.detect_and_decode(frame)
        
        print(f"[ADVANCED] Detection finished. Total QRs found: {count}")
        if count > 0:
            print(f"[ADVANCED] Detected QR Data: {[r['data'] for r in results]}")
        else:
            print("[ADVANCED] No QR codes found.")
            
        print("[ADVANCED] --------------------------------------------------\n")
        return results, count
    except Exception as e:
        logger.exception("detect_qr_advanced Error")
        print(f"[ADVANCED] EXCEPTION: {e}")
        return [], 0


def detect_composition(frame):
    """Placeholder for composition detection."""
    return {'Unknown': 0}


def get_gpu_status():
    """Return current GPU status."""
    return {
        'gpu_enabled': False,  # GPU preprocessing removed for speed
        'detection_mode': 'YOLO + Pyzbar',
        'pyzbar': PYZBAR_AVAILABLE,
        'yolo': YOLO_AVAILABLE
    }