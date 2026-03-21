# config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
SAVE_FOLDER = BASE_DIR / "keg_frames"
DB_PATH = BASE_DIR / "keg_detection.db"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
SAVE_FOLDER.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)



# Jetson Orin / ICAM-540 (Production) - Uncomment when deploying to Jetson
CAMERA_CONFIG = {
     'type': 'v4l2',    # Jetson typically uses v4l2
     'device': 10,       # ICAM-540 device ID
     'width': 1408,     # ICAM-540 max resolution
     'height': 1080,
     'fps': 30,
 }


# ========== API CONFIGURATION ==========
API_ENDPOINT = "https://api2.checkology-cloud.io/api/kegs/fillingareaupdatecamera"
BEER_TYPES_ENDPOINT = "https://api2.checkology-cloud.io/api/kegs/cam/beer-types"
CAMERA_NAME = "ICAM-540"
CAMERA_MAC_ID = "3C:6D:66:01:5A:F0"
API_TIMEOUT = 10
API_MAX_RETRIES = 3
SSL_VERIFY = False  # Set to True in production
ENABLE_PAYLOAD_HASH = True

# Cloud sync settings
CLOUD_CONFIG_ENDPOINT = f"{API_ENDPOINT}/api/current-config"
CLOUD_SYNC_INTERVAL = 30


DEFAULT_KEG_COUNT = 6
MAX_KEG_COUNT = 20
MIN_KEG_COUNT = 1
STABILITY_THRESHOLD = 5

# Model Configuration
QR_MODEL_PATH = MODELS_DIR / "model_qr" / "best.pt"
QR_CONF_THRESHOLD = 0.5

# QR Detection Tuning (detector.py)
QR_CROP_PADDING = 15       # Pixels around detected QR
QR_MIN_CROP_SIZE = 100     # Minimum crop for upscaling
QR_UPSCALE_FACTOR = 2      # Upscale multiplier for small crops
QREADER_MODEL_SIZE = 's'   # QReader model: 's' or 'l'
DETECTOR_WORKERS = 2       # QRDetector thread pool

# Advanced QR Detection Tuning (advanced.py)
ADV_YOLO_CONF = 0.3
ADV_YOLO_IOU = 0.5

# Pallet Status options
PALLET_STATUS = ["CREATED"]

# ========== GPU CONFIGURATION ==========
GPU_CONFIG = {
    'device': 10,                    # CUDA device ID (0 = first GPU)
    'half_precision': False,        # FP16 might not be supported on CPU or some GPUs
    'memory_fraction': 0.8,         # Max GPU memory fraction to use
    'warmup_enabled': False,        # Warm up GPU at startup
    'force_cpu': True,              # ENABLE GPU for Jetson
}

# Advanced QR Detection
TILE_SIZE = (1280, 960)
OVERLAP_RATIO = 0.2
SCALE_FACTORS = [1.0, 1.2, 1.5]
MIN_CROP_SIZE = 50
MIN_UPSCALE_SIZE = 100

# Database
DB_TIMEOUT = 60            # SQLite connection timeout (seconds)

# Process Worker
PROCESS_WORKERS = 4        # Background batch thread pool
SHUTDOWN_TIMEOUT = 30      # Seconds to wait on worker shutdown

# Recovery
RECOVERY_STUCK_TIMEOUT_MIN = 10   # Minutes before batch is "stuck"
RETRY_CLEANUP_DAYS = 7            # Days before old retries are purged

# Retry Configuration
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_MINUTES = [1, 2, 4, 8, 16]
RETRY_CHECK_INTERVAL = 60
NETWORK_CHECK_INTERVAL = 30

# UI Color Scheme for HMI (main.py)
UI_COLORS = {
    'accent_blue':   (0.157, 0.306, 0.780, 1),
    'accent_teal':   (0.031, 0.682, 0.612, 1),
    'accent_green':  (0.133, 0.694, 0.298, 1),
    'accent_amber':  (0.961, 0.647, 0.098, 1),
    'accent_red':    (0.863, 0.196, 0.184, 1),
    'text_dark':     (0.129, 0.145, 0.196, 1),
    'text_med':      (0.384, 0.408, 0.471, 1),
    'text_light':    (0.667, 0.686, 0.733, 1),
    'text_white':    (1, 1, 1, 1),
    'bg_app':        (0.953, 0.957, 0.965, 1),
    'bg_card':       (1, 1, 1, 1),
    'divider':       (0.882, 0.890, 0.910, 1),
    'disabled_btn':  (0.820, 0.839, 0.863, 1),
    'status_tint':   (0.937, 0.965, 0.976, 1),
}