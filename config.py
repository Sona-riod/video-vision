# config.py
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
SAVE_FOLDER = BASE_DIR / "keg_frames"
DB_PATH = BASE_DIR / "keg_detection.db"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
BATCH_MEMORY_FILE = BASE_DIR / "last_batch.txt"

# Create directories
SAVE_FOLDER.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)



# Jetson Orin Configuration (Active)
CAMERA_CONFIG = {
    'type': 'v4l2',    # Jetson typically uses v4l2
    'device': 10,       # Default to 0 (Video0) - Adjust if using specific ID like 10
    'width': 1408,     # ICAM-540 max resolution
    'height': 1080,
    'fps': 30,
}


# ========== CAMERA HARDWARE INIT (Advantech REST API) ==========
CAMERA_INIT_ENABLED   = True                  # False = skip the camera-init splash (dev)
CAMERA_API_BASE_URL   = "http://localhost:5000"
CAMERA_INIT_WIDTH     = 3840                   # 4K, matches camera_configure.sh
CAMERA_INIT_HEIGHT    = 2160
CAMERA_INIT_TIMEOUT   = 10                     # per-request timeout (s)
# Per-step "settle" waits (s) — mirror the sleeps in camera_configure.sh
CAMERA_INIT_SETTLE_AFTER_CLOSE = 5
CAMERA_INIT_SETTLE_AFTER_OPEN  = 5
CAMERA_INIT_SETTLE_AFTER_PLAY  = 5
CAMERA_INIT_SETTLE_AFTER_RESET = 2


# ========== API CONFIGURATION ==========
BASE_URL = "https://api-dev.checkology-cloud.io"
API_ENDPOINT = f"{BASE_URL}/api/kegs/fillingareaupdatecamera"
BEER_TYPES_ENDPOINT = f"{BASE_URL}/api/kegs/cam/beer-types"
CAMERA_NAME = "ICAM-540"
CAMERA_MAC_ID = "3C:6D:66:01:5A:F0"
CAMERA_SERIAL = "icam-540"
API_TIMEOUT = 10
API_MAX_RETRIES = 3
SSL_VERIFY = False  # Set to True in production
ENABLE_PAYLOAD_HASH = True

# Cloud sync settings
CLOUD_CONFIG_ENDPOINT = f"{BASE_URL}/api/current-config"
CLOUD_SYNC_INTERVAL = 30


DEFAULT_KEG_COUNT = 6
MAX_KEG_COUNT = 20
MIN_KEG_COUNT = 1
STABILITY_THRESHOLD = 3

# FOV Validation
FOV_BOUNDARY_RATIO = 0.9
MIN_OCCLUSION_THRESHOLD = 0.7

# Model Configuration
QR_MODEL_PATH = MODELS_DIR / "model_qr" / "best.pt"
QR_CONF_THRESHOLD = 0.5

# Pallet Status options
PALLET_STATUS = ["CREATED"]

# ========== GPU CONFIGURATION ==========
GPU_CONFIG = {
    'device': 0,                    # CUDA device ID (0 = first GPU)
    'half_precision': False,        # FP16 might not be supported on CPU or some GPUs
    'memory_fraction': 0.8,         # Max GPU memory fraction to use
    'warmup_enabled': False,        # Warm up GPU at startup
    'force_cpu': False,             # ENABLE GPU for Jetson
}

# Advanced QR Detection
TILE_SIZE = (1280, 960)
OVERLAP_RATIO = 0.2
SCALE_FACTORS = [1.0, 1.2, 1.5]
MIN_CROP_SIZE = 50
MIN_UPSCALE_SIZE = 100

# Retry Configuration
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_MINUTES = [1, 2, 4, 8, 16]
RETRY_CHECK_INTERVAL = 60
NETWORK_CHECK_INTERVAL = 30

# Alarm System Configuration
ALARM_BLINK_INTERVAL = 0.5
ENABLE_PHYSICAL_ALERTS = False
FOV_ENABLED = True # Enabled for production/Jetsons

# Application Settings
MAX_FOLDER_SIZE_MB = 500

# Color Scheme for HMI
COLOR_SCHEME = {
    'bg_light': (1, 1, 1, 1),
    'panel_bg': (0.95, 0.95, 0.95, 1),
    'highlight': (0, 0.3, 0.6, 1),
    'text_dark': (0, 0, 0, 1),
    'alert_red': (0.8, 0.2, 0.2, 1),
    'status_green': (0.2, 0.7, 0.3, 1),
    'status_orange': (0.9, 0.6, 0.2, 1)
}
