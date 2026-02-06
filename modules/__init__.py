# modules/__init__.py
# Lazy imports to avoid torch/torchvision compatibility issues at startup

__all__ = [
    'CameraManager',
    'QRDetector',
    'DatabaseManager',
    'APISender',
    'AdvancedQRDetector',
    'setup_logging',
    'manage_storage',
    'save_default_keg_count',
    'submit_batch',
    'shutdown',
    'get_active_tasks',
    'run_advanced_detection',
    'recover_system',
    'check_database_integrity',
    'ReportGenerator'
]

# Import non-problematic modules directly
from .camera import CameraManager
from .database import DatabaseManager
from .api_sender import APISender
from .utils import setup_logging, manage_storage, save_default_keg_count
from .process_worker import submit_batch, shutdown, get_active_tasks
from .recovery import recover_system, check_database_integrity
from .reports import ReportGenerator

# Lazy load detector modules to handle torch/torchvision issues
_detector_module = None
_advanced_module = None

def _get_detector_module():
    global _detector_module
    if _detector_module is None:
        from . import detector as _detector_module
    return _detector_module

def _get_advanced_module():
    global _advanced_module
    if _advanced_module is None:
        from . import advanced as _advanced_module
    return _advanced_module

# These will be resolved lazily when accessed
def __getattr__(name):
    if name == 'QRDetector':
        return _get_detector_module().QRDetector
    elif name == 'AdvancedQRDetector':
        return _get_advanced_module().AdvancedQRDetector
    elif name == 'run_advanced_detection':
        return _get_advanced_module().run_advanced_detection
    raise AttributeError(f"module 'modules' has no attribute '{name}'")