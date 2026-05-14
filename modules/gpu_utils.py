# gpu_utils.py - GPU Detection and Acceleration Utilities
"""
GPU Utilities for Palletization YOLO Detection System
Provides GPU detection, initialization, and acceleration utilities.
Supports:
- NVIDIA CUDA (Linux/Windows/Jetson)
- Apple Metal/MPS (macOS)
"""

import cv2
import logging
import os
import sys
import platform

# Initialize logger
_gpu_logger = None

def get_gpu_logger() -> logging.Logger:
    global _gpu_logger
    if _gpu_logger is None:
        _gpu_logger = logging.getLogger("GPU_Utils")
        _gpu_logger.setLevel(logging.INFO)
        if not _gpu_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - [GPU] - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            _gpu_logger.addHandler(handler)
        _gpu_logger.propagate = False
    return _gpu_logger

# ========== GPU DETECTION RESULTS ==========
GPU_STATUS = {
    'cuda_available': False,
    'mps_available': False,  # Mac Metal Performance Shaders
    'gpu_name': 'N/A',
    'gpu_memory': 'N/A',
    'torch_device': 'cpu',
    'opencv_cuda': False,
    'platform': platform.system(),
}

# Check if running on Mac
IS_MAC = platform.system() == 'Darwin'


class GPUInfo:
    """Singleton class to store GPU information and status."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.cuda_available = False
        self.mps_available = False  # Mac Metal
        self.cuda_device_count = 0
        self.cuda_device_name = "N/A"
        self.opencv_cuda_enabled = False
        self.cupy_available = False
        self.cuda_backend_available = False
        self.is_mac = IS_MAC
        
        self._detect_gpu()
        self._initialized = True
    
    def _detect_gpu(self):
        """Detect GPU capabilities and log results."""
        logger = get_gpu_logger()
        
        print("\n" + "="*60)
        print("          GPU DETECTION AND INITIALIZATION")
        print(f"          Platform: {platform.system()} ({platform.machine()})")
        print("="*60)
        
        # === MAC: Check for Metal/MPS support ===
        if self.is_mac:
            self._detect_mac_gpu(logger)
        else:
            # === LINUX/WINDOWS: Check OpenCV CUDA support ===
            self._detect_cuda_opencv(logger)
            # Check PyTorch CUDA
            self._detect_pytorch_cuda(logger)
            # Check CuPy
            self._detect_cupy(logger)
        
        # Summary
        self._print_summary(logger)

    def _detect_mac_gpu(self, logger):
        """Detect Mac Metal GPU capabilities."""
        print("[Mac] Checking Metal/MPS GPU support...")
        
        # OpenCV on Mac doesn't use CUDA
        print("[!] OpenCV CUDA: NOT APPLICABLE (Mac uses Metal)")
        self.opencv_cuda_enabled = False
        
        # Check for Metal via PyTorch MPS
        try:
            import torch
            if self._is_mps_available(torch):
                self._enable_mps(logger)
            else:
                self._handle_mps_unavailable(logger)
        except ImportError:
            print("PyTorch: NOT INSTALLED")
        except Exception as e:
            print("MPS Detection Error: {e}")

    def _is_mps_available(self, torch):
        """Check if PyTorch MPS is available."""
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    def _enable_mps(self, logger):
        """Enable MPS and update status."""
        self.mps_available = True
        GPU_STATUS['mps_available'] = True
        GPU_STATUS['torch_device'] = 'mps'
        GPU_STATUS['gpu_name'] = 'Apple Metal GPU'
        
        # Get Mac GPU info via subprocess
        gpu_name = self._get_mac_gpu_name_from_system()
        if gpu_name:
            GPU_STATUS['gpu_name'] = gpu_name
            self.cuda_device_name = gpu_name
        
        print("PyTorch MPS (Metal): AVAILABLE")
        print("GPU: {GPU_STATUS['gpu_name']}")
        logger.info(f"Mac Metal GPU enabled: {GPU_STATUS['gpu_name']}")

    def _handle_mps_unavailable(self, logger):
        """Handle case where MPS is not available."""
        print("PyTorch MPS: NOT AVAILABLE")
        print("    - Requires macOS 12.3+ and Apple Silicon or AMD GPU")
        logger.warning("MPS not available - using CPU")

    def _get_mac_gpu_name_from_system(self):
        """Retrieve Mac GPU name using system_profiler."""
        try:
            import subprocess
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if 'Chipset Model' in line or 'Chip' in line:
                    return line.split(':')[-1].strip()
        except Exception:
            pass
        return None

    def _detect_cuda_opencv(self, logger):
        """Detect OpenCV CUDA support (Linux/Windows)."""
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            self.cuda_device_count = cuda_count
            
            if cuda_count > 0:
                self.opencv_cuda_enabled = True
                self.cuda_available = True
                
                cv2.cuda.setDevice(0)
                
                print("OpenCV CUDA: ENABLED")
                print(f"CUDA Devices Found: {cuda_count}")
                logger.info(f"OpenCV CUDA enabled with {cuda_count} device(s)")
            else:
                print("OpenCV CUDA: NOT AVAILABLE")
                print("OpenCV compiled WITHOUT CUDA support")
                logger.warning("OpenCV CUDA not available - using CPU fallback")
                
        except Exception as e:
            print(f"OpenCV CUDA: ERROR - {e}")
            logger.error(f"OpenCV CUDA detection failed: {e}")

    def _detect_pytorch_cuda(self, logger):
        """Detect PyTorch CUDA support (non-Mac systems)."""
        try:
            import torch
            
            if torch.cuda.is_available():
                GPU_STATUS['cuda_available'] = True
                GPU_STATUS['gpu_name'] = torch.cuda.get_device_name(0)
                GPU_STATUS['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                GPU_STATUS['torch_device'] = 'cuda'
                
                self.cuda_device_name = GPU_STATUS['gpu_name']
                self.cuda_available = True
                
                print("PyTorch CUDA: AVAILABLE")
                print(f"GPU Name: {GPU_STATUS['gpu_name']}")
            else:
                print("PyTorch CUDA: NOT AVAILABLE")
        except ImportError:
            print("PyTorch: NOT INSTALLED")
        except Exception as e:
            print(f"[!] PyTorch Error: {e}")

    def _detect_cupy(self, logger):
        """Detect CuPy availability (CUDA only)."""
        try:
            import cupy as cp
            self.cupy_available = True
            print("CuPy: AVAILABLE")
        except ImportError:
            print("CuPy: NOT INSTALLED")
        except Exception as e:
            print(f"CuPy: ERROR - {e}")

    def _print_summary(self, logger):
        """Print GPU detection summary."""
        print("-"*60)
        if self.is_mac and self.mps_available:
            print("[RESULT] GPU ACCELERATION: ENABLED (Apple Metal/MPS)")
        elif self.cuda_available:
            print("[RESULT] GPU ACCELERATION: ENABLED (NVIDIA CUDA)")
        else:
            print("[RESULT] GPU ACCELERATION: DISABLED (CPU Fallback)")
        print("="*60 + "\n")
        
        # Sync legacy dict
        GPU_STATUS['opencv_cuda'] = self.opencv_cuda_enabled

    def print_status(self):
        """Print current GPU status."""
        print("\n--- GPU Status ---")
        print(f"Platform: {platform.system()}")
        print(f"CUDA Available: {self.cuda_available}")
        print(f"MPS Available: {self.mps_available}")
        print(f"Device Name: {self.cuda_device_name}")
        print(f"OpenCV CUDA: {self.opencv_cuda_enabled}")
        print("-"*20)


# Global GPU info instance
GPU_INFO = None

def init_gpu():
    """Initialize GPU and return GPU info."""
    global GPU_INFO
    if GPU_INFO is None:
        GPU_INFO = GPUInfo()
    return GPU_INFO

def detect_gpu():
    """Legacy wrapper for existing code compatibility."""
    init_gpu()
    return GPU_STATUS

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available (CUDA or MPS)."""
    global GPU_INFO
    if GPU_INFO is None:
        GPU_INFO = GPUInfo()
    return GPU_INFO.cuda_available or GPU_INFO.mps_available

def get_torch_device() -> str:
    """Get the appropriate PyTorch device string."""
    global GPU_INFO
    if GPU_INFO is None:
        GPU_INFO = GPUInfo()
    if GPU_INFO.mps_available:
        return 'mps'
    elif GPU_INFO.cuda_available:
        return 'cuda'
    return 'cpu'


class GPUImageProcessor:
    """
    GPU-accelerated image processing utilities.
    Uses CUDA for Linux/Windows, CPU fallback for Mac (OpenCV doesn't support Metal).
    Note: For Mac, PyTorch operations can still use MPS, but OpenCV uses CPU.
    """
    
    def __init__(self):
        global GPU_INFO
        if GPU_INFO is None:
            GPU_INFO = GPUInfo()
        
        self.is_mac = IS_MAC
        # On Mac, OpenCV uses CPU (no cv2.cuda), but PyTorch can use MPS
        # For image processing with OpenCV, we use CPU on Mac
        self.gpu_available = GPU_INFO.opencv_cuda_enabled  # Only CUDA counts for OpenCV ops
        self.mps_available = GPU_INFO.mps_available
        self.logger = get_gpu_logger()
        
        if self.gpu_available:
            self.logger.info("GPUImageProcessor initialized with CUDA acceleration")
        elif self.is_mac:
            self.logger.info("GPUImageProcessor initialized (Mac - OpenCV uses CPU, PyTorch can use MPS)")
        else:
            self.logger.info("GPUImageProcessor initialized with CPU fallback")
    
    def upload_to_gpu(self, frame):
        """Upload frame to GPU memory (CUDA only)."""
        if not self.gpu_available:
            return frame
        
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            return gpu_frame
        except Exception as e:
            self.logger.warning(f"GPU upload failed: {e}")
            return frame
    
    def download_from_gpu(self, gpu_frame):
        """Download frame from GPU memory."""
        if not self.gpu_available:
            return gpu_frame
        
        try:
            if hasattr(cv2, 'cuda_GpuMat') and isinstance(gpu_frame, cv2.cuda_GpuMat):
                return gpu_frame.download()
            return gpu_frame
        except Exception as e:
            self.logger.warning(f"GPU download failed: {e}")
            return gpu_frame
    
    def cvt_color_gpu(self, frame, code):
        """GPU-accelerated color conversion (CUDA) or CPU fallback."""
        if not self.gpu_available:
            return cv2.cvtColor(frame, code)
        
        try:
            if not isinstance(frame, cv2.cuda_GpuMat):
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
            else:
                gpu_frame = frame
            
            result = cv2.cuda.cvtColor(gpu_frame, code)
            return result
        except Exception as e:
            self.logger.warning(f"GPU cvtColor failed, using CPU: {e}")
            if hasattr(cv2, 'cuda_GpuMat') and isinstance(frame, cv2.cuda_GpuMat):
                frame = frame.download()
            return cv2.cvtColor(frame, code)
    
    def resize_gpu(self, frame, size):
        """GPU-accelerated resize (CUDA) or CPU fallback."""
        if not self.gpu_available:
            return cv2.resize(frame, size)
        
        try:
            if not isinstance(frame, cv2.cuda_GpuMat):
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
            else:
                gpu_frame = frame
            
            result = cv2.cuda.resize(gpu_frame, size)
            return result
        except Exception as e:
            self.logger.warning(f"GPU resize failed, using CPU: {e}")
            if hasattr(cv2, 'cuda_GpuMat') and isinstance(frame, cv2.cuda_GpuMat):
                frame = frame.download()
            return cv2.resize(frame, size)
    
    def gaussian_blur_gpu(self, frame, ksize=(5, 5)):
        """GPU-accelerated Gaussian blur (CUDA) or CPU fallback."""
        if not self.gpu_available:
            return cv2.GaussianBlur(frame, ksize, 0)
        
        try:
            if not isinstance(frame, cv2.cuda_GpuMat):
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
            else:
                gpu_frame = frame
            
            gaussian_filter = cv2.cuda.createGaussianFilter(
                gpu_frame.type(), -1, ksize, 0
            )
            result = gaussian_filter.apply(gpu_frame)
            return result
        except Exception as e:
            self.logger.warning(f"GPU GaussianBlur failed, using CPU: {e}")
            if hasattr(cv2, 'cuda_GpuMat') and isinstance(frame, cv2.cuda_GpuMat):
                frame = frame.download()
            return cv2.GaussianBlur(frame, ksize, 0)


# Create global image processor
GPU_PROCESSOR = None

def get_gpu_processor() -> GPUImageProcessor:
    """Get the global GPU image processor instance."""
    global GPU_PROCESSOR
    if GPU_PROCESSOR is None:
        GPU_PROCESSOR = GPUImageProcessor()
    return GPU_PROCESSOR

# Export legacy functions for compatibility
def warm_up_gpu():
    """Legacy warmup."""
    pass