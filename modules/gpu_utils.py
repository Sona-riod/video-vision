# gpu_utils.py - GPU Detection and Acceleration Utilities
"""
GPU Utilities for Palletization YOLO Detection System
Provides GPU detection, initialization, and acceleration utilities for NVIDIA GPUs.
"""

import cv2
import logging
import os
import sys

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
    'gpu_name': 'N/A',
    'gpu_memory': 'N/A',
    'torch_device': 'cpu',
    'opencv_cuda': False,
}

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
        self.cuda_device_count = 0
        self.cuda_device_name = "N/A"
        self.opencv_cuda_enabled = False
        self.cupy_available = False
        self.cuda_backend_available = False
        
        self._detect_gpu()
        self._initialized = True
    
    def _detect_gpu(self):
        """Detect GPU capabilities and log results."""
        logger = get_gpu_logger()
        
        print("\n" + "="*60)
        print("          GPU DETECTION AND INITIALIZATION")
        print("="*60)
        
        # 1. Check OpenCV CUDA support
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            self.cuda_device_count = cuda_count
            
            if cuda_count > 0:
                self.opencv_cuda_enabled = True
                self.cuda_available = True
                
                # Get device name
                cv2.cuda.setDevice(0)
                # device_props = cv2.cuda.getDevice() # getDevice returns int id, not props in some versions
                # self.cuda_device_name = f"CUDA Device {device_props}"
                
                print(f"[✓] OpenCV CUDA: ENABLED")
                print(f"    - CUDA Devices Found: {cuda_count}")
                logger.info(f"OpenCV CUDA enabled with {cuda_count} device(s)")
            else:
                print(f"[✗] OpenCV CUDA: NOT AVAILABLE")
                print(f"    - OpenCV compiled WITHOUT CUDA support")
                logger.warning("OpenCV CUDA not available - using CPU fallback")
                
        except Exception as e:
            print(f"[✗] OpenCV CUDA: ERROR - {e}")
            logger.error(f"OpenCV CUDA detection failed: {e}")
            
        # 2. Check PyTorch (Legacy support for existing code)
        try:
            import torch
            if torch.cuda.is_available():
                GPU_STATUS['cuda_available'] = True
                GPU_STATUS['gpu_name'] = torch.cuda.get_device_name(0)
                GPU_STATUS['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                GPU_STATUS['torch_device'] = 'cuda'
                
                self.cuda_device_name = GPU_STATUS['gpu_name']
                self.cuda_available = True # PyTorch confirms CUDA
                
                print(f"[✓] PyTorch CUDA: AVAILABLE")
                print(f"    - GPU Name: {GPU_STATUS['gpu_name']}")
            else:
                print(f"[!] PyTorch CUDA: NOT AVAILABLE")
        except ImportError:
            print("[!] PyTorch: NOT INSTALLED")
        except Exception as e:
            print(f"[!] PyTorch Error: {e}")

        # 3. Check for CuPy
        try:
            import cupy as cp
            self.cupy_available = True
            print(f"[✓] CuPy: AVAILABLE")
        except ImportError:
            print(f"[!] CuPy: NOT INSTALLED")
        except Exception as e:
            print(f"[!] CuPy: ERROR - {e}")
        
        # Summary
        print("-"*60)
        if self.cuda_available:
            print("[RESULT] GPU ACCELERATION: ENABLED")
        else:
            print("[RESULT] GPU ACCELERATION: DISABLED (CPU Fallback)")
        print("="*60 + "\n")
        
        # Sync legacy dict
        GPU_STATUS['opencv_cuda'] = self.opencv_cuda_enabled

    def print_status(self):
        """Print current GPU status."""
        print("\n--- GPU Status ---")
        print(f"CUDA Available: {self.cuda_available}")
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
    # Also return legacy dict for compatibility
    return GPU_INFO

def detect_gpu():
    """Legacy wrapper for existing code compatibility."""
    init_gpu()
    return GPU_STATUS

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    global GPU_INFO
    if GPU_INFO is None:
        GPU_INFO = GPUInfo()
    return GPU_INFO.cuda_available


class GPUImageProcessor:
    """GPU-accelerated image processing utilities."""
    
    def __init__(self):
        self.gpu_available = is_gpu_available()
        self.logger = get_gpu_logger()
        
        if self.gpu_available:
            self.logger.info("GPUImageProcessor initialized with CUDA acceleration")
        else:
            self.logger.info("GPUImageProcessor initialized with CPU fallback")
    
    def upload_to_gpu(self, frame):
        """Upload frame to GPU memory."""
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
            if isinstance(gpu_frame, cv2.cuda_GpuMat):
                return gpu_frame.download()
            return gpu_frame
        except Exception as e:
            self.logger.warning(f"GPU download failed: {e}")
            return gpu_frame
    
    def cvt_color_gpu(self, frame, code):
        """GPU-accelerated color conversion."""
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
            if isinstance(frame, cv2.cuda_GpuMat):
                frame = frame.download() # Ensure it's CPU mat
            return cv2.cvtColor(frame, code)
    
    def resize_gpu(self, frame, size):
        """GPU-accelerated resize. Returns GpuMat if input was GpuMat of if uploaded."""
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
            if isinstance(frame, cv2.cuda_GpuMat):
                frame = frame.download()
            return cv2.resize(frame, size)
    
    def gaussian_blur_gpu(self, frame, ksize=(5, 5)):
        """GPU-accelerated Gaussian blur."""
        if not self.gpu_available:
            return cv2.GaussianBlur(frame, ksize, 0)
        
        try:
            if not isinstance(frame, cv2.cuda_GpuMat):
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
            else:
                gpu_frame = frame
            
            # Create Gaussian filter
            gaussian_filter = cv2.cuda.createGaussianFilter(
                gpu_frame.type(), -1, ksize, 0
            )
            result = gaussian_filter.apply(gpu_frame)
            return result
        except Exception as e:
            self.logger.warning(f"GPU GaussianBlur failed, using CPU: {e}")
            if isinstance(frame, cv2.cuda_GpuMat):
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

# Export legacy functions for compatibility if any
def warm_up_gpu():
    """Legacy warmup."""
    pass
