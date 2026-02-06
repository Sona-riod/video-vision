
import cv2
import time
import sys
import os

# Add modules to path
sys.path.append(os.getcwd())

from modules.camera import CameraManager
from modules.gpu_utils import detect_gpu

def test_camera():
    print("Testing Camera Configuration...")
    
    # Check GPU/Device status first
    print("Checking GPU/Device status...")
    detect_gpu()
    
    # Initialize Camera
    print("Initializing CameraManager...")
    cam = CameraManager()
    
    if not cam.start():
        print("Failed to start camera!")
        return
    
    print("Camera started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cam.get_frame()
            
            if ret and frame is not None:
                cv2.imshow("Camera Test - Press 'q' to quit", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera Test Finished.")

if __name__ == "__main__":
    test_camera()
