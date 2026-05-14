# modules/advanced.py
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from qreader import QReader
import shutil
import logging
from config import QR_MODEL_PATH, TILE_SIZE, OVERLAP_RATIO, SCALE_FACTORS, MIN_CROP_SIZE, MIN_UPSCALE_SIZE

# Setup logger
logger = logging.getLogger(__name__)

class AdvancedQRDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or str(QR_MODEL_PATH)
        self.qreader = QReader(model_size='s', min_confidence=0.5)  
        self.slice_width, self.slice_height = TILE_SIZE
        self.overlap_width_ratio = OVERLAP_RATIO
        self.overlap_height_ratio = OVERLAP_RATIO
        self.scale_factors = SCALE_FACTORS
        self.min_crop_size = MIN_CROP_SIZE
        self.min_upscale_size = MIN_UPSCALE_SIZE
    
    def tile_with_overlap(self, image_path, output_dir):
        """Tile image with overlap - Matches your initial logic exactly."""
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        
        logger.info(f"Advanced: Original image size: {img_w}x{img_h}")

        overlap_w = int(self.slice_width * self.overlap_width_ratio)
        overlap_h = int(self.slice_height * self.overlap_height_ratio)

        step_x = max(self.slice_width - overlap_w, 1)
        step_y = max(self.slice_height - overlap_h, 1)
        
        logger.info(f"Advanced: Tile size: {self.slice_width}x{self.slice_height}")
        logger.info(f"Advanced: Overlap: {overlap_w}x{overlap_h} (steps: {step_x}x{step_y})")

        tile_positions = self._generate_tile_positions(img_w, img_h, step_x, step_y)
        
        logger.info(f"Advanced: Total tiles to generate: {len(tile_positions)}")
        
        tile_count = 0
        for x, y in tile_positions:
            if self._save_tile(image, x, y, img_w, img_h, output_dir, tile_count):
                tile_count += 1

        logger.info(f"Advanced: Saved {tile_count} tiles to: {os.path.abspath(output_dir)}")
        return tile_count

    def _generate_tile_positions(self, img_w, img_h, step_x, step_y):
        tile_positions = []
        
        y = 0
        while y < img_h:
            x = 0
            while x < img_w:
                tile_positions.append((x, y))
                x += step_x
                if x >= img_w:
                    break
            y += step_y
            if y >= img_h:
                break
        
        # Add right and bottom edge tiles
        self._add_edge_tiles(tile_positions, img_w, img_h, step_x, step_y)
        
        return sorted(list(set(tile_positions)))

    def _add_edge_tiles(self, tile_positions, img_w, img_h, step_x, step_y):
        if img_w > self.slice_width:
            right_x = max(0, img_w - self.slice_width)
            for y in range(0, img_h, step_y):
                tile_positions.append((right_x, y))
        
        if img_h > self.slice_height:
            bottom_y = max(0, img_h - self.slice_height)
            for x in range(0, img_w, step_x):
                tile_positions.append((x, bottom_y))
        
        if img_w > self.slice_width and img_h > self.slice_height:
            corner_pos = (max(0, img_w - self.slice_width), max(0, img_h - self.slice_height))
            tile_positions.append(corner_pos)

    def _save_tile(self, image, x, y, img_w, img_h, output_dir, tile_count):
        x_end = min(x + self.slice_width, img_w)
        y_end = min(y + self.slice_height, img_h)
        
        actual_width = x_end - x
        actual_height = y_end - y
        
        if actual_width < self.slice_width * 0.25 or actual_height < self.slice_height * 0.25:
            return False
        
        box = (x, y, x_end, y_end)
        tile = image.crop(box)

        if tile.size != (self.slice_width, self.slice_height):
            padded = Image.new("RGB", (self.slice_width, self.slice_height), (0, 0, 0))  # Black padding as in your initial
            padded.paste(tile, (0, 0))
            tile = padded

        filename = f"tile_{tile_count:03d}_x{x}_y{y}.png"
        tile.save(os.path.join(output_dir, filename))
        return True

    def unblur_image(self, image):
        """Unblur - Matches your initial logic exactly."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        sharpen_kernel = np.array([[-1,-1,-1], 
                                 [-1, 9,-1], 
                                 [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        
        gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        combined = cv2.addWeighted(sharpened, 0.7, unsharp_mask, 0.3, 0)
        
        if len(image.shape) == 3:
            combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        
        return combined

    def decode_qr_pyzbar(self, image):
        """Pyzbar decode with enhancements - Matches your initial logic exactly."""
        try:
            decoded_objects = pyzbar.decode(image, symbols=[ZBarSymbol.QRCODE])
            if decoded_objects:
                return decoded_objects, True
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            decoded_objects = pyzbar.decode(gray, symbols=[ZBarSymbol.QRCODE])
            if decoded_objects:
                return decoded_objects, True
                
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            decoded_objects = pyzbar.decode(adaptive_thresh, symbols=[ZBarSymbol.QRCODE])
            if decoded_objects:
                return decoded_objects, True
                
            kernel = np.ones((3,3), np.uint8)
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            decoded_objects = pyzbar.decode(morph, symbols=[ZBarSymbol.QRCODE])
            
            return decoded_objects, len(decoded_objects) > 0
            
        except Exception as e:
            logger.error(f"Advanced: Error decoding QR code with pyzbar: {e}")
            return [], False

    def detect_and_crop_qr_yolo(self, image_path, output_dir):
        """YOLO detect/crop - Matches your initial logic (loads model per call, but efficient for fallback)."""
        model = YOLO(self.model_path)  # As in your initial - loads fresh for isolation
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Advanced: Image not found: {image_path}")

        results = model.predict(image, conf=0.3, iou=0.5, verbose=False)
        os.makedirs(output_dir, exist_ok=True)

        cropped_paths = []
        detection_count = len(results[0].boxes) if results[0].boxes else 0

        logger.info(f"Advanced: Found {detection_count} QR code(s) in {os.path.basename(image_path)}")

        if detection_count == 0:
            return cropped_paths, detection_count

        for i, box in enumerate(results[0].boxes):
            confidence = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            qr_width, qr_height = x2 - x1, y2 - y1
            padding = max(10, min(qr_width, qr_height) // 10)
            
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
            
            qr_crop = image[y1:y2, x1:x2]
            
            if qr_crop.shape[0] < self.min_crop_size or qr_crop.shape[1] < self.min_crop_size:
                continue
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for scale in self.scale_factors:
                new_w = int(qr_crop.shape[1] * scale)
                new_h = int(qr_crop.shape[0] * scale)
                
                if new_w < self.min_upscale_size or new_h < self.min_upscale_size:
                    continue
                    
                scaled_qr = cv2.resize(qr_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                unblurred_qr = self.unblur_image(scaled_qr)
                
                output_path = os.path.join(output_dir, f"{base_name}_qr_{i}_scale_{scale}.png")
                unblurred_path = os.path.join(output_dir, f"{base_name}_qr_{i}_scale_{scale}_unblurred.png")
                
                cv2.imwrite(output_path, scaled_qr)
                cv2.imwrite(unblurred_path, unblurred_qr)
                cropped_paths.extend([output_path, unblurred_path])

        return cropped_paths, detection_count

    def decode_qr_qreader(self, image_path):
        """QReader decode - Matches your initial logic exactly."""
        image = cv2.imread(image_path)
        if image is None:
            return []

        try:
            decoded_qrs = self.qreader.detect_and_decode(image=image)
            return [qr.strip() for qr in decoded_qrs if qr and len(qr.strip()) > 0]
        except Exception as e:
            logger.error(f"Advanced: QReader error for {image_path}: {e}")
            return []

    def detect_advanced(self, image_path, temp_dir="advanced_temp"):
        """Main advanced detection - Matches your initial logic exactly (renamed from advanced_qr_detection for consistency)."""
        logger.info("ADVANCED QR DETECTION ACTIVATED")
        
        # Create temporary directories
        tiles_dir = os.path.join(temp_dir, "tiles")
        cropped_dir = os.path.join(temp_dir, "cropped")
        
        # Clean up previous runs
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        os.makedirs(tiles_dir, exist_ok=True)
        os.makedirs(cropped_dir, exist_ok=True)
        
        unique_qr_codes = set()
        
        try:
            # Step 1: Tile the image
            logger.info("Advanced: Step 1 - Tiling image with overlap...")
            tile_count = self.tile_with_overlap(image_path, tiles_dir)
            
            if tile_count == 0:
                logger.warning("Advanced: No tiles generated, skipping advanced detection")
                return list(unique_qr_codes)
            
            # Step 2: Process each tile
            all_cropped_paths, total_detections = self._process_tiles(tiles_dir, cropped_dir)
            logger.info(f"Advanced: Total detections across tiles: {total_detections}")
            
            # Step 3: Decode all cropped QR images
            self._decode_cropped_images(all_cropped_paths, unique_qr_codes)
            logger.info(f"Advanced: Decoding completed. Unique QR codes found: {len(unique_qr_codes)}")
            
        except Exception as e:
            logger.error(f"Advanced: Error in advanced detection pipeline: {e}")
        
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Advanced: Temporary files cleaned up")
        
        return list(unique_qr_codes)

    def _process_tiles(self, tiles_dir, cropped_dir):
        logger.info(f"Advanced: Step 2 - Processing tiles...")
        all_cropped_paths = []
        total_detections = 0
        
        tile_files = sorted([f for f in os.listdir(tiles_dir) if f.lower().endswith(".png")])
        logger.info(f"Advanced: Processing {len(tile_files)} tiles...")

        for tile_file in tile_files:
            tile_path = os.path.join(tiles_dir, tile_file)
            cropped_paths, detections = self.detect_and_crop_qr_yolo(tile_path, cropped_dir)
            total_detections += detections
            all_cropped_paths.extend(cropped_paths)
        
        return all_cropped_paths, total_detections

    def _decode_cropped_images(self, all_cropped_paths, unique_qr_codes):
        logger.info("Advanced: Step 3 - Decoding QR codes...")
        
        for cropped_img in all_cropped_paths:
            self._process_single_crop(cropped_img, unique_qr_codes)

        return unique_qr_codes

    def _process_single_crop(self, cropped_img, unique_qr_codes):
        # Try QReader first
        qr_results = self.decode_qr_qreader(cropped_img)
        for qr in qr_results:
            self._add_unique_qr(qr, unique_qr_codes, "QReader")
            
        # Try pyzbar as backup
        self._detect_with_pyzbar_fallback(cropped_img, unique_qr_codes)

    def _detect_with_pyzbar_fallback(self, cropped_img, unique_qr_codes):
        image = cv2.imread(cropped_img)
        if image is None:
            return

        decoded_objects, success = self.decode_qr_pyzbar(image)
        if success:
            for obj in decoded_objects:
                qr_data = obj.data.decode('utf-8').strip()
                if qr_data:
                    self._add_unique_qr(qr_data, unique_qr_codes, "pyzbar")

    def _add_unique_qr(self, qr_text, unique_qr_codes, source):
        if qr_text not in unique_qr_codes:
            unique_qr_codes.add(qr_text)
            logger.info(f"Advanced: {source} decoded: {qr_text[:50]}{'...' if len(qr_text) > 50 else ''}")

# Standalone function for external use (as in your initial)
def run_advanced_detection(image_path, model_path=None):
    detector = AdvancedQRDetector(model_path)
    return detector.detect_advanced(image_path)

if __name__ == "__main__":
    image_path = input("Enter image path for advanced detection test: ").strip()
    if os.path.exists(image_path):
        results = run_advanced_detection(image_path)
        print(f"Advanced detection found {len(results)} QR codes:")
        for i, qr in enumerate(results, 1):
            print(f"{i}. {qr}")
    else:
        print("Image not found!")