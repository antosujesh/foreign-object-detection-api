import cv2
import numpy as np
from config import PreprocessingConfig
from utils import get_logger

logger = get_logger("Preprocessing")

class ImagePreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Runs the preprocessing pipeline on an input image.
        1. Optional cropping of inspection area
        2. Resize to standard resolution
        3. Denoise using bilateral filter
        4. Normalize brightness and contrast using CLAHE (on LAB L-channel)
        """
        # 1. Optional cropping
        processed = img.copy()
        if self.config.crop_inspection_area:
            ymin, xmin, ymax, xmax = self.config.crop_coords
            h, w = processed.shape[:2]
            
            # Bounds checking
            ymin = max(0, min(ymin, h - 1))
            ymax = max(ymin + 1, min(ymax, h))
            xmin = max(0, min(xmin, w - 1))
            xmax = max(xmin + 1, min(xmax, w))
            
            # Only crop if valid coordinates are supplied
            if ymax > ymin and xmax > xmin:
                processed = processed[ymin:ymax, xmin:xmax]
                logger.info(f"Cropped inspection area to [{ymin}:{ymax}, {xmin}:{xmax}]")

        # 2. Resize preserving aspect ratio (based on target width)
        target_w, target_h = self.config.resize_resolution
        h, w = processed.shape[:2]
        dynamic_h = int(h * (target_w / w))
        # Ensure height is at least 2 pixels and even to prevent alignment errors
        dynamic_h = max(2, (dynamic_h // 2) * 2)
        processed = cv2.resize(processed, (target_w, dynamic_h), interpolation=cv2.INTER_LANCZOS4)
        logger.info(f"Resized image preserving aspect ratio: {target_w}x{dynamic_h} (Original: {w}x{h})")

        # 3. Denoise with Bilateral Filter
        # Bilateral filter is perfect for preserving sharp edges (necessary for difference detection)
        # while removing smooth surface noise/sensor grain.
        d = 9
        sigma_color = self.config.denoise_strength * 10
        sigma_space = self.config.denoise_strength * 10
        processed = cv2.bilateralFilter(processed, d, sigma_color, sigma_space)
        logger.info(f"Applied bilateral filter with denoise strength: {self.config.denoise_strength}")

        # 4. Normalize contrast and illumination via CLAHE in LAB color space
        # Converting to LAB and equalizing the 'L' (Lightness) channel avoids shifting color tones.
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        if self.config.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            logger.info("Applied CLAHE to Lightness channel")
        else:
            # Fallback to simple normalization if CLAHE is disabled
            l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
            logger.info("Applied MinMax normalization to Lightness channel")
            
        lab = cv2.merge((l, a, b))
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return processed
