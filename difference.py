import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import DifferenceConfig
from utils import get_logger

logger = get_logger("Difference")

class DifferenceDetector:
    def __init__(self, config: DifferenceConfig):
        self.config = config

    def detect_differences(self, ref_img: np.ndarray, warped_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes differences using multiple methods and combines them.
        Returns:
            - Combined grayscale difference map (0-255)
            - Thresholded binary mask (0-255)
        """
        # Convert images to grayscale for intensity-based differences
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

        # 1. SSIM Difference Map
        # ssim returns score and full ssim image where 1.0 means identical.
        # We invert it so differences are bright (1.0 - ssim).
        _, ssim_img = ssim(ref_gray, warped_gray, full=True)
        ssim_diff = (1.0 - ssim_img) * 255
        ssim_diff = np.clip(ssim_diff, 0, 255).astype(np.uint8)

        # 2. Absolute Difference Map
        abs_diff = cv2.absdiff(ref_gray, warped_gray)

        # 3. LAB Delta-E Color Difference Map
        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        warped_lab = cv2.cvtColor(warped_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate Euclidean distance in LAB color space
        # This is a solid approximation of perceptual Delta-E color difference
        lab_diff = np.sqrt(
            (ref_lab[:, :, 0] - warped_lab[:, :, 0]) ** 2 +
            (ref_lab[:, :, 1] - warped_lab[:, :, 1]) ** 2 +
            (ref_lab[:, :, 2] - warped_lab[:, :, 2]) ** 2
        )
        # Normalize to 0-255
        lab_diff = cv2.normalize(lab_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 4. Edge Difference Map
        # Using Sobel filter to get edges, which is more robust to slight alignment offsets than Canny
        sobel_x_ref = cv2.Sobel(ref_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_ref = cv2.Sobel(ref_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_ref = np.sqrt(sobel_x_ref**2 + sobel_y_ref**2)
        
        sobel_x_warped = cv2.Sobel(warped_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_warped = cv2.Sobel(warped_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_warped = np.sqrt(sobel_x_warped**2 + sobel_y_warped**2)
        
        edge_diff = cv2.absdiff(edge_ref.astype(np.uint8), edge_warped.astype(np.uint8))
        edge_diff = cv2.normalize(edge_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 5. Combine difference maps using configuration weights
        w_ssim = self.config.weight_ssim
        w_abs = self.config.weight_abs_diff
        w_lab = self.config.weight_lab_diff
        w_edge = self.config.weight_edge_diff

        # Normalize weights just in case
        total_weight = w_ssim + w_abs + w_lab + w_edge
        if total_weight > 0:
            w_ssim /= total_weight
            w_abs /= total_weight
            w_lab /= total_weight
            w_edge /= total_weight

        combined_diff = (
            w_ssim * ssim_diff +
            w_abs * abs_diff +
            w_lab * lab_diff +
            w_edge * edge_diff
        ).astype(np.uint8)

        # Mask out regions where the warped image has no data (black borders due to alignment warping)
        # This prevents the boundary of warping from appearing as a massive foreign object.
        invalid_warp_mask = warped_gray == 0
        # Dilate invalid mask slightly to ignore alignment edge artifacts
        kernel_invalid = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        invalid_warp_mask = cv2.dilate(invalid_warp_mask.astype(np.uint8), kernel_invalid) > 0
        
        combined_diff[invalid_warp_mask] = 0

        # 6. Thresholding
        thresh_method = self.config.threshold_method.upper()
        if thresh_method == "OTSU":
            _, binary = cv2.threshold(combined_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logger.info("Applied Otsu thresholding")
        elif thresh_method == "ADAPTIVE":
            # Adaptive threshold requires odd block size
            block_size = self.config.adaptive_block_size
            if block_size % 2 == 0:
                block_size += 1
            binary = cv2.adaptiveThreshold(
                combined_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, self.config.adaptive_c
            )
            logger.info(f"Applied Adaptive thresholding (block_size={block_size}, c={self.config.adaptive_c})")
        else: # STATIC
            _, binary = cv2.threshold(combined_diff, self.config.static_threshold_val, 255, cv2.THRESH_BINARY)
            logger.info(f"Applied Static thresholding (val={self.config.static_threshold_val})")

        # In case adaptive thresholding inverts the background (which is mostly 0), check and fix it
        # If more than 50% of the image is white, invert it
        if np.sum(binary == 255) > (binary.size * 0.5):
            binary = cv2.bitwise_not(binary)

        # 7. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_kernel_size, self.config.morph_kernel_size))
        
        # Opening: removes small isolated noise pixels
        if self.config.open_iterations > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=self.config.open_iterations)
            
        # Closing: bridges small breaks, merges close fragments
        if self.config.close_iterations > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.config.close_iterations)

        # 8. Fill Holes
        # Find contours and draw them filled on a new mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(binary)
        cv2.drawContours(filled_mask, contours, -1, 255, -1)
        
        logger.info(f"Post-processing completed. Found {len(contours)} initial contour blobs.")

        return combined_diff, filled_mask
