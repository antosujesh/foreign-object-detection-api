from PIL import Image
from pathlib import Path
from pydantic import BaseModel, Field
from skimage.metrics import structural_similarity as skssim
from skimage.metrics import structural_similarity as ssim
from typing import List, Optional, Tuple, Dict, Any
from typing import Tuple, List, Optional
from typing import Tuple, Optional
from typing import Union
from ultralytics import YOLO
import cv2
import logging
import numpy as np
import os
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms



# ============================================================
# Extracted from config.py
# ============================================================

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for directory in [UPLOADS_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class PreprocessingConfig(BaseModel):
    crop_inspection_area: bool = Field(default=False, description="Whether to crop a specific inspection area")
    crop_coords: list[int] = Field(default=[0, 0, 0, 0], description="Bounding box [ymin, xmin, ymax, xmax] to crop if crop_inspection_area is true")
    resize_resolution: tuple[int, int] = Field(default=(1024, 1024), description="Standard resolution to resize images for inspection")
    apply_clahe: bool = Field(default=True, description="Apply CLAHE for contrast normalization")
    denoise_strength: float = Field(default=3.0, description="Denoise strength for Bilateral filter")

class AlignmentConfig(BaseModel):
    method: str = Field(default="SIFT", description="Feature detection method: SIFT, ORB, AKAZE")
    max_features: int = Field(default=2000, description="Maximum number of features to detect")
    ransac_threshold: float = Field(default=5.0, description="RANSAC outlier threshold")
    min_matches: int = Field(default=10, description="Minimum number of matches to attempt alignment")
    min_align_score: float = Field(default=0.4, description="Minimum acceptable alignment score (ratio of inliers or correlation)")

class DifferenceConfig(BaseModel):
    weight_ssim: float = Field(default=0.35, description="Weight of SSIM difference map")
    weight_abs_diff: float = Field(default=0.25, description="Weight of Absolute intensity difference map")
    weight_lab_diff: float = Field(default=0.25, description="Weight of LAB Delta-E color difference map")
    weight_edge_diff: float = Field(default=0.15, description="Weight of Canny edge difference map")
    
    # Thresholding
    threshold_method: str = Field(default="OTSU", description="Threshold method: OTSU, ADAPTIVE, STATIC")
    static_threshold_val: int = Field(default=30, description="Static threshold value (0-255) if method is STATIC")
    adaptive_block_size: int = Field(default=25, description="Block size for adaptive thresholding")
    adaptive_c: int = Field(default=5, description="C constant subtracted from mean in adaptive thresholding")
    
    # Morphological operations
    morph_kernel_size: int = Field(default=5, description="Kernel size for morphology operations")
    open_iterations: int = Field(default=1, description="Number of opening operations to remove noise")
    close_iterations: int = Field(default=2, description="Number of closing operations to bridge gaps")

class RegionConfig(BaseModel):
    min_area: int = Field(default=400, description="Minimum area in pixels for a valid candidate region")
    max_area: int = Field(default=500000, description="Maximum area in pixels for a valid candidate region")
    min_solidity: float = Field(default=0.5, description="Minimum solidity (contour_area / convex_hull_area)")
    min_circularity: float = Field(default=0.2, description="Minimum circularity (4*pi*area / perimeter^2)")
    max_aspect_ratio: float = Field(default=5.0, description="Maximum aspect ratio (width/height or height/width)")
    merge_distance_threshold: int = Field(default=75, description="Distance in pixels to merge close contours")

class ValidationConfig(BaseModel):
    enable_dl_validation: bool = Field(default=True, description="Enable deep learning validation of candidate regions")
    dl_validator_type: str = Field(default="yolo", description="DL validation method: 'deep_features' (ResNet/MobileNet cosine similarity) or 'yolo' (YOLOv8 object recognition)")
    cnn_model_name: str = Field(default="mobilenet_v3_small", description="PyTorch model for deep feature extraction: mobilenet_v3_small, resnet18")
    deep_feature_threshold: float = Field(default=0.75, description="Maximum feature similarity score above which difference is considered shadow/lighting change (and rejected)")
    yolo_model_path: str = Field(default="yolov8n.pt", description="Path to YOLOv8 model weights (will download if not present)")
    yolo_conf: float = Field(default=0.60, description="Confidence threshold for YOLO validation")
    ignore_border_pixels: int = Field(default=10, description="Number of pixels from border to ignore detections")

class DetectionConfig(BaseModel):
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    difference: DifferenceConfig = DifferenceConfig()
    region: RegionConfig = RegionConfig()
    validation: ValidationConfig = ValidationConfig()

DEFAULT_CONFIG = DetectionConfig()


# ============================================================
# Extracted from utils.py
# ============================================================

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "fod_api.log", encoding="utf-8")
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name."""
    return logging.getLogger(name)

logger = get_logger("Utils")

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Safely loads an image in BGR format."""
    path_str = str(path)
    img = cv2.imread(path_str)
    if img is None:
        raise ValueError(f"Could not read image from path: {path_str}")
    return img

def save_image(img: np.ndarray, path: Union[str, Path]) -> bool:
    """Safely saves an image."""
    path_str = str(path)
    # Ensure directory exists
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(path_str, img)
    if not success:
        logger.error(f"Failed to save image to: {path_str}")
    return success

def create_diff_heatmap(diff_map: np.ndarray) -> np.ndarray:
    """Converts a single-channel grayscale difference map (0-255) to a JET color heatmap."""
    # Ensure image is uint8
    if diff_map.dtype != np.uint8:
        diff_map = (diff_map * 255).astype(np.uint8) if diff_map.max() <= 1.0 else diff_map.astype(np.uint8)
    
    # Apply JET colormap
    heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    return heatmap

def draw_detections(image: np.ndarray, detections: list[dict], mask: np.ndarray = None) -> np.ndarray:
    """
    Draws bounding boxes, labels, confidence scores, and optional segmentation mask
    outlines on the image.
    """
    output = image.copy()
    
    # Overlay semi-transparent mask if provided
    # Disabled per user request to only show bounding boxes
    if False and mask is not None:
        # Create a colored overlay (e.g., semi-transparent red)
        overlay = output.copy()
        overlay[mask > 0] = [0, 0, 255] # Red overlay
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        
        # Add contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 0, 255), 2)
        
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        conf = det.get("confidence", 1.0)
        label = det.get("label", "foreign_object")
        
        # Draw bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Format label text
        txt = f"{label} ({conf:.2f})"
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x, y - text_h - 8), (x + text_w, y), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(output, txt, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
    return output

def clean_old_files(directory: Path, max_age_seconds: int = 3600):
    """Deletes files in a directory that are older than max_age_seconds."""
    try:
        current_time = time.time()
        for file_path in directory.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path.name}")
    except Exception as e:
        logger.error(f"Error cleaning directory {directory}: {e}")


# ============================================================
# Extracted from preprocessing.py
# ============================================================

logger = logging.getLogger("Standalone")

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


# ============================================================
# Extracted from alignment.py
# ============================================================


logger = logging.getLogger("Standalone")


class ImageAligner:
    def __init__(self, config: AlignmentConfig):
        self.config = config
        self._init_detector()

    def _init_detector(self):
        """Initializes the keypoint detector and matcher based on configuration."""
        method = self.config.method.upper()
        logger.info(f"Initializing feature detector: {method}")

        if method == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=self.config.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif method == "AKAZE":
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == "ORB":
            self.detector = cv2.ORB_create(nfeatures=self.config.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            logger.warning(f"Unknown alignment method: {method}. Defaulting to SIFT.")
            self.detector = cv2.SIFT_create(nfeatures=self.config.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def align(self, ref_img: np.ndarray, curr_img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Aligns the current image to the reference image.
        Multi-strategy approach:
          1. SIFT/ORB/AKAZE + RANSAC Homography (primary)
          2. ECC (Enhanced Correlation Coefficient) refinement
          3. Simple resize fallback if structural features match
        Accepts alignment even at lower scores when RANSAC inlier count is high.
        """
        # Strategy 1: Feature-based homography
        warped, score, inliers = self._align_core(ref_img, curr_img)

        # Accept if we have >= 10 inliers (mathematically robust warp found)
        if warped is not None and inliers >= 10:
            logger.info(f"Direct alignment accepted: score={score:.3f}, inliers={inliers}")
            # Ensure score reflects inlier quality even if NCC is low
            accepted_score = max(score, min(0.5, inliers / 100.0))
            return warped, accepted_score

        # Strategy 1b: Try 180-degree rotation
        logger.info(f"Trying 180° rotation (direct: score={score:.3f}, inliers={inliers})")
        curr_rotated = cv2.rotate(curr_img, cv2.ROTATE_180)
        warped_rot, score_rot, inliers_rot = self._align_core(ref_img, curr_rotated)

        if warped_rot is not None and inliers_rot >= 10:
            accepted_score = max(score_rot, min(0.5, inliers_rot / 100.0))
            logger.info(f"180° rotation alignment accepted: score={score_rot:.3f}, inliers={inliers_rot}")
            return warped_rot, accepted_score

        # Strategy 2: Simple resize as ultimate fallback (same scene, different crop)
        # Resize curr to match ref dimensions — valid for linear camera scans
        logger.info("Falling back to simple resize alignment (same-scene linear scan assumption)")
        h_ref, w_ref = ref_img.shape[:2]
        resized_curr = cv2.resize(curr_img, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)

        # Score the resize using structural correlation on grayscale
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        res_gray = cv2.cvtColor(resized_curr, cv2.COLOR_BGR2GRAY)
        ncc = self._ncc(ref_gray, res_gray)
        resize_score = max(0.1, float(ncc))  # Guarantee a minimum passing score
        logger.info(f"Resize fallback NCC score: {resize_score:.3f}")

        # Always return resize warped — even if NCC is low, it's better than None
        return resized_curr, resize_score

    def _ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Normalized Cross-Correlation between two grayscale images."""
        f1 = img1.astype(np.float32).ravel()
        f2 = img2.astype(np.float32).ravel()
        mean1, mean2 = np.mean(f1), np.mean(f2)
        std1, std2 = np.std(f1) + 1e-6, np.std(f2) + 1e-6
        return float(np.mean((f1 - mean1) * (f2 - mean2)) / (std1 * std2))

    def _align_core(self, ref_img: np.ndarray, curr_img: np.ndarray) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Core alignment: SIFT/ORB feature matching + RANSAC homography.
        Returns (warped_image, score, inlier_count).
        """
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to normalize lighting before feature extraction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ref_gray_eq = clahe.apply(ref_gray)
        curr_gray_eq = clahe.apply(curr_gray)

        kp_ref, desc_ref = self.detector.detectAndCompute(ref_gray_eq, None)
        kp_curr, desc_curr = self.detector.detectAndCompute(curr_gray_eq, None)

        if desc_ref is None or desc_curr is None or len(kp_ref) < 4 or len(kp_curr) < 4:
            logger.warning("Insufficient keypoints for alignment.")
            return None, 0.0, 0

        method = self.config.method.upper()
        good_matches = []

        try:
            if method in ["SIFT"] and len(kp_ref) > 5 and len(kp_curr) > 5:
                matches = self.matcher.knnMatch(desc_curr, desc_ref, k=2)
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
            else:
                matches = self.matcher.match(desc_curr, desc_ref)
                good_matches = sorted(matches, key=lambda x: x.distance)[:150]
        except Exception as e:
            logger.error(f"Matching error: {e}")
            return None, 0.0, 0

        num_matches = len(good_matches)
        if num_matches < self.config.min_matches:
            logger.warning(f"Too few matches: {num_matches} < {self.config.min_matches}")
            return None, 0.0, 0

        pts_curr = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_curr, pts_ref, cv2.RANSAC, self.config.ransac_threshold)

        if H is None or mask is None:
            logger.warning("Homography computation failed.")
            return None, 0.0, 0

        inliers_count = int(np.sum(mask))
        inlier_ratio = float(inliers_count) / float(num_matches) if num_matches > 0 else 0.0

        h, w = ref_img.shape[:2]
        warped_curr = cv2.warpPerspective(
            curr_img, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Compute NCC only in warped region
        warped_gray = cv2.cvtColor(warped_curr, cv2.COLOR_BGR2GRAY)
        valid_mask = warped_gray > 0
        valid_pixels = int(np.sum(valid_mask))
        total_pixels = warped_gray.size

        if valid_pixels > 0.1 * total_pixels:
            # Sufficient valid warped area
            ref_vals = ref_gray_eq[valid_mask].astype(np.float32)
            warp_vals = warped_gray[valid_mask].astype(np.float32)
            mean_r, mean_w = np.mean(ref_vals), np.mean(warp_vals)
            std_r = np.std(ref_vals) + 1e-6
            std_w = np.std(warp_vals) + 1e-6
            ncc_score = max(0.0, float(np.mean((ref_vals - mean_r) * (warp_vals - mean_w)) / (std_r * std_w)))
        else:
            # Warp result is mostly empty → NCC not reliable
            ncc_score = inlier_ratio  # Use inlier ratio as proxy

        alignment_score = 0.4 * inlier_ratio + 0.6 * ncc_score
        logger.info(
            f"Alignment core: matches={num_matches}, inliers={inliers_count} "
            f"(ratio={inlier_ratio:.3f}), NCC={ncc_score:.3f}, combined={alignment_score:.3f}, "
            f"valid_coverage={valid_pixels/total_pixels:.2%}"
        )

        return warped_curr, alignment_score, inliers_count

    def select_best_reference(
        self, ref_images: List[np.ndarray], curr_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Evaluates multiple reference images and selects the best one.
        Returns: (best_ref, warped_curr, best_score, best_idx)
        """
        best_score = -1.0
        best_ref = None
        best_warped = None
        best_idx = -1

        logger.info(f"Evaluating {len(ref_images)} reference images.")

        for idx, ref_img in enumerate(ref_images):
            try:
                warped_curr, score = self.align(ref_img, curr_image)
                if warped_curr is not None and score > best_score:
                    best_score = score
                    best_ref = ref_img
                    best_warped = warped_curr
                    best_idx = idx
                    logger.info(f"Ref {idx}: score={score:.3f} (new best)")
                else:
                    logger.info(f"Ref {idx}: score={score:.3f}")
            except Exception as e:
                logger.error(f"Error aligning reference {idx}: {e}")
                continue

        if best_idx == -1:
            # Ultimate fallback: use first reference with simple resize
            logger.warning("All references failed alignment. Using first reference with resize fallback.")
            h_ref, w_ref = ref_images[0].shape[:2]
            resized = cv2.resize(curr_image, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
            return ref_images[0], resized, 0.1, 0

        logger.info(f"Best reference: index={best_idx}, score={best_score:.3f}")
        return best_ref, best_warped, best_score, best_idx


# ============================================================
# Extracted from difference.py
# ============================================================


logger = logging.getLogger("Standalone")

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


# ============================================================
# Extracted from segmentation.py
# ============================================================

logger = logging.getLogger("Standalone")

class RegionProposer:
    def __init__(self, config: RegionConfig):
        self.config = config

    def propose_regions(self, binary_mask: np.ndarray) -> list[dict]:
        """
        Extracts candidate foreign object regions from the binary mask.
        Filters candidate regions based on shape and size constraints.
        Merges close or overlapping candidate bounding boxes.
        Returns:
            - A list of dictionary objects representing the candidate regions:
              [{"x": x, "y": y, "width": w, "height": h, "area": area, "contour": contour, "mask": region_mask}]
        """
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 1. Filter by area
            if area < self.config.min_area or area > self.config.max_area:
                continue
                
            # Compute geometric parameters
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # 2. Filter by aspect ratio
            aspect_ratio = max(float(w) / float(h), float(h) / float(w)) if h > 0 and w > 0 else 0.0
            if aspect_ratio > self.config.max_aspect_ratio:
                continue

            # 3. Filter by solidity (ratio of contour area to its convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0
            if solidity < self.config.min_solidity:
                continue

            # 4. Filter by circularity
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
            if circularity < self.config.min_circularity:
                continue

            # If all checks pass, it's a valid candidate
            # Create a mask for this specific region
            region_mask = np.zeros_like(binary_mask)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            
            candidates.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": int(area),
                "solidity": solidity,
                "circularity": circularity,
                "contour": contour,
                "mask": region_mask
            })
            
        logger.info(f"Proposed {len(candidates)} regions after geometric filtering (out of {len(contours)} initial contours).")
        
        # Merge close bounding boxes
        if len(candidates) > 1:
            merged_candidates = self._merge_close_boxes(candidates, self.config.merge_distance_threshold, binary_mask.shape)
            logger.info(f"Merged regions count: {len(candidates)} -> {len(merged_candidates)}")
            
            final_candidates = []
            for cand in merged_candidates:
                w, h = cand["width"], cand["height"]
                aspect_ratio = max(float(w) / float(h), float(h) / float(w)) if h > 0 and w > 0 else 0.0
                if aspect_ratio > self.config.max_aspect_ratio:
                    continue
                if cand["solidity"] < self.config.min_solidity:
                    continue
                if cand["circularity"] < self.config.min_circularity:
                    continue
                final_candidates.append(cand)
            
            logger.info(f"After post-merge geometric filtering: {len(final_candidates)} candidates remain.")
            return final_candidates
            
        return candidates

    def _merge_close_boxes(self, candidates: list[dict], threshold: int, img_shape: tuple) -> list[dict]:
        """
        Merges bounding boxes that are close to each other.
        """
        def boxes_are_close(box1, box2, dist_thresh):
            # Calculate distance between boundaries
            x1_min, y1_min = box1["x"], box1["y"]
            x1_max, y1_max = x1_min + box1["width"], y1_min + box1["height"]
            
            x2_min, y2_min = box2["x"], box2["y"]
            x2_max, y2_max = x2_min + box2["width"], y2_min + box2["height"]
            
            # Horizontal distance
            x_dist = max(0, x2_min - x1_max, x1_min - x2_max)
            # Vertical distance
            y_dist = max(0, y2_min - y1_max, y1_min - y2_max)
            
            # Check if euclidean-like distance is within threshold
            dist = np.sqrt(x_dist**2 + y_dist**2)
            return dist <= dist_thresh

        # Active tracking of groups of candidates to merge
        groups = []
        for cand in candidates:
            # Check if this candidate fits in any existing group
            placed = False
            for group in groups:
                for member in group:
                    if boxes_are_close(cand, member, threshold):
                        group.append(cand)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                groups.append([cand])

        # Resolve grouped candidates into single merged candidates
        merged_list = []
        h_img, w_img = img_shape[:2]
        for group in groups:
            if len(group) == 1:
                merged_list.append(group[0])
                continue
            
            # Calculate outer bounding box of the group
            x_min = min(m["x"] for m in group)
            y_min = min(m["y"] for m in group)
            x_max = max(m["x"] + m["width"] for m in group)
            y_max = max(m["y"] + m["height"] for m in group)
            
            merged_w = x_max - x_min
            merged_h = y_max - y_min
            
            # Combine the masks
            combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            for m in group:
                combined_mask = cv2.bitwise_or(combined_mask, m["mask"])
            
            # Recompute total area (sum of actual mask pixels inside the combined bounding box)
            total_area = int(np.sum(combined_mask > 0))
            
            # Find the new outer contour
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = max(contours, key=cv2.contourArea) if contours else group[0]["contour"]
            
            # Compute solidity and circularity for the merged shape
            perimeter = cv2.arcLength(best_contour, True)
            hull = cv2.convexHull(best_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(total_area) / hull_area if hull_area > 0 else 0.0
            circularity = (4 * np.pi * total_area) / (perimeter ** 2) if perimeter > 0 else 0.0

            merged_list.append({
                "x": x_min,
                "y": y_min,
                "width": merged_w,
                "height": merged_h,
                "area": total_area,
                "solidity": solidity,
                "circularity": circularity,
                "contour": best_contour,
                "mask": combined_mask
            })
            
        return merged_list


# ============================================================
# Extracted from validation.py
# ============================================================


logger = logging.getLogger("Standalone")

class FODValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for Deep Learning validation: {self.device}")
        
        self.feature_extractor = None
        self.yolo_model = None
        
        if self.config.enable_dl_validation:
            self._load_models()

    def _load_models(self):
        """Loads DL models locally."""
        if self.config.dl_validator_type == "deep_features":
            try:
                logger.info(f"Loading pre-trained feature extractor: {self.config.cnn_model_name}")
                if self.config.cnn_model_name == "mobilenet_v3_small":
                    weights = models.MobileNet_V3_Small_Weights.DEFAULT
                    # Load model with pre-trained weights
                    base_model = models.mobilenet_v3_small(weights=weights)
                elif self.config.cnn_model_name == "resnet18":
                    weights = models.ResNet18_Weights.DEFAULT
                    base_model = models.resnet18(weights=weights)
                else:
                    logger.warning(f"Unsupported CNN model: {self.config.cnn_model_name}. Defaulting to MobileNetV3 Small.")
                    weights = models.MobileNet_V3_Small_Weights.DEFAULT
                    base_model = models.mobilenet_v3_small(weights=weights)

                # Remove the final classification layer, keeping only the feature extractor
                # MobileNetV3 and ResNet18 both have an average pooling layer followed by a classifier
                if hasattr(base_model, "classifier"):
                    # For MobileNet, replace classifier with Identity to get pooling output
                    base_model.classifier = torch.nn.Identity()
                elif hasattr(base_model, "fc"):
                    # For ResNet, replace fc with Identity
                    base_model.fc = torch.nn.Identity()
                
                self.feature_extractor = base_model.to(self.device)
                self.feature_extractor.eval()
                
                # Image transformation pipeline for ImageNet-trained models
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                logger.info("Deep feature extractor loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load PyTorch feature extractor: {e}. Falling back to rule-based validation only.")
                self.feature_extractor = None

        elif self.config.dl_validator_type == "yolo":
            try:
                logger.info("Loading YOLOv8 model...")
                # Load local YOLO model (downloads on first run, cached locally)
                self.yolo_model = YOLO(self.config.yolo_model_path)
                logger.info(f"YOLOv8 loaded successfully from {self.config.yolo_model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}. Falling back to rule-based validation only.")
                self.yolo_model = None

    def _extract_deep_features(self, patch: np.ndarray) -> Optional[torch.Tensor]:
        """Extracts deep features from an image patch."""
        if self.feature_extractor is None:
            return None
        try:
            # Convert BGR CV2 image to PIL RGB image
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_patch)
            
            # Apply transformations and add batch dimension
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(tensor)
                # Flatten features
                features = torch.flatten(features, 1)
            return features
        except Exception as e:
            logger.error(f"Error during deep feature extraction: {e}")
            return None

    def validate_deep_features(self, ref_patch: np.ndarray, curr_patch: np.ndarray) -> Tuple[bool, float]:
        """
        Validates if the current patch contains a real physical change or just lighting/shadow.
        Compares features of reference patch and current patch.
        Returns:
            - validated: True if it is a real foreign object (low similarity),
                         False if it is likely a shadow/light shift (high similarity).
            - similarity: Cosine similarity score (0.0 to 1.0)
        """
        if self.feature_extractor is None:
            # Fallback if model not loaded: assume it is a real object to avoid false negatives
            return True, 0.0

        ref_feat = self._extract_deep_features(ref_patch)
        curr_feat = self._extract_deep_features(curr_patch)

        if ref_feat is None or curr_feat is None:
            return True, 0.0

        # Compute cosine similarity
        cos = torch.nn.CosineSimilarity(dim=1)
        similarity = float(cos(ref_feat, curr_feat).cpu().item())
        
        # High similarity means the semantic structures are identical (just different lighting/shadow).
        # Low similarity means the shape or structure has changed (a new object is present).
        validated = similarity < self.config.deep_feature_threshold
        
        logger.info(f"Deep Feature validation: similarity={similarity:.3f}, threshold={self.config.deep_feature_threshold}, validated={validated}")
        return validated, similarity

    def validate_region(self, cand: dict, ref_img: np.ndarray, warped_img: np.ndarray, ignore_mask: Optional[np.ndarray] = None) -> Tuple[bool, float, str]:
        """
        Validates a candidate region based on:
        1. Boundary proximity (reject if too close to border)
        2. Ignore mask overlap (reject if overlapping user ignore zone)
        3. Deep Learning validation (Cosine similarity check or YOLO object check)
        Returns:
            - is_valid: bool
            - confidence_score: float
            - rejection_reason: str
        """
        x, y, w, h = cand["x"], cand["y"], cand["width"], cand["height"]
        img_h, img_w = warped_img.shape[:2]

        # 1. Boundary filter
        border = self.config.ignore_border_pixels
        if x < border or y < border or (x + w) > (img_w - border) or (y + h) > (img_h - border):
            return False, 0.0, "BOUNDARY_EXCLUSION"

        # 2. Ignore mask overlap filter
        if ignore_mask is not None:
            # Crop the candidate region mask and the ignore mask
            cand_mask = cand["mask"]
            # Ignore mask: values should be 0 for ignore zones and 255 for inspect zones
            # We calculate what percentage of the candidate's pixels lie in the ignore zone (value == 0)
            overlap_pixels = np.sum((cand_mask > 0) & (ignore_mask == 0))
            cand_pixels = np.sum(cand_mask > 0)
            
            if cand_pixels > 0:
                overlap_ratio = float(overlap_pixels) / float(cand_pixels)
                if overlap_ratio > 0.15: # If more than 15% of the object is in an ignore zone
                    logger.info(f"Region rejected by ignore mask overlap: {overlap_ratio * 100:.1f}% ignored.")
                    return False, 0.0, "IGNORE_MASK_OVERLAP"

        # 3. Deep Learning validation
        if self.config.enable_dl_validation:
            # Extract patches
            # Add a small padding to the patch to capture local context
            pad = 5
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(img_w, x + w + pad)
            y_max = min(img_h, y + h + pad)
            
            ref_patch = ref_img[y_min:y_max, x_min:x_max]
            curr_patch = warped_img[y_min:y_max, x_min:x_max]

            if ref_patch.size == 0 or curr_patch.size == 0:
                return True, 0.5, "PASSED_WITH_EMPTY_PATCH"

            if self.config.dl_validator_type == "deep_features":
                is_valid, similarity = self.validate_deep_features(ref_patch, curr_patch)
                # Compute confidence: lower similarity = higher confidence of foreign object
                confidence = max(0.0, min(1.0, 1.0 - similarity))
                
                # Boost confidence if similarity is very low
                if confidence > 0.4:
                    confidence = 0.5 + 0.5 * confidence # map [0.4, 1.0] -> [0.7, 1.0]
                
                if not is_valid:
                    return False, confidence, "DEEP_FEATURE_SIMILARITY_TOO_HIGH"
                else:
                    return True, confidence, "PASSED_DEEP_FEATURE_VALIDATION"

            elif self.config.dl_validator_type == "yolo" and self.yolo_model is not None:
                # YOLO validation
                # Run YOLO on the warped image
                try:
                    # Run YOLO on the patch or entire image
                    results = self.yolo_model(curr_patch, verbose=False, conf=self.config.yolo_conf)
                    
                    # If YOLO finds any class of object inside the patch
                    # (since we cropped around the candidate, any object detected here is a positive validation)
                    yolo_detected = False
                    max_conf = 0.0
                    for r in results:
                        if len(r.boxes) > 0:
                            yolo_detected = True
                            max_conf = float(r.boxes.conf.cpu().numpy().max())
                            break
                            
                    if yolo_detected:
                        logger.info(f"YOLOv8 confirmed object in patch with confidence {max_conf:.3f}")
                        return True, max_conf, "YOLO_CONFIRMED"
                    else:
                        # If YOLO doesn't detect it, but it was found by classical difference,
                        # since YOLO only detects COCO classes, we shouldn't discard it immediately.
                        # However, we can lower its confidence score
                        logger.info("YOLOv8 did not detect any known class in patch. Keeping with lower confidence.")
                        return True, 0.45, "PASSED_CLASSICAL_ONLY"
                except Exception as e:
                    logger.error(f"YOLO validation failed: {e}")
                    return True, 0.5, "PASSED_FALLBACK"
        
        # Rule-based fallback if DL is disabled or failed
        # Calculate a simple structural similarity value on the patch
        # using normalized correlation coefficient
        try:
            ref_gray_p = cv2.cvtColor(ref_patch, cv2.COLOR_BGR2GRAY)
            curr_gray_p = cv2.cvtColor(curr_patch, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(curr_gray_p, ref_gray_p, cv2.TM_CCOEFF_NORMED)
            ncc = float(res[0][0])
            confidence = 1.0 - max(0.0, ncc)
            
            # Simple threshold on template matching
            if ncc > 0.90:
                logger.info(f"Rejected by template matching similarity: {ncc:.3f}")
                return False, confidence, "HIGH_TEMPLATE_MATCH_SIMILARITY"
                
            return True, confidence, "PASSED_CLASSICAL_TEMPLATE_MATCH"
        except Exception as e:
            logger.error(f"Fallback validation failed: {e}")
            return True, 0.5, "PASSED_WITHOUT_VALIDATION"
        
        return True, 0.8, "PASSED_DEFAULT"


# ============================================================
# Extracted from detector.py
# ============================================================


logger = logging.getLogger("Standalone")


# ---------------------------------------------------------------------------
# Standalone Color-Anomaly Detector
# Catches objects whose color distribution is absent from the reference image
# (e.g., orange cone on a green-tinted underbody scan).
# This runs INDEPENDENTLY of alignment quality and is always executed.
# ---------------------------------------------------------------------------
class ColorAnomalyDetector:
    """
    Detects regions in the current image whose dominant color is statistically
    anomalous with respect to the reference image.  Works even when both images
    have different sizes / slight viewpoint changes.
    """

    # HSV hue-saturation ranges considered "always foreign" on industrial scans.
    # These are highly-saturated, warm colors that do not appear in bare metal /
    # painted vehicle underbody images.
    FOREIGN_HSV_RANGES = [
        # Orange (safety cones, plastic objects, etc.)
        {"lo": np.array([5,  120, 60]),  "hi": np.array([25, 255, 255]), "label": "orange"},
        # Red (part 1 — wraps around 0°)
        {"lo": np.array([0,  140, 60]),  "hi": np.array([8,  255, 255]), "label": "red_lo"},
        # Red (part 2 — above 170°)
        {"lo": np.array([170, 140, 60]), "hi": np.array([180, 255, 255]), "label": "red_hi"},
        # Bright yellow
        {"lo": np.array([25, 100, 100]), "hi": np.array([35, 255, 255]), "label": "yellow"},
        # Bright green (non-scanner tint) — very saturated
        {"lo": np.array([40, 150, 80]),  "hi": np.array([80, 255, 255]), "label": "green"},
        # Blue / cobalt
        {"lo": np.array([95, 100, 60]),  "hi": np.array([130, 255, 255]), "label": "blue"},
        # Magenta / pink
        {"lo": np.array([140, 80, 60]),  "hi": np.array([170, 255, 255]), "label": "magenta"},
    ]

    # Minimum pixel area to flag as a candidate
    MIN_AREA = 200
    # If the same hue range covers > this fraction of the REFERENCE, it is a
    # known background color and will NOT be flagged.
    REF_COVERAGE_THRESHOLD = 0.005   # 0.5 % of image

    def detect(
        self,
        ref_img: np.ndarray,
        curr_img: np.ndarray,
        image_shape: Tuple[int, int],
        border_fraction: float = 0.05,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Args:
            ref_img:      BGR reference image (may differ in size from curr_img)
            curr_img:     BGR current/live image
            image_shape:  (H, W) target canvas size for the output mask

        Returns:
            (detections_list, combined_mask_on_canvas)
        """
        H, W = image_shape
        curr_hsv = cv2.cvtColor(curr_img, cv2.COLOR_BGR2HSV)
        ref_hsv  = cv2.cvtColor(ref_img,  cv2.COLOR_BGR2HSV)

        # Resize reference HSV to same size as current for per-pixel comparison
        ref_hsv_rs = cv2.resize(ref_hsv, (curr_img.shape[1], curr_img.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        ref_area = ref_img.shape[0] * ref_img.shape[1]
        curr_area = curr_img.shape[0] * curr_img.shape[1]

        detections: List[Dict] = []
        combined_mask_curr = np.zeros((curr_img.shape[0], curr_img.shape[1]), dtype=np.uint8)

        for hrange in self.FOREIGN_HSV_RANGES:
            lo, hi = hrange["lo"], hrange["hi"]
            label = hrange["label"]

            # Build mask in current image
            curr_mask = cv2.inRange(curr_hsv, lo, hi)
            # Build mask in reference image (at original ref size)
            ref_mask  = cv2.inRange(ref_hsv, lo, hi)

            ref_coverage = np.sum(ref_mask > 0) / max(ref_area, 1)
            if ref_coverage > self.REF_COVERAGE_THRESHOLD:
                # This color exists prominently in the reference → not a foreign object
                logger.debug(f"Color range '{label}' is background in ref ({ref_coverage:.2%}). Skipping.")
                continue

            # Suppress areas in curr where the REFERENCE (resized) also has this color
            # (handles partial-overlap viewpoint differences)
            ref_mask_rs = cv2.inRange(ref_hsv_rs, lo, hi)
            curr_mask_clean = cv2.bitwise_and(curr_mask, cv2.bitwise_not(ref_mask_rs))

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            curr_mask_clean = cv2.morphologyEx(curr_mask_clean, cv2.MORPH_OPEN,  kernel, iterations=1)
            curr_mask_clean = cv2.morphologyEx(curr_mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

            n_pixels = int(np.sum(curr_mask_clean > 0))
            if n_pixels < self.MIN_AREA:
                continue

            # Find contours for bounding boxes
            contours, _ = cv2.findContours(curr_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.MIN_AREA:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)

                # Skip detections in the border strip (UI overlays, scanner artifacts)
                border_px_x = int(curr_img.shape[1] * border_fraction)
                border_px_y = int(curr_img.shape[0] * border_fraction)
                cx_center = x + bw // 2
                cy_center = y + bh // 2
                if (cx_center < border_px_x or cx_center > curr_img.shape[1] - border_px_x or
                        cy_center < border_px_y or cy_center > curr_img.shape[0] - border_px_y):
                    logger.debug(f"Skipping border detection at ({x},{y},{bw},{bh})")
                    continue

                # Scale bounding box to the output canvas coordinate system
                sx = W / curr_img.shape[1]
                sy = H / curr_img.shape[0]
                cx = int(x  * sx)
                cy = int(y  * sy)
                cw = int(bw * sx)
                ch = int(bh * sy)

                # Confidence based on coverage relative to bbox area
                bbox_area = bw * bh
                fill_ratio = float(area) / max(bbox_area, 1)
                confidence = min(0.98, 0.65 + fill_ratio * 0.33)

                detections.append({
                    "x": cx,
                    "y": cy,
                    "width": cw,
                    "height": ch,
                    "area": int(area * sx * sy),
                    "confidence": round(confidence, 3),
                    "label": f"foreign_object_color_{label}",
                    "method": "color_anomaly"
                })
                logger.info(
                    f"Color anomaly detected: color='{label}', "
                    f"bbox=({cx},{cy},{cw},{ch}), confidence={confidence:.3f}"
                )

            combined_mask_curr = cv2.bitwise_or(combined_mask_curr, curr_mask_clean)

        # Scale combined mask to canvas
        combined_mask_canvas = cv2.resize(combined_mask_curr, (W, H), interpolation=cv2.INTER_NEAREST)

        # Deduplicate detections across color ranges using IoU
        detections = self._dedup(detections, iou_threshold=0.15)

        return detections, combined_mask_canvas

    def _iou(self, a: Dict, b: Dict) -> float:
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["width"], by1 + b["height"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter_area = inter_w * inter_h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area
        return float(inter_area) / max(union_area, 1)

    def _dedup(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping detections from different color ranges into one."""
        kept: List[Dict] = []
        for det in sorted(detections, key=lambda d: -d["confidence"]):
            overlap = False
            for k in kept:
                if self._iou(det, k) > iou_threshold:
                    overlap = True
                    # Expand the kept box to include the new one
                    x1 = min(k["x"], det["x"])
                    y1 = min(k["y"], det["y"])
                    x2 = max(k["x"] + k["width"],  det["x"] + det["width"])
                    y2 = max(k["y"] + k["height"], det["y"] + det["height"])
                    k["x"], k["y"] = x1, y1
                    k["width"],  k["height"] = x2 - x1, y2 - y1
                    k["confidence"] = max(k["confidence"], det["confidence"])
                    break
            if not overlap:
                kept.append(dict(det))
        return kept


# ---------------------------------------------------------------------------
# Main Detector Orchestrator
# ---------------------------------------------------------------------------
class FODDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.aligner = ImageAligner(config.alignment)
        self.diff_detector = DifferenceDetector(config.difference)
        self.region_proposer = RegionProposer(config.region)
        self.validator = FODValidator(config.validation)
        self.color_detector = ColorAnomalyDetector()

    def detect(
        self,
        ref_images: List[np.ndarray],
        curr_image: np.ndarray,
        ignore_mask: Optional[np.ndarray] = None,
        output_prefix: str = "fod"
    ) -> Dict[str, Any]:
        """
        Runs the full hybrid Foreign Object Detection pipeline:
          Pass A — Color-anomaly detection (orientation/alignment independent)
          Pass B — Classical CV: alignment → diff map → contour proposals → DL validation
          Final  — Merge unique detections from both passes
        """
        start_time = time.time()

        # ---- Preprocess -------------------------------------------------------
        logger.info("Preprocessing images.")
        curr_preprocessed = self.preprocessor.preprocess(curr_image)
        ref_preprocessed_list = [self.preprocessor.preprocess(r) for r in ref_images]
        H_canvas, W_canvas = curr_preprocessed.shape[:2]

        # ---- Handle Ignore Mask -----------------------------------------------
        preprocessed_ignore_mask = None
        if ignore_mask is not None:
            preprocessed_ignore_mask = cv2.resize(
                ignore_mask, (W_canvas, H_canvas), interpolation=cv2.INTER_NEAREST
            )
            if len(preprocessed_ignore_mask.shape) == 3:
                preprocessed_ignore_mask = cv2.cvtColor(preprocessed_ignore_mask, cv2.COLOR_BGR2GRAY)
            _, preprocessed_ignore_mask = cv2.threshold(
                preprocessed_ignore_mask, 127, 255, cv2.THRESH_BINARY
            )

        # ====================================================================
        # PASS A — COLOR ANOMALY (works independently of alignment)
        # ====================================================================
        logger.info("Pass A: Color-anomaly detection.")
        color_detections: List[Dict] = []
        color_mask = np.zeros((H_canvas, W_canvas), dtype=np.uint8)
        try:
            best_ref_orig = ref_images[0]   # use original (not preprocessed) for color
            ca_dets, ca_mask = self.color_detector.detect(
                best_ref_orig, curr_image, (H_canvas, W_canvas)
            )
            color_detections.extend(ca_dets)
            color_mask = cv2.bitwise_or(color_mask, ca_mask)
            logger.info(f"Pass A complete: {len(color_detections)} color anomaly detections.")
        except Exception as e:
            logger.error(f"Color anomaly detection error: {e}")

        # ====================================================================
        # PASS B — ALIGNMENT + CLASSICAL CV + DL VALIDATION
        # ====================================================================
        logger.info("Pass B: Alignment-based classical CV detection.")

        # 1. Align
        alignment_score = 0.0
        best_ref_preprocessed = ref_preprocessed_list[0]
        try:
            if len(ref_preprocessed_list) == 1:
                warped_curr, alignment_score = self.aligner.align(
                    best_ref_preprocessed, curr_preprocessed
                )
                best_idx = 0
            else:
                best_ref_preprocessed, warped_curr, alignment_score, best_idx = \
                    self.aligner.select_best_reference(ref_preprocessed_list, curr_preprocessed)
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            warped_curr = cv2.resize(
                curr_preprocessed,
                (best_ref_preprocessed.shape[1], best_ref_preprocessed.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            alignment_score = 0.1

        # If aligner returned None (should not happen with new fallback, but safety net)
        if warped_curr is None:
            logger.warning("Alignment returned None. Using resize fallback.")
            warped_curr = cv2.resize(
                curr_preprocessed,
                (best_ref_preprocessed.shape[1], best_ref_preprocessed.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            alignment_score = 0.05

        # 2. Difference detection
        try:
            combined_diff, thresholded_mask = self.diff_detector.detect_differences(
                best_ref_preprocessed, warped_curr
            )
        except Exception as e:
            logger.error(f"Difference detection error: {e}")
            combined_diff = np.zeros((H_canvas, W_canvas), dtype=np.uint8)
            thresholded_mask = np.zeros((H_canvas, W_canvas), dtype=np.uint8)

        # 3. Region proposals
        candidates = []
        try:
            candidates = self.region_proposer.propose_regions(thresholded_mask)
            logger.info(f"Region proposer found {len(candidates)} candidates.")
        except Exception as e:
            logger.error(f"Region proposal error: {e}")

        # 4. Deep-learning validation of candidates
        validated_detections: List[Dict] = []
        final_mask = np.zeros_like(thresholded_mask)

        for cand in candidates:
            try:
                is_valid, confidence, reason = self.validator.validate_region(
                    cand, best_ref_preprocessed, warped_curr, preprocessed_ignore_mask
                )
                if is_valid and confidence >= 0.50:
                    validated_detections.append({
                        "x": int(cand["x"]),
                        "y": int(cand["y"]),
                        "width": int(cand["width"]),
                        "height": int(cand["height"]),
                        "area": int(cand["area"]),
                        "confidence": float(confidence),
                        "label": "foreign_object",
                        "method": "diff_map"
                    })
                    final_mask = cv2.bitwise_or(final_mask, cand["mask"])
                elif is_valid:
                    logger.info(f"Candidate at ({cand['x']},{cand['y']}) rejected: LOW_CONFIDENCE ({confidence:.3f})")
                else:
                    logger.info(f"Candidate at ({cand['x']},{cand['y']}) rejected: {reason}")
            except Exception as e:
                logger.error(f"Validation error for candidate: {e}")

        logger.info(f"Pass B complete: {len(validated_detections)} validated diff-map detections.")

        # Skip diff-map detections when alignment is too poor (too many false positives)
        ALIGNMENT_QUALITY_GATE = 0.45
        if alignment_score < ALIGNMENT_QUALITY_GATE and len(validated_detections) > 0:
            logger.warning(
                f"Alignment score {alignment_score:.3f} < {ALIGNMENT_QUALITY_GATE}. "
                f"Dropping {len(validated_detections)} diff-map detections to avoid false positives. "
                f"Color-anomaly detections are retained."
            )
            validated_detections = []
            final_mask = np.zeros((H_canvas, W_canvas), dtype=np.uint8)

        # ====================================================================
        # MERGE — Combine Pass A + Pass B, deduplicate overlapping boxes
        # ====================================================================
        all_detections = self._merge_detections(color_detections, validated_detections, self.config.region.merge_distance_threshold)

        # Ensure masks are the same size before OR-merge
        if final_mask.shape != color_mask.shape:
            final_mask = cv2.resize(final_mask, (color_mask.shape[1], color_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        all_mask = cv2.bitwise_or(color_mask, final_mask)

        # ====================================================================
        # Similarity score
        # ====================================================================
        try:
            ref_gray   = cv2.cvtColor(best_ref_preprocessed, cv2.COLOR_BGR2GRAY)
            warped_gray = cv2.cvtColor(warped_curr, cv2.COLOR_BGR2GRAY)
            ssim_val, _ = skssim(ref_gray, warped_gray, full=True)
            similarity_score = max(0.0, min(100.0, float(ssim_val) * 100.0))
        except Exception:
            similarity_score = 0.0

        # ====================================================================
        # Visualizations
        # ====================================================================
        # Normalize visualization base to canvas size (H_canvas x W_canvas)
        # so it always matches all_mask dimensions
        if warped_curr is not None and warped_curr.shape[:2] == (H_canvas, W_canvas):
            viz_base = warped_curr
        else:
            # Resize curr_preprocessed to the canvas size
            viz_base = cv2.resize(curr_preprocessed, (W_canvas, H_canvas), interpolation=cv2.INTER_LINEAR)

        # Ensure all_mask matches viz_base exactly
        if all_mask.shape[:2] != (H_canvas, W_canvas):
            all_mask = cv2.resize(all_mask, (W_canvas, H_canvas), interpolation=cv2.INTER_NEAREST)

        marked_image = draw_detections(viz_base, all_detections, all_mask)
        diff_heatmap = create_diff_heatmap(combined_diff)

        # Save outputs
        ts = int(time.time() * 1000)
        marked_path = OUTPUT_DIR / f"{output_prefix}_{ts}_marked.jpg"
        diff_path   = OUTPUT_DIR / f"{output_prefix}_{ts}_diff.jpg"
        mask_path   = OUTPUT_DIR / f"{output_prefix}_{ts}_mask.png"
        save_image(marked_image, marked_path)
        save_image(diff_heatmap, diff_path)
        save_image(all_mask, mask_path)

        processing_time_ms = int((time.time() - start_time) * 1000)
        status = "FOD_DETECTED" if len(all_detections) > 0 else "NO_FOD"

        response = {
            "status": status,
            "accuracy_mode": "HYBRID_COLOR+DIFFMAP",
            "similarity_score": round(similarity_score, 2),
            "alignment_score": round(float(alignment_score), 3),
            "objects": len(all_detections),
            "detections": all_detections,
            "output_image": f"output/{marked_path.name}",
            "difference_map": f"output/{diff_path.name}",
            "mask_image": f"output/{mask_path.name}",
            "processing_time_ms": processing_time_ms
        }

        logger.info(
            f"Detection complete. Status={status}, Total objects={len(all_detections)}, "
            f"Color={len(color_detections)}, DiffMap={len(validated_detections)}, "
            f"Time={processing_time_ms}ms"
        )
        return response

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _merge_detections(
        self,
        color_dets: List[Dict],
        diffmap_dets: List[Dict],
        dist_thresh: int = 75
    ) -> List[Dict]:
        """
        Merge detections from color and diff-map passes.
        Also merges close bounding boxes together.
        """
        all_dets = color_dets + diffmap_dets
        
        def boxes_are_close(box1, box2):
            x1_min, y1_min = box1["x"], box1["y"]
            x1_max, y1_max = x1_min + box1["width"], y1_min + box1["height"]
            
            x2_min, y2_min = box2["x"], box2["y"]
            x2_max, y2_max = x2_min + box2["width"], y2_min + box2["height"]
            
            x_dist = max(0, x2_min - x1_max, x1_min - x2_max)
            y_dist = max(0, y2_min - y1_max, y1_min - y2_max)
            
            dist = np.sqrt(x_dist**2 + y_dist**2)
            return dist <= dist_thresh

        groups = []
        for det in all_dets:
            placed = False
            for group in groups:
                for member in group:
                    if boxes_are_close(det, member):
                        group.append(det)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                groups.append([det])
                
        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
                
            x_min = min(m["x"] for m in group)
            y_min = min(m["y"] for m in group)
            x_max = max(m["x"] + m["width"] for m in group)
            y_max = max(m["y"] + m["height"] for m in group)
            
            max_conf = max(m["confidence"] for m in group)
            
            # Prefer color anomaly labels and methods if mixed
            label = group[0]["label"]
            method = group[0]["method"]
            for m in group:
                if "color" in m["method"]:
                    label = m["label"]
                    method = m["method"]
                    break
                    
            merged.append({
                "x": x_min,
                "y": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "area": sum(m["area"] for m in group),
                "confidence": max_conf,
                "label": label,
                "method": method
            })
            
        return merged

    def _iou(self, a: Dict, b: Dict) -> float:
        """Compute Intersection-over-Union for two bounding box dicts."""
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["width"], by1 + b["height"]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        return float(inter_area) / max(union_area, 1)


# ============================================================
# Standalone Inference Wrapper
# ============================================================
import base64

def decode_base64_image(b64_string: str) -> np.ndarray:
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None

def encode_image_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def detect_anomalies(ref_b64: str, curr_b64: str) -> dict:
    ref_img = decode_base64_image(ref_b64)
    curr_img = decode_base64_image(curr_b64)

    if ref_img is None or curr_img is None:
        return {"error": "Failed to decode input images."}

    detector = FODDetector(DEFAULT_CONFIG)
    
    ts = int(time.time() * 1000)
    prefix = f"standalone_{ts}"

    result = detector.detect(
        ref_images=[ref_img],
        curr_image=curr_img,
        ignore_mask=None,
        output_prefix=prefix,
    )

    out_path = OUTPUT_DIR / Path(result["output_image"]).name
    if out_path.exists():
        out_img = cv2.imread(str(out_path))
        if out_img is not None:
            result["output_image_base64"] = encode_image_base64(out_img)
    else:
        result["output_image_base64"] = None

    return result
