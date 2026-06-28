import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import Union

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
