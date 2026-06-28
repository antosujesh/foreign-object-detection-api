import cv2
import json
from pathlib import Path
from config import DEFAULT_CONFIG
from detector import FODDetector

uploads_dir = Path("uploads")
# Find the latest uploaded JPG files (the high-resolution scans from the browser)
ref_files = sorted(list(uploads_dir.glob("ref_0_*.jpg")))
curr_files = sorted(list(uploads_dir.glob("curr_*.jpg")))

if not ref_files or not curr_files:
    print("Error: Could not find the uploaded high-resolution PNG scans in the uploads directory.")
    exit(1)

ref_path = ref_files[-1]
curr_path = curr_files[-1]

print(f"Reference Image Path: {ref_path}")
print(f"Inspection Image Path: {curr_path}")

# Load images
ref_img = cv2.imread(str(ref_path))
curr_img = cv2.imread(str(curr_path))

if ref_img is None or curr_img is None:
    print("Error: Failed to decode one or both of the PNG images.")
    exit(1)

# Initialize detector with default config
# SIFT feature alignment with 180-degree auto-rotation recovery, combined SSIM + LAB color + Sobel edge difference, 
# contour filtering, and class-agnostic deep feature similarity validation.
detector = FODDetector(DEFAULT_CONFIG)

print("Starting hybrid detection pipeline on chassis scans...")
result = detector.detect(
    ref_images=[ref_img],
    curr_image=curr_img,
    output_prefix="real_chassis_inspection"
)

print("\n--- DETECTION RESPONSE JSON ---")
print(json.dumps(result, indent=2))
print("--------------------------------")
