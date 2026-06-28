"""
Test the FOD pipeline against the real vehicle underbody scan images.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
from config import DEFAULT_CONFIG
from detector import FODDetector

UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")

# The REAL vehicle scan images (PNG files from 00:47-00:48)
ref_path  = os.path.join(UPLOADS, "ref_0_1782501496221.png")
curr_path = os.path.join(UPLOADS, "curr_1782501496294.png")

print(f"Reference : {os.path.basename(ref_path)}")
print(f"Current   : {os.path.basename(curr_path)}")

ref_img  = cv2.imread(ref_path)
curr_img = cv2.imread(curr_path)

if ref_img is None or curr_img is None:
    print("ERROR: Could not read images.")
    sys.exit(1)

print(f"Ref  size : {ref_img.shape}")
print(f"Curr size : {curr_img.shape}")

detector = FODDetector(DEFAULT_CONFIG)
result = detector.detect([ref_img], curr_img, output_prefix="vehicle_scan")

print("\n========== DETECTION RESULT ==========")
print(f"Status           : {result['status']}")
print(f"Accuracy Mode    : {result['accuracy_mode']}")
print(f"Objects detected : {result['objects']}")
print(f"Alignment score  : {result['alignment_score']}")
print(f"Similarity score : {result['similarity_score']}")
print(f"Processing time  : {result['processing_time_ms']} ms")
print(f"Output image     : {result.get('output_image','')}")
print()
for i, det in enumerate(result['detections']):
    print(f"  Detection {i+1}: [{det['label']}] conf={det['confidence']:.3f} "
          f"bbox=({det['x']},{det['y']},{det['width']}x{det['height']}) "
          f"method={det.get('method','?')}")
print("=======================================")
