import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from app import app
from utils import get_logger

logger = get_logger("TestAPI")

def generate_test_images():
    """
    Generates synthetic reference, current, and ignore mask images
    to test the alignment, difference, and validation pipeline.
    """
    logger.info("Generating synthetic test images...")
    
    # 1. Generate Reference Image (OK state)
    # Light gray background
    ref = np.ones((800, 800, 3), dtype=np.uint8) * 180
    
    # Draw static chassis features (to provide SIFT/ORB keypoints)
    # Draw some grid lines
    for i in range(100, 800, 100):
        cv2.line(ref, (i, 0), (i, 800), (120, 120, 120), 2)
        cv2.line(ref, (0, i), (800, i), (120, 120, 120), 2)
        
    # Draw chassis plates (large rectangles)
    cv2.rectangle(ref, (150, 150), (450, 500), (100, 100, 100), -1)
    cv2.circle(ref, (600, 300), 80, (90, 90, 90), -1)
    # Add high-frequency textures for robust matching
    for i in range(10):
        cv2.circle(ref, (200 + i*20, 200 + i*15), 10, (50, 50, 50), 2)
        cv2.circle(ref, (550 + i*10, 250 + i*10), 8, (60, 60, 60), -1)
    
    # 2. Generate Current Image (Warped, with lighting change, plus foreign objects)
    # Rotate ref image slightly (+1.5 degrees) and translate by (+10, -5) to simulate camera shift
    h, w = ref.shape[:2]
    angle = 1.5
    scale = 1.0
    tx, ty = 10, -5
    
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    
    curr = cv2.warpAffine(ref, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    curr = cv2.warpAffine(curr, M_trans, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # Add a global lighting change (make it 20 levels darker)
    curr = np.clip(curr.astype(np.int16) - 20, 0, 255).astype(np.uint8)
    
    # Add Foreign Object 1: A red tool-like rectangle (Inspect Zone)
    # In the aligned coordinate frame, let's put it near (300, 400).
    # Since the image is rotated, let's draw it directly on the warped image
    # Red rectangle
    cv2.rectangle(curr, (280, 380), (360, 410), (50, 50, 180), -1) # BGR
    # Draw a little handle
    cv2.line(curr, (320, 410), (320, 450), (20, 20, 20), 8)
    
    # Add Foreign Object 2: A dark blue square (placed inside the Ignore Zone)
    # Let's put it near (600, 300)
    cv2.rectangle(curr, (580, 280), (620, 320), (180, 50, 50), -1) # BGR
    
    # 3. Generate Ignore Mask
    # 255 (white) = Inspect, 0 (black) = Ignore
    # Let's ignore a square region around the circular plate at (600, 300)
    # This covers Foreign Object 2.
    ignore_mask = np.ones((800, 800), dtype=np.uint8) * 255
    cv2.rectangle(ignore_mask, (550, 250), (650, 350), 0, -1)
    
    # Save the files in the workspace directory
    base_path = Path(__file__).resolve().parent
    ref_path = base_path / "test_ref.jpg"
    curr_path = base_path / "test_curr.jpg"
    mask_path = base_path / "test_mask.jpg"
    
    cv2.imwrite(str(ref_path), ref)
    cv2.imwrite(str(curr_path), curr)
    cv2.imwrite(str(mask_path), ignore_mask)
    
    logger.info(f"Test files saved successfully:")
    logger.info(f" - Reference: {ref_path.name}")
    logger.info(f" - Current: {curr_path.name}")
    logger.info(f" - Ignore Mask: {mask_path.name}")
    
    return ref_path, curr_path, mask_path

def run_integration_test():
    """Runs automated integration test against the local FastAPI router."""
    ref_path, curr_path, mask_path = generate_test_images()
    
    logger.info("Initializing FastAPI TestClient...")
    client = TestClient(app)
    
    # Open files for multipart upload
    files = [
        ("reference_images", ("test_ref.jpg", open(ref_path, "rb"), "image/jpeg")),
        ("current_image", ("test_curr.jpg", open(curr_path, "rb"), "image/jpeg")),
        ("ignore_mask", ("test_mask.jpg", open(mask_path, "rb"), "image/jpeg")),
    ]
    
    # Send request with custom parameters
    data = {
        "min_area": 200,
        "threshold_method": "OTSU",
        "enable_dl_validation": "true",
        "deep_feature_threshold": 0.85,
    }
    
    logger.info("Sending inspection request to POST /detect-fod...")
    response = client.post("/detect-fod", files=files, data=data)
    
    # Close open files
    for _, file_tuple in files:
        file_tuple[1].close()
        
    logger.info(f"Response Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        logger.info("Inspection Successful!")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Alignment Score: {result['alignment_score']}")
        logger.info(f"Similarity Score: {result['similarity_score']}")
        logger.info(f"Objects Found: {result['objects']}")
        
        # We expect exactly 1 object to be detected, because Object 2 is inside the ignore zone
        logger.info("Detections details:")
        for idx, det in enumerate(result['detections']):
            logger.info(f" - FOD #{idx+1}: Box=[{det['x']}, {det['y']}, {det['width']}, {det['height']}] Area={det['area']} Conf={det['confidence']:.2f}")
            
        assert result["status"] == "FOD_DETECTED", "Expected FOD to be detected."
        assert result["objects"] == 1, f"Expected exactly 1 object, found {result['objects']}"
        logger.info("SUCCESS: Integration test completed. Ignore mask and alignment both validated.")
    else:
        logger.error(f"FAILURE: Inspection failed with detail: {response.text}")
        
if __name__ == "__main__":
    run_integration_test()
