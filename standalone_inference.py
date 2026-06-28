import base64
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

# Import existing components from the project
from config import DEFAULT_CONFIG, OUTPUT_DIR
from detector import FODDetector

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decodes a base64 string to an OpenCV BGR image."""
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
    """Encodes an OpenCV BGR image to a base64 string."""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def detect_anomalies(ref_b64: str, curr_b64: str) -> Dict[str, Any]:
    """
    Main function to compare an OK Reference Image with an Inspection Image.
    Takes base64 strings as input and returns a dictionary with the results and marked image.
    """
    # 1. Decode images
    ref_img = decode_base64_image(ref_b64)
    curr_img = decode_base64_image(curr_b64)

    if ref_img is None or curr_img is None:
        return {"error": "Failed to decode input images."}

    # 2. Initialize detector with default config
    detector = FODDetector(DEFAULT_CONFIG)
    
    # 3. Generate a unique prefix for this run
    ts = int(time.time() * 1000)
    prefix = f"standalone_{ts}"

    # 4. Run detection
    result = detector.detect(
        ref_images=[ref_img],
        curr_image=curr_img,
        ignore_mask=None,
        output_prefix=prefix,
    )

    # 5. Get the marked image and convert to base64
    out_path = OUTPUT_DIR / Path(result["output_image"]).name
    if out_path.exists():
        out_img = cv2.imread(str(out_path))
        if out_img is not None:
            result["output_image_base64"] = encode_image_base64(out_img)
    else:
        result["output_image_base64"] = None

    # Return final output dictionary
    return result

# Example usage (if run directly)
if __name__ == "__main__":
    print("Standalone Inference Script loaded.")
    print("To use: result = detect_anomalies(ref_base64_str, curr_base64_str)")
