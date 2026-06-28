import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from config import DEFAULT_CONFIG
from alignment import ImageAligner

uploads_dir = Path("uploads")
ref_files = sorted(list(uploads_dir.glob("ref_0_*.png")))
curr_files = sorted(list(uploads_dir.glob("curr_*.png")))

if not ref_files or not curr_files:
    print("No PNG files found.")
    exit(1)

ref_path = ref_files[-1]
curr_path = curr_files[-1]

ref_img = cv2.imread(str(ref_path))
curr_img = cv2.imread(str(curr_path))

# Run preprocessing resizing to standard width (1280) preserving aspect ratio
h_ref, w_ref = ref_img.shape[:2]
target_w = 1280
h_ref_target = int(h_ref * (target_w / w_ref))
h_ref_target = max(2, (h_ref_target // 2) * 2)

ref_resized = cv2.resize(ref_img, (target_w, h_ref_target), interpolation=cv2.INTER_LANCZOS4)
curr_resized = cv2.resize(curr_img, (target_w, h_ref_target), interpolation=cv2.INTER_LANCZOS4)

# Denoise & CLAHE
def preprocess_simple(img):
    denoised = cv2.bilateralFilter(img, 9, 30, 30)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

ref_prep = preprocess_simple(ref_resized)
curr_prep = preprocess_simple(curr_resized)

# Test various spatial orientations
orientations = {
    "DIRECT": curr_prep,
    "ROTATE_180": cv2.rotate(curr_prep, cv2.ROTATE_180),
    "FLIP_HORIZONTAL (Mirror)": cv2.flip(curr_prep, 1),
    "FLIP_VERTICAL": cv2.flip(curr_prep, 0),
}

aligner = ImageAligner(DEFAULT_CONFIG.alignment)

print(f"SIFT Keypoints Count - Reference: {len(aligner.detector.detect(cv2.cvtColor(ref_prep, cv2.COLOR_BGR2GRAY), None))}")

for name, img_variant in orientations.items():
    print(f"\nEvaluating orientation: {name}")
    gray_variant = cv2.cvtColor(img_variant, cv2.COLOR_BGR2GRAY)
    print(f"Keypoints count in variant: {len(aligner.detector.detect(gray_variant, None))}")
    
    # Try SIFT alignment
    warped, score = aligner._align_core(ref_prep, img_variant)
    if warped is not None:
        # Calculate NCC
        ref_gray = cv2.cvtColor(ref_prep, cv2.COLOR_BGR2GRAY)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        valid = warped_gray > 0
        
        ncc = 0.0
        ssim_val = 0.0
        if np.sum(valid) > 100:
            ref_valid = ref_gray[valid]
            warped_valid = warped_gray[valid]
            ref_mean = np.mean(ref_valid)
            warped_mean = np.mean(warped_valid)
            ref_std = np.std(ref_valid) + 1e-5
            warped_std = np.std(warped_valid) + 1e-5
            ncc = np.mean((ref_valid - ref_mean) * (warped_valid - warped_mean)) / (ref_std * warped_std)
            ssim_val, _ = ssim(ref_gray, warped_gray, full=True)
            
        print(f" -> Success! Alignment Score: {score:.3f}, NCC: {ncc:.3f}, SSIM: {ssim_val:.3f}")
    else:
        print(f" -> Alignment FAILED.")
