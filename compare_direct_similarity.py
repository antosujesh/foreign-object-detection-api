import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

uploads_dir = Path("uploads")
ref_path = sorted(list(uploads_dir.glob("ref_0_*.png")))[-1]
curr_path = sorted(list(uploads_dir.glob("curr_*.png")))[-1]

ref_img = cv2.imread(str(ref_path))
curr_img = cv2.imread(str(curr_path))

# Crop out top and bottom banners
ref_cropped = ref_img[45:283, :]
curr_cropped = curr_img[45:283, :]

# Resize preserving aspect ratio (width=1280)
h_ref, w_ref = ref_cropped.shape[:2]
target_w = 1280
h_target = int(h_ref * (target_w / w_ref))
h_target = max(2, (h_target // 2) * 2)

ref_resized = cv2.resize(ref_cropped, (target_w, h_target), interpolation=cv2.INTER_LANCZOS4)
curr_resized = cv2.resize(curr_cropped, (target_w, h_target), interpolation=cv2.INTER_LANCZOS4)

# Preprocess L channel
def preprocess(img):
    denoised = cv2.bilateralFilter(img, 9, 30, 30)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(l)

ref_l = preprocess(ref_resized)
curr_l = preprocess(curr_resized)

orientations = {
    "DIRECT": curr_l,
    "ROTATE_180": cv2.rotate(curr_l, cv2.ROTATE_180),
    "FLIP_HORIZONTAL (Mirror)": cv2.flip(curr_l, 1),
    "FLIP_VERTICAL": cv2.flip(curr_l, 0),
}

print("--- DIRECT SIMILARITY METRICS (NO WARP) ---")
for name, img_var in orientations.items():
    # Calculate NCC
    ref_mean = np.mean(ref_l)
    var_mean = np.mean(img_var)
    ref_std = np.std(ref_l) + 1e-5
    var_std = np.std(img_var) + 1e-5
    ncc = np.mean((ref_l - ref_mean) * (img_var - var_mean)) / (ref_std * var_std)
    
    # Calculate SSIM
    ssim_val, _ = ssim(ref_l, img_var, full=True)
    
    print(f"{name}:")
    print(f" -> NCC: {ncc:.3f}")
    print(f" -> SSIM: {ssim_val:.3f}")
