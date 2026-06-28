import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

uploads_dir = Path("uploads")
ref_path = sorted(list(uploads_dir.glob("ref_0_*.png")))[-1]
curr_path = sorted(list(uploads_dir.glob("curr_*.png")))[-1]

ref_img = cv2.imread(str(ref_path))
curr_img = cv2.imread(str(curr_path))

# Crop out top and bottom banners (ymin=45, ymax=283) in the original 1381x318 space
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
def get_preprocessed_l(img):
    denoised = cv2.bilateralFilter(img, 9, 30, 30)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(l)

ref_l = get_preprocessed_l(ref_resized)
curr_l = get_preprocessed_l(curr_resized)

# SIFT alignment
sift = cv2.SIFT_create(nfeatures=4000)
kp_ref, desc_ref = sift.detectAndCompute(ref_l, None)
kp_curr, desc_curr = sift.detectAndCompute(curr_l, None)

matcher = cv2.BFMatcher(cv2.NORM_L2)
matches = matcher.knnMatch(desc_curr, desc_ref, k=2)

good = []
for m, n in matches:
    if m.distance < 0.80 * n.distance:
        good.append(m)

num_matches = len(good)
print(f"SIFT on Cropped Chassis (L-channel):")
print(f"Keypoints count - Ref: {len(kp_ref)}, Curr: {len(kp_curr)}")
print(f"Good matches: {num_matches}")

if num_matches >= 10:
    pts_curr = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_curr, pts_ref, cv2.RANSAC, 5.0)
    
    inliers = np.sum(mask) if mask is not None else 0
    inlier_ratio = inliers / num_matches if num_matches > 0 else 0
    
    h_ref_sz, w_ref_sz = ref_l.shape[:2]
    warped_l = cv2.warpPerspective(curr_l, H, (w_ref_sz, h_ref_sz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # NCC on valid warped region
    valid = warped_l > 0
    ref_valid = ref_l[valid]
    warped_valid = warped_l[valid]
    
    ref_mean = np.mean(ref_valid)
    warped_mean = np.mean(warped_valid)
    ref_std = np.std(ref_valid) + 1e-5
    warped_std = np.std(warped_valid) + 1e-5
    
    ncc = np.mean((ref_valid - ref_mean) * (warped_valid - warped_mean)) / (ref_std * warped_std)
    ssim_val, _ = ssim(ref_l, warped_l, full=True)
    
    alignment_score = 0.4 * inlier_ratio + 0.6 * ncc
    
    print(f"Alignment results:")
    print(f" -> Inliers: {inliers}/{num_matches} (ratio={inlier_ratio:.3f})")
    print(f" -> NCC: {ncc:.3f}")
    print(f" -> SSIM: {ssim_val:.3f}")
    print(f" -> Combined Alignment Score: {alignment_score:.3f}")
    
    # Save comparison image
    warped_curr = cv2.warpPerspective(curr_resized, H, (w_ref_sz, h_ref_sz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    cv2.imwrite("aligned_comparison_cropped.jpg", np.vstack((ref_resized, warped_curr)))
else:
    print("Alignment failed due to insufficient matches.")
