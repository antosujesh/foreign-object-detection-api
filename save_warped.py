import cv2
import numpy as np
from pathlib import Path

uploads_dir = Path("uploads")
ref_path = sorted(list(uploads_dir.glob("ref_0_*.png")))[-1]
curr_path = sorted(list(uploads_dir.glob("curr_*.png")))[-1]

ref_img = cv2.imread(str(ref_path))
curr_img = cv2.imread(str(curr_path))

# Resize preserving aspect ratio (width=1280)
h_ref, w_ref = ref_img.shape[:2]
target_w = 1280
h_target = int(h_ref * (target_w / w_ref))
h_target = max(2, (h_target // 2) * 2)

ref_resized = cv2.resize(ref_img, (target_w, h_target), interpolation=cv2.INTER_LANCZOS4)
curr_resized = cv2.resize(curr_img, (target_w, h_target), interpolation=cv2.INTER_LANCZOS4)

# Get L channel
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

pts_curr = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(pts_curr, pts_ref, cv2.RANSAC, 5.0)

# Warp current image
h_ref_sz, w_ref_sz = ref_resized.shape[:2]
warped_curr = cv2.warpPerspective(curr_resized, H, (w_ref_sz, h_ref_sz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

# Save side-by-side
combined = np.vstack((ref_resized, warped_curr))
cv2.imwrite("aligned_comparison.jpg", combined)
print("Saved aligned comparison to 'aligned_comparison.jpg'.")

# Print coordinates of some matches to verify
inliers_curr = pts_curr[mask.ravel() == 1]
inliers_ref = pts_ref[mask.ravel() == 1]
print(f"Number of SIFT inliers: {len(inliers_curr)}")
for i in range(min(5, len(inliers_curr))):
    print(f"Inlier {i}: Current {inliers_curr[i][0]} -> Reference {inliers_ref[i][0]}")
