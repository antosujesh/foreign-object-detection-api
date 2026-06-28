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

# Preprocess
def preprocess(img):
    denoised = cv2.bilateralFilter(img, 9, 30, 30)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

ref_prep = preprocess(ref_resized)
curr_prep = preprocess(curr_resized)

ref_gray = cv2.cvtColor(ref_prep, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(curr_prep, cv2.COLOR_BGR2GRAY)

print("--- EVALUATING DETECTORS AND RATIOS ---")

# 1. SIFT
sift = cv2.SIFT_create(nfeatures=4000)
kp_ref_sift, desc_ref_sift = sift.detectAndCompute(ref_gray, None)
kp_curr_sift, desc_curr_sift = sift.detectAndCompute(curr_gray, None)
print(f"SIFT Keypoints - Ref: {len(kp_ref_sift)}, Curr: {len(kp_curr_sift)}")

matcher_l2 = cv2.BFMatcher(cv2.NORM_L2)

for ratio in [0.70, 0.75, 0.80, 0.82, 0.85]:
    matches = matcher_l2.knnMatch(desc_curr_sift, desc_ref_sift, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    
    num_matches = len(good)
    if num_matches >= 4:
        pts_curr = np.float32([kp_curr_sift[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref_sift[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts_curr, pts_ref, cv2.RANSAC, 5.0)
        inliers = np.sum(mask) if mask is not None else 0
        inlier_ratio = inliers / num_matches if num_matches > 0 else 0
        print(f"SIFT Ratio={ratio:.2f}: matches={num_matches}, RANSAC inliers={inliers} (ratio={inlier_ratio:.3f})")
    else:
        print(f"SIFT Ratio={ratio:.2f}: Insufficient matches ({num_matches})")

# 2. ORB
orb = cv2.ORB_create(nfeatures=4000)
kp_ref_orb, desc_ref_orb = orb.detectAndCompute(ref_gray, None)
kp_curr_orb, desc_curr_orb = orb.detectAndCompute(curr_gray, None)
print(f"\nORB Keypoints - Ref: {len(kp_ref_orb)}, Curr: {len(kp_curr_orb)}")

matcher_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
for ratio in [0.75, 0.80, 0.85, 0.90]:
    try:
        matches = matcher_hamming.knnMatch(desc_curr_orb, desc_ref_orb, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        num_matches = len(good)
        if num_matches >= 4:
            pts_curr = np.float32([kp_curr_orb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_ref = np.float32([kp_ref_orb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(pts_curr, pts_ref, cv2.RANSAC, 5.0)
            inliers = np.sum(mask) if mask is not None else 0
            inlier_ratio = inliers / num_matches if num_matches > 0 else 0
            print(f"ORB Ratio={ratio:.2f}: matches={num_matches}, RANSAC inliers={inliers} (ratio={inlier_ratio:.3f})")
        else:
            print(f"ORB Ratio={ratio:.2f}: Insufficient matches ({num_matches})")
    except Exception as e:
        print(f"ORB Ratio={ratio:.2f} failed: {e}")
