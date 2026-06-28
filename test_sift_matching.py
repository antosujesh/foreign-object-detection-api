import cv2
from pathlib import Path

uploads_dir = Path("uploads")
ref_path = sorted(list(uploads_dir.glob("ref_0_*.png")))[-1]
curr_path = sorted(list(uploads_dir.glob("curr_*.png")))[-1]

ref_img = cv2.imread(str(ref_path))
curr_img = cv2.imread(str(curr_path))

# Crop out top and bottom banners
ref_cropped = ref_img[45:283, :]
curr_cropped = curr_img[45:283, :]

# Resize
target_w = 1280
h_ref, w_ref = ref_cropped.shape[:2]
h_target = int(h_ref * (target_w / w_ref))
h_target = max(2, (h_target // 2) * 2)

ref_resized = cv2.resize(ref_cropped, (target_w, h_target), interpolation=cv2.INTER_LANCZOS4)
curr_resized = cv2.resize(curr_cropped, (target_w, h_target), interpolation=cv2.INTER_LANCZOS4)

# Get L channel
def preprocess(img):
    denoised = cv2.bilateralFilter(img, 9, 30, 30)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(l)

ref_l = preprocess(ref_resized)
curr_l = preprocess(curr_resized)

# SIFT
sift = cv2.SIFT_create(nfeatures=4000)
kp_ref, desc_ref = sift.detectAndCompute(ref_l, None)
kp_curr, desc_curr = sift.detectAndCompute(curr_l, None)

matcher = cv2.BFMatcher(cv2.NORM_L2)
matches = matcher.knnMatch(desc_curr, desc_ref, k=2)

good = []
for m, n in matches:
    if m.distance < 0.80 * n.distance:
        good.append(m)

# Sort by distance
good = sorted(good, key=lambda x: x.distance)

# Draw top 50 matches
match_img = cv2.drawMatches(curr_resized, kp_curr, ref_resized, kp_ref, good[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("sift_matches.jpg", match_img)
print(f"Saved matches image to 'sift_matches.jpg'. Matches count: {len(good)}")
