import cv2
from pathlib import Path

uploads_dir = Path("uploads")
ref_files = sorted(list(uploads_dir.glob("ref_0_*.png")))

if ref_files:
    ref_path = ref_files[-1]
    img = cv2.imread(str(ref_path))
    if img is not None:
        h, w = img.shape[:2]
        print(f"Original image size: width={w}, height={h}, aspect_ratio={w/h:.3f}")
    else:
        print("Failed to load image.")
else:
    print("No PNG files found.")
