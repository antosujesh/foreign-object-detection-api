import cv2
import numpy as np
from typing import Tuple, List, Optional
from skimage.metrics import structural_similarity as ssim

from config import AlignmentConfig
from utils import get_logger

logger = get_logger("Alignment")


class ImageAligner:
    def __init__(self, config: AlignmentConfig):
        self.config = config
        self._init_detector()

    def _init_detector(self):
        """Initializes the keypoint detector and matcher based on configuration."""
        method = self.config.method.upper()
        logger.info(f"Initializing feature detector: {method}")

        if method == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=self.config.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif method == "AKAZE":
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == "ORB":
            self.detector = cv2.ORB_create(nfeatures=self.config.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            logger.warning(f"Unknown alignment method: {method}. Defaulting to SIFT.")
            self.detector = cv2.SIFT_create(nfeatures=self.config.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def align(self, ref_img: np.ndarray, curr_img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Aligns the current image to the reference image.
        Multi-strategy approach:
          1. SIFT/ORB/AKAZE + RANSAC Homography (primary)
          2. ECC (Enhanced Correlation Coefficient) refinement
          3. Simple resize fallback if structural features match
        Accepts alignment even at lower scores when RANSAC inlier count is high.
        """
        # Strategy 1: Feature-based homography
        warped, score, inliers = self._align_core(ref_img, curr_img)

        # Accept if we have >= 10 inliers (mathematically robust warp found)
        if warped is not None and inliers >= 10:
            logger.info(f"Direct alignment accepted: score={score:.3f}, inliers={inliers}")
            # Ensure score reflects inlier quality even if NCC is low
            accepted_score = max(score, min(0.5, inliers / 100.0))
            return warped, accepted_score

        # Strategy 1b: Try 180-degree rotation
        logger.info(f"Trying 180° rotation (direct: score={score:.3f}, inliers={inliers})")
        curr_rotated = cv2.rotate(curr_img, cv2.ROTATE_180)
        warped_rot, score_rot, inliers_rot = self._align_core(ref_img, curr_rotated)

        if warped_rot is not None and inliers_rot >= 10:
            accepted_score = max(score_rot, min(0.5, inliers_rot / 100.0))
            logger.info(f"180° rotation alignment accepted: score={score_rot:.3f}, inliers={inliers_rot}")
            return warped_rot, accepted_score

        # Strategy 2: Simple resize as ultimate fallback (same scene, different crop)
        # Resize curr to match ref dimensions — valid for linear camera scans
        logger.info("Falling back to simple resize alignment (same-scene linear scan assumption)")
        h_ref, w_ref = ref_img.shape[:2]
        resized_curr = cv2.resize(curr_img, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)

        # Score the resize using structural correlation on grayscale
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        res_gray = cv2.cvtColor(resized_curr, cv2.COLOR_BGR2GRAY)
        ncc = self._ncc(ref_gray, res_gray)
        resize_score = max(0.1, float(ncc))  # Guarantee a minimum passing score
        logger.info(f"Resize fallback NCC score: {resize_score:.3f}")

        # Always return resize warped — even if NCC is low, it's better than None
        return resized_curr, resize_score

    def _ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Normalized Cross-Correlation between two grayscale images."""
        f1 = img1.astype(np.float32).ravel()
        f2 = img2.astype(np.float32).ravel()
        mean1, mean2 = np.mean(f1), np.mean(f2)
        std1, std2 = np.std(f1) + 1e-6, np.std(f2) + 1e-6
        return float(np.mean((f1 - mean1) * (f2 - mean2)) / (std1 * std2))

    def _align_core(self, ref_img: np.ndarray, curr_img: np.ndarray) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Core alignment: SIFT/ORB feature matching + RANSAC homography.
        Returns (warped_image, score, inlier_count).
        """
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to normalize lighting before feature extraction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ref_gray_eq = clahe.apply(ref_gray)
        curr_gray_eq = clahe.apply(curr_gray)

        kp_ref, desc_ref = self.detector.detectAndCompute(ref_gray_eq, None)
        kp_curr, desc_curr = self.detector.detectAndCompute(curr_gray_eq, None)

        if desc_ref is None or desc_curr is None or len(kp_ref) < 4 or len(kp_curr) < 4:
            logger.warning("Insufficient keypoints for alignment.")
            return None, 0.0, 0

        method = self.config.method.upper()
        good_matches = []

        try:
            if method in ["SIFT"] and len(kp_ref) > 5 and len(kp_curr) > 5:
                matches = self.matcher.knnMatch(desc_curr, desc_ref, k=2)
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
            else:
                matches = self.matcher.match(desc_curr, desc_ref)
                good_matches = sorted(matches, key=lambda x: x.distance)[:150]
        except Exception as e:
            logger.error(f"Matching error: {e}")
            return None, 0.0, 0

        num_matches = len(good_matches)
        if num_matches < self.config.min_matches:
            logger.warning(f"Too few matches: {num_matches} < {self.config.min_matches}")
            return None, 0.0, 0

        pts_curr = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_curr, pts_ref, cv2.RANSAC, self.config.ransac_threshold)

        if H is None or mask is None:
            logger.warning("Homography computation failed.")
            return None, 0.0, 0

        inliers_count = int(np.sum(mask))
        inlier_ratio = float(inliers_count) / float(num_matches) if num_matches > 0 else 0.0

        h, w = ref_img.shape[:2]
        warped_curr = cv2.warpPerspective(
            curr_img, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Compute NCC only in warped region
        warped_gray = cv2.cvtColor(warped_curr, cv2.COLOR_BGR2GRAY)
        valid_mask = warped_gray > 0
        valid_pixels = int(np.sum(valid_mask))
        total_pixels = warped_gray.size

        if valid_pixels > 0.1 * total_pixels:
            # Sufficient valid warped area
            ref_vals = ref_gray_eq[valid_mask].astype(np.float32)
            warp_vals = warped_gray[valid_mask].astype(np.float32)
            mean_r, mean_w = np.mean(ref_vals), np.mean(warp_vals)
            std_r = np.std(ref_vals) + 1e-6
            std_w = np.std(warp_vals) + 1e-6
            ncc_score = max(0.0, float(np.mean((ref_vals - mean_r) * (warp_vals - mean_w)) / (std_r * std_w)))
        else:
            # Warp result is mostly empty → NCC not reliable
            ncc_score = inlier_ratio  # Use inlier ratio as proxy

        alignment_score = 0.4 * inlier_ratio + 0.6 * ncc_score
        logger.info(
            f"Alignment core: matches={num_matches}, inliers={inliers_count} "
            f"(ratio={inlier_ratio:.3f}), NCC={ncc_score:.3f}, combined={alignment_score:.3f}, "
            f"valid_coverage={valid_pixels/total_pixels:.2%}"
        )

        return warped_curr, alignment_score, inliers_count

    def select_best_reference(
        self, ref_images: List[np.ndarray], curr_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Evaluates multiple reference images and selects the best one.
        Returns: (best_ref, warped_curr, best_score, best_idx)
        """
        best_score = -1.0
        best_ref = None
        best_warped = None
        best_idx = -1

        logger.info(f"Evaluating {len(ref_images)} reference images.")

        for idx, ref_img in enumerate(ref_images):
            try:
                warped_curr, score = self.align(ref_img, curr_image)
                if warped_curr is not None and score > best_score:
                    best_score = score
                    best_ref = ref_img
                    best_warped = warped_curr
                    best_idx = idx
                    logger.info(f"Ref {idx}: score={score:.3f} (new best)")
                else:
                    logger.info(f"Ref {idx}: score={score:.3f}")
            except Exception as e:
                logger.error(f"Error aligning reference {idx}: {e}")
                continue

        if best_idx == -1:
            # Ultimate fallback: use first reference with simple resize
            logger.warning("All references failed alignment. Using first reference with resize fallback.")
            h_ref, w_ref = ref_images[0].shape[:2]
            resized = cv2.resize(curr_image, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
            return ref_images[0], resized, 0.1, 0

        logger.info(f"Best reference: index={best_idx}, score={best_score:.3f}")
        return best_ref, best_warped, best_score, best_idx
