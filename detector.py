import time
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict, Any

from config import DetectionConfig
from preprocessing import ImagePreprocessor
from alignment import ImageAligner
from difference import DifferenceDetector
from segmentation import RegionProposer
from validation import FODValidator
from utils import get_logger, draw_detections, create_diff_heatmap, save_image

logger = get_logger("Detector")


# ---------------------------------------------------------------------------
# Standalone Color-Anomaly Detector
# Catches objects whose color distribution is absent from the reference image
# (e.g., orange cone on a green-tinted underbody scan).
# This runs INDEPENDENTLY of alignment quality and is always executed.
# ---------------------------------------------------------------------------
class ColorAnomalyDetector:
    """
    Detects regions in the current image whose dominant color is statistically
    anomalous with respect to the reference image.  Works even when both images
    have different sizes / slight viewpoint changes.
    """

    # HSV hue-saturation ranges considered "always foreign" on industrial scans.
    # These are highly-saturated, warm colors that do not appear in bare metal /
    # painted vehicle underbody images.
    FOREIGN_HSV_RANGES = [
        # Orange (safety cones, plastic objects, etc.)
        {"lo": np.array([5,  120, 60]),  "hi": np.array([25, 255, 255]), "label": "orange"},
        # Red (part 1 — wraps around 0°)
        {"lo": np.array([0,  140, 60]),  "hi": np.array([8,  255, 255]), "label": "red_lo"},
        # Red (part 2 — above 170°)
        {"lo": np.array([170, 140, 60]), "hi": np.array([180, 255, 255]), "label": "red_hi"},
        # Bright yellow
        {"lo": np.array([25, 100, 100]), "hi": np.array([35, 255, 255]), "label": "yellow"},
        # Bright green (non-scanner tint) — very saturated
        {"lo": np.array([40, 150, 80]),  "hi": np.array([80, 255, 255]), "label": "green"},
        # Blue / cobalt
        {"lo": np.array([95, 100, 60]),  "hi": np.array([130, 255, 255]), "label": "blue"},
        # Magenta / pink
        {"lo": np.array([140, 80, 60]),  "hi": np.array([170, 255, 255]), "label": "magenta"},
    ]

    # Minimum pixel area to flag as a candidate
    MIN_AREA = 200
    # If the same hue range covers > this fraction of the REFERENCE, it is a
    # known background color and will NOT be flagged.
    REF_COVERAGE_THRESHOLD = 0.005   # 0.5 % of image

    def detect(
        self,
        ref_img: np.ndarray,
        curr_img: np.ndarray,
        image_shape: Tuple[int, int],
        border_fraction: float = 0.05,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Args:
            ref_img:      BGR reference image (may differ in size from curr_img)
            curr_img:     BGR current/live image
            image_shape:  (H, W) target canvas size for the output mask

        Returns:
            (detections_list, combined_mask_on_canvas)
        """
        H, W = image_shape
        curr_hsv = cv2.cvtColor(curr_img, cv2.COLOR_BGR2HSV)
        ref_hsv  = cv2.cvtColor(ref_img,  cv2.COLOR_BGR2HSV)

        # Resize reference HSV to same size as current for per-pixel comparison
        ref_hsv_rs = cv2.resize(ref_hsv, (curr_img.shape[1], curr_img.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        ref_area = ref_img.shape[0] * ref_img.shape[1]
        curr_area = curr_img.shape[0] * curr_img.shape[1]

        detections: List[Dict] = []
        combined_mask_curr = np.zeros((curr_img.shape[0], curr_img.shape[1]), dtype=np.uint8)

        for hrange in self.FOREIGN_HSV_RANGES:
            lo, hi = hrange["lo"], hrange["hi"]
            label = hrange["label"]

            # Build mask in current image
            curr_mask = cv2.inRange(curr_hsv, lo, hi)
            # Build mask in reference image (at original ref size)
            ref_mask  = cv2.inRange(ref_hsv, lo, hi)

            ref_coverage = np.sum(ref_mask > 0) / max(ref_area, 1)
            if ref_coverage > self.REF_COVERAGE_THRESHOLD:
                # This color exists prominently in the reference → not a foreign object
                logger.debug(f"Color range '{label}' is background in ref ({ref_coverage:.2%}). Skipping.")
                continue

            # Suppress areas in curr where the REFERENCE (resized) also has this color
            # (handles partial-overlap viewpoint differences)
            ref_mask_rs = cv2.inRange(ref_hsv_rs, lo, hi)
            curr_mask_clean = cv2.bitwise_and(curr_mask, cv2.bitwise_not(ref_mask_rs))

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            curr_mask_clean = cv2.morphologyEx(curr_mask_clean, cv2.MORPH_OPEN,  kernel, iterations=1)
            curr_mask_clean = cv2.morphologyEx(curr_mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

            n_pixels = int(np.sum(curr_mask_clean > 0))
            if n_pixels < self.MIN_AREA:
                continue

            # Find contours for bounding boxes
            contours, _ = cv2.findContours(curr_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.MIN_AREA:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)

                # Skip detections in the border strip (UI overlays, scanner artifacts)
                border_px_x = int(curr_img.shape[1] * border_fraction)
                border_px_y = int(curr_img.shape[0] * border_fraction)
                cx_center = x + bw // 2
                cy_center = y + bh // 2
                if (cx_center < border_px_x or cx_center > curr_img.shape[1] - border_px_x or
                        cy_center < border_px_y or cy_center > curr_img.shape[0] - border_px_y):
                    logger.debug(f"Skipping border detection at ({x},{y},{bw},{bh})")
                    continue

                # Scale bounding box to the output canvas coordinate system
                sx = W / curr_img.shape[1]
                sy = H / curr_img.shape[0]
                cx = int(x  * sx)
                cy = int(y  * sy)
                cw = int(bw * sx)
                ch = int(bh * sy)

                # Confidence based on coverage relative to bbox area
                bbox_area = bw * bh
                fill_ratio = float(area) / max(bbox_area, 1)
                confidence = min(0.98, 0.65 + fill_ratio * 0.33)

                detections.append({
                    "x": cx,
                    "y": cy,
                    "width": cw,
                    "height": ch,
                    "area": int(area * sx * sy),
                    "confidence": round(confidence, 3),
                    "label": f"foreign_object_color_{label}",
                    "method": "color_anomaly"
                })
                logger.info(
                    f"Color anomaly detected: color='{label}', "
                    f"bbox=({cx},{cy},{cw},{ch}), confidence={confidence:.3f}"
                )

            combined_mask_curr = cv2.bitwise_or(combined_mask_curr, curr_mask_clean)

        # Scale combined mask to canvas
        combined_mask_canvas = cv2.resize(combined_mask_curr, (W, H), interpolation=cv2.INTER_NEAREST)

        # Deduplicate detections across color ranges using IoU
        detections = self._dedup(detections, iou_threshold=0.15)

        return detections, combined_mask_canvas

    def _iou(self, a: Dict, b: Dict) -> float:
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["width"], by1 + b["height"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter_area = inter_w * inter_h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area
        return float(inter_area) / max(union_area, 1)

    def _dedup(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping detections from different color ranges into one."""
        kept: List[Dict] = []
        for det in sorted(detections, key=lambda d: -d["confidence"]):
            overlap = False
            for k in kept:
                if self._iou(det, k) > iou_threshold:
                    overlap = True
                    # Expand the kept box to include the new one
                    x1 = min(k["x"], det["x"])
                    y1 = min(k["y"], det["y"])
                    x2 = max(k["x"] + k["width"],  det["x"] + det["width"])
                    y2 = max(k["y"] + k["height"], det["y"] + det["height"])
                    k["x"], k["y"] = x1, y1
                    k["width"],  k["height"] = x2 - x1, y2 - y1
                    k["confidence"] = max(k["confidence"], det["confidence"])
                    break
            if not overlap:
                kept.append(dict(det))
        return kept


# ---------------------------------------------------------------------------
# Main Detector Orchestrator
# ---------------------------------------------------------------------------
class FODDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.aligner = ImageAligner(config.alignment)
        self.diff_detector = DifferenceDetector(config.difference)
        self.region_proposer = RegionProposer(config.region)
        self.validator = FODValidator(config.validation)
        self.color_detector = ColorAnomalyDetector()

    def detect(
        self,
        ref_images: List[np.ndarray],
        curr_image: np.ndarray,
        ignore_mask: Optional[np.ndarray] = None,
        output_prefix: str = "fod"
    ) -> Dict[str, Any]:
        """
        Runs the full hybrid Foreign Object Detection pipeline:
          Pass A — Color-anomaly detection (orientation/alignment independent)
          Pass B — Classical CV: alignment → diff map → contour proposals → DL validation
          Final  — Merge unique detections from both passes
        """
        start_time = time.time()

        # ---- Preprocess -------------------------------------------------------
        logger.info("Preprocessing images.")
        curr_preprocessed = self.preprocessor.preprocess(curr_image)
        ref_preprocessed_list = [self.preprocessor.preprocess(r) for r in ref_images]
        H_canvas, W_canvas = curr_preprocessed.shape[:2]

        # ---- Handle Ignore Mask -----------------------------------------------
        preprocessed_ignore_mask = None
        if ignore_mask is not None:
            preprocessed_ignore_mask = cv2.resize(
                ignore_mask, (W_canvas, H_canvas), interpolation=cv2.INTER_NEAREST
            )
            if len(preprocessed_ignore_mask.shape) == 3:
                preprocessed_ignore_mask = cv2.cvtColor(preprocessed_ignore_mask, cv2.COLOR_BGR2GRAY)
            _, preprocessed_ignore_mask = cv2.threshold(
                preprocessed_ignore_mask, 127, 255, cv2.THRESH_BINARY
            )

        # ====================================================================
        # PASS A — COLOR ANOMALY (works independently of alignment)
        # ====================================================================
        logger.info("Pass A: Color-anomaly detection.")
        color_detections: List[Dict] = []
        color_mask = np.zeros((H_canvas, W_canvas), dtype=np.uint8)
        try:
            best_ref_orig = ref_images[0]   # use original (not preprocessed) for color
            ca_dets, ca_mask = self.color_detector.detect(
                best_ref_orig, curr_image, (H_canvas, W_canvas)
            )
            color_detections.extend(ca_dets)
            color_mask = cv2.bitwise_or(color_mask, ca_mask)
            logger.info(f"Pass A complete: {len(color_detections)} color anomaly detections.")
        except Exception as e:
            logger.error(f"Color anomaly detection error: {e}")

        # ====================================================================
        # PASS B — ALIGNMENT + CLASSICAL CV + DL VALIDATION
        # ====================================================================
        logger.info("Pass B: Alignment-based classical CV detection.")

        # 1. Align
        alignment_score = 0.0
        best_ref_preprocessed = ref_preprocessed_list[0]
        try:
            if len(ref_preprocessed_list) == 1:
                warped_curr, alignment_score = self.aligner.align(
                    best_ref_preprocessed, curr_preprocessed
                )
                best_idx = 0
            else:
                best_ref_preprocessed, warped_curr, alignment_score, best_idx = \
                    self.aligner.select_best_reference(ref_preprocessed_list, curr_preprocessed)
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            warped_curr = cv2.resize(
                curr_preprocessed,
                (best_ref_preprocessed.shape[1], best_ref_preprocessed.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            alignment_score = 0.1

        # If aligner returned None (should not happen with new fallback, but safety net)
        if warped_curr is None:
            logger.warning("Alignment returned None. Using resize fallback.")
            warped_curr = cv2.resize(
                curr_preprocessed,
                (best_ref_preprocessed.shape[1], best_ref_preprocessed.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            alignment_score = 0.05

        # 2. Difference detection
        try:
            combined_diff, thresholded_mask = self.diff_detector.detect_differences(
                best_ref_preprocessed, warped_curr
            )
        except Exception as e:
            logger.error(f"Difference detection error: {e}")
            combined_diff = np.zeros((H_canvas, W_canvas), dtype=np.uint8)
            thresholded_mask = np.zeros((H_canvas, W_canvas), dtype=np.uint8)

        # 3. Region proposals
        candidates = []
        try:
            candidates = self.region_proposer.propose_regions(thresholded_mask)
            logger.info(f"Region proposer found {len(candidates)} candidates.")
        except Exception as e:
            logger.error(f"Region proposal error: {e}")

        # 4. Deep-learning validation of candidates
        validated_detections: List[Dict] = []
        final_mask = np.zeros_like(thresholded_mask)

        for cand in candidates:
            try:
                is_valid, confidence, reason = self.validator.validate_region(
                    cand, best_ref_preprocessed, warped_curr, preprocessed_ignore_mask
                )
                if is_valid and confidence >= 0.50:
                    validated_detections.append({
                        "x": int(cand["x"]),
                        "y": int(cand["y"]),
                        "width": int(cand["width"]),
                        "height": int(cand["height"]),
                        "area": int(cand["area"]),
                        "confidence": float(confidence),
                        "label": "foreign_object",
                        "method": "diff_map"
                    })
                    final_mask = cv2.bitwise_or(final_mask, cand["mask"])
                elif is_valid:
                    logger.info(f"Candidate at ({cand['x']},{cand['y']}) rejected: LOW_CONFIDENCE ({confidence:.3f})")
                else:
                    logger.info(f"Candidate at ({cand['x']},{cand['y']}) rejected: {reason}")
            except Exception as e:
                logger.error(f"Validation error for candidate: {e}")

        logger.info(f"Pass B complete: {len(validated_detections)} validated diff-map detections.")

        # Skip diff-map detections when alignment is too poor (too many false positives)
        ALIGNMENT_QUALITY_GATE = 0.45
        if alignment_score < ALIGNMENT_QUALITY_GATE and len(validated_detections) > 0:
            logger.warning(
                f"Alignment score {alignment_score:.3f} < {ALIGNMENT_QUALITY_GATE}. "
                f"Dropping {len(validated_detections)} diff-map detections to avoid false positives. "
                f"Color-anomaly detections are retained."
            )
            validated_detections = []
            final_mask = np.zeros((H_canvas, W_canvas), dtype=np.uint8)

        # ====================================================================
        # MERGE — Combine Pass A + Pass B, deduplicate overlapping boxes
        # ====================================================================
        all_detections = self._merge_detections(color_detections, validated_detections, self.config.region.merge_distance_threshold)

        # Ensure masks are the same size before OR-merge
        if final_mask.shape != color_mask.shape:
            final_mask = cv2.resize(final_mask, (color_mask.shape[1], color_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        all_mask = cv2.bitwise_or(color_mask, final_mask)

        # ====================================================================
        # Similarity score
        # ====================================================================
        try:
            ref_gray   = cv2.cvtColor(best_ref_preprocessed, cv2.COLOR_BGR2GRAY)
            warped_gray = cv2.cvtColor(warped_curr, cv2.COLOR_BGR2GRAY)
            from skimage.metrics import structural_similarity as skssim
            ssim_val, _ = skssim(ref_gray, warped_gray, full=True)
            similarity_score = max(0.0, min(100.0, float(ssim_val) * 100.0))
        except Exception:
            similarity_score = 0.0

        # ====================================================================
        # Visualizations
        # ====================================================================
        # Normalize visualization base to canvas size (H_canvas x W_canvas)
        # so it always matches all_mask dimensions
        if warped_curr is not None and warped_curr.shape[:2] == (H_canvas, W_canvas):
            viz_base = warped_curr
        else:
            # Resize curr_preprocessed to the canvas size
            viz_base = cv2.resize(curr_preprocessed, (W_canvas, H_canvas), interpolation=cv2.INTER_LINEAR)

        # Ensure all_mask matches viz_base exactly
        if all_mask.shape[:2] != (H_canvas, W_canvas):
            all_mask = cv2.resize(all_mask, (W_canvas, H_canvas), interpolation=cv2.INTER_NEAREST)

        marked_image = draw_detections(viz_base, all_detections, all_mask)
        diff_heatmap = create_diff_heatmap(combined_diff)

        # Save outputs
        from config import OUTPUT_DIR
        ts = int(time.time() * 1000)
        marked_path = OUTPUT_DIR / f"{output_prefix}_{ts}_marked.jpg"
        diff_path   = OUTPUT_DIR / f"{output_prefix}_{ts}_diff.jpg"
        mask_path   = OUTPUT_DIR / f"{output_prefix}_{ts}_mask.png"
        save_image(marked_image, marked_path)
        save_image(diff_heatmap, diff_path)
        save_image(all_mask, mask_path)

        processing_time_ms = int((time.time() - start_time) * 1000)
        status = "FOD_DETECTED" if len(all_detections) > 0 else "NO_FOD"

        response = {
            "status": status,
            "accuracy_mode": "HYBRID_COLOR+DIFFMAP",
            "similarity_score": round(similarity_score, 2),
            "alignment_score": round(float(alignment_score), 3),
            "objects": len(all_detections),
            "detections": all_detections,
            "output_image": f"output/{marked_path.name}",
            "difference_map": f"output/{diff_path.name}",
            "mask_image": f"output/{mask_path.name}",
            "processing_time_ms": processing_time_ms
        }

        logger.info(
            f"Detection complete. Status={status}, Total objects={len(all_detections)}, "
            f"Color={len(color_detections)}, DiffMap={len(validated_detections)}, "
            f"Time={processing_time_ms}ms"
        )
        return response

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _merge_detections(
        self,
        color_dets: List[Dict],
        diffmap_dets: List[Dict],
        dist_thresh: int = 75
    ) -> List[Dict]:
        """
        Merge detections from color and diff-map passes.
        Also merges close bounding boxes together.
        """
        all_dets = color_dets + diffmap_dets
        
        def boxes_are_close(box1, box2):
            x1_min, y1_min = box1["x"], box1["y"]
            x1_max, y1_max = x1_min + box1["width"], y1_min + box1["height"]
            
            x2_min, y2_min = box2["x"], box2["y"]
            x2_max, y2_max = x2_min + box2["width"], y2_min + box2["height"]
            
            x_dist = max(0, x2_min - x1_max, x1_min - x2_max)
            y_dist = max(0, y2_min - y1_max, y1_min - y2_max)
            
            dist = np.sqrt(x_dist**2 + y_dist**2)
            return dist <= dist_thresh

        groups = []
        for det in all_dets:
            placed = False
            for group in groups:
                for member in group:
                    if boxes_are_close(det, member):
                        group.append(det)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                groups.append([det])
                
        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
                
            x_min = min(m["x"] for m in group)
            y_min = min(m["y"] for m in group)
            x_max = max(m["x"] + m["width"] for m in group)
            y_max = max(m["y"] + m["height"] for m in group)
            
            max_conf = max(m["confidence"] for m in group)
            
            # Prefer color anomaly labels and methods if mixed
            label = group[0]["label"]
            method = group[0]["method"]
            for m in group:
                if "color" in m["method"]:
                    label = m["label"]
                    method = m["method"]
                    break
                    
            merged.append({
                "x": x_min,
                "y": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "area": sum(m["area"] for m in group),
                "confidence": max_conf,
                "label": label,
                "method": method
            })
            
        return merged

    def _iou(self, a: Dict, b: Dict) -> float:
        """Compute Intersection-over-Union for two bounding box dicts."""
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["width"], by1 + b["height"]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        return float(inter_area) / max(union_area, 1)
