import cv2
import numpy as np
from config import RegionConfig
from utils import get_logger

logger = get_logger("Segmentation")

class RegionProposer:
    def __init__(self, config: RegionConfig):
        self.config = config

    def propose_regions(self, binary_mask: np.ndarray) -> list[dict]:
        """
        Extracts candidate foreign object regions from the binary mask.
        Filters candidate regions based on shape and size constraints.
        Merges close or overlapping candidate bounding boxes.
        Returns:
            - A list of dictionary objects representing the candidate regions:
              [{"x": x, "y": y, "width": w, "height": h, "area": area, "contour": contour, "mask": region_mask}]
        """
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 1. Filter by area
            if area < self.config.min_area or area > self.config.max_area:
                continue
                
            # Compute geometric parameters
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # 2. Filter by aspect ratio
            aspect_ratio = max(float(w) / float(h), float(h) / float(w)) if h > 0 and w > 0 else 0.0
            if aspect_ratio > self.config.max_aspect_ratio:
                continue

            # 3. Filter by solidity (ratio of contour area to its convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0
            if solidity < self.config.min_solidity:
                continue

            # 4. Filter by circularity
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
            if circularity < self.config.min_circularity:
                continue

            # If all checks pass, it's a valid candidate
            # Create a mask for this specific region
            region_mask = np.zeros_like(binary_mask)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            
            candidates.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": int(area),
                "solidity": solidity,
                "circularity": circularity,
                "contour": contour,
                "mask": region_mask
            })
            
        logger.info(f"Proposed {len(candidates)} regions after geometric filtering (out of {len(contours)} initial contours).")
        
        # Merge close bounding boxes
        if len(candidates) > 1:
            merged_candidates = self._merge_close_boxes(candidates, self.config.merge_distance_threshold, binary_mask.shape)
            logger.info(f"Merged regions count: {len(candidates)} -> {len(merged_candidates)}")
            
            final_candidates = []
            for cand in merged_candidates:
                w, h = cand["width"], cand["height"]
                aspect_ratio = max(float(w) / float(h), float(h) / float(w)) if h > 0 and w > 0 else 0.0
                if aspect_ratio > self.config.max_aspect_ratio:
                    continue
                if cand["solidity"] < self.config.min_solidity:
                    continue
                if cand["circularity"] < self.config.min_circularity:
                    continue
                final_candidates.append(cand)
            
            logger.info(f"After post-merge geometric filtering: {len(final_candidates)} candidates remain.")
            return final_candidates
            
        return candidates

    def _merge_close_boxes(self, candidates: list[dict], threshold: int, img_shape: tuple) -> list[dict]:
        """
        Merges bounding boxes that are close to each other.
        """
        def boxes_are_close(box1, box2, dist_thresh):
            # Calculate distance between boundaries
            x1_min, y1_min = box1["x"], box1["y"]
            x1_max, y1_max = x1_min + box1["width"], y1_min + box1["height"]
            
            x2_min, y2_min = box2["x"], box2["y"]
            x2_max, y2_max = x2_min + box2["width"], y2_min + box2["height"]
            
            # Horizontal distance
            x_dist = max(0, x2_min - x1_max, x1_min - x2_max)
            # Vertical distance
            y_dist = max(0, y2_min - y1_max, y1_min - y2_max)
            
            # Check if euclidean-like distance is within threshold
            dist = np.sqrt(x_dist**2 + y_dist**2)
            return dist <= dist_thresh

        # Active tracking of groups of candidates to merge
        groups = []
        for cand in candidates:
            # Check if this candidate fits in any existing group
            placed = False
            for group in groups:
                for member in group:
                    if boxes_are_close(cand, member, threshold):
                        group.append(cand)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                groups.append([cand])

        # Resolve grouped candidates into single merged candidates
        merged_list = []
        h_img, w_img = img_shape[:2]
        for group in groups:
            if len(group) == 1:
                merged_list.append(group[0])
                continue
            
            # Calculate outer bounding box of the group
            x_min = min(m["x"] for m in group)
            y_min = min(m["y"] for m in group)
            x_max = max(m["x"] + m["width"] for m in group)
            y_max = max(m["y"] + m["height"] for m in group)
            
            merged_w = x_max - x_min
            merged_h = y_max - y_min
            
            # Combine the masks
            combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            for m in group:
                combined_mask = cv2.bitwise_or(combined_mask, m["mask"])
            
            # Recompute total area (sum of actual mask pixels inside the combined bounding box)
            total_area = int(np.sum(combined_mask > 0))
            
            # Find the new outer contour
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = max(contours, key=cv2.contourArea) if contours else group[0]["contour"]
            
            # Compute solidity and circularity for the merged shape
            perimeter = cv2.arcLength(best_contour, True)
            hull = cv2.convexHull(best_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(total_area) / hull_area if hull_area > 0 else 0.0
            circularity = (4 * np.pi * total_area) / (perimeter ** 2) if perimeter > 0 else 0.0

            merged_list.append({
                "x": x_min,
                "y": y_min,
                "width": merged_w,
                "height": merged_h,
                "area": total_area,
                "solidity": solidity,
                "circularity": circularity,
                "contour": best_contour,
                "mask": combined_mask
            })
            
        return merged_list
