import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Optional

from config import ValidationConfig
from utils import get_logger

logger = get_logger("Validation")

class FODValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for Deep Learning validation: {self.device}")
        
        self.feature_extractor = None
        self.yolo_model = None
        
        if self.config.enable_dl_validation:
            self._load_models()

    def _load_models(self):
        """Loads DL models locally."""
        if self.config.dl_validator_type == "deep_features":
            try:
                logger.info(f"Loading pre-trained feature extractor: {self.config.cnn_model_name}")
                if self.config.cnn_model_name == "mobilenet_v3_small":
                    weights = models.MobileNet_V3_Small_Weights.DEFAULT
                    # Load model with pre-trained weights
                    base_model = models.mobilenet_v3_small(weights=weights)
                elif self.config.cnn_model_name == "resnet18":
                    weights = models.ResNet18_Weights.DEFAULT
                    base_model = models.resnet18(weights=weights)
                else:
                    logger.warning(f"Unsupported CNN model: {self.config.cnn_model_name}. Defaulting to MobileNetV3 Small.")
                    weights = models.MobileNet_V3_Small_Weights.DEFAULT
                    base_model = models.mobilenet_v3_small(weights=weights)

                # Remove the final classification layer, keeping only the feature extractor
                # MobileNetV3 and ResNet18 both have an average pooling layer followed by a classifier
                if hasattr(base_model, "classifier"):
                    # For MobileNet, replace classifier with Identity to get pooling output
                    base_model.classifier = torch.nn.Identity()
                elif hasattr(base_model, "fc"):
                    # For ResNet, replace fc with Identity
                    base_model.fc = torch.nn.Identity()
                
                self.feature_extractor = base_model.to(self.device)
                self.feature_extractor.eval()
                
                # Image transformation pipeline for ImageNet-trained models
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                logger.info("Deep feature extractor loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load PyTorch feature extractor: {e}. Falling back to rule-based validation only.")
                self.feature_extractor = None

        elif self.config.dl_validator_type == "yolo":
            try:
                logger.info("Loading YOLOv8 model...")
                from ultralytics import YOLO
                # Load local YOLO model (downloads on first run, cached locally)
                self.yolo_model = YOLO(self.config.yolo_model_path)
                logger.info(f"YOLOv8 loaded successfully from {self.config.yolo_model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}. Falling back to rule-based validation only.")
                self.yolo_model = None

    def _extract_deep_features(self, patch: np.ndarray) -> Optional[torch.Tensor]:
        """Extracts deep features from an image patch."""
        if self.feature_extractor is None:
            return None
        try:
            # Convert BGR CV2 image to PIL RGB image
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_patch)
            
            # Apply transformations and add batch dimension
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(tensor)
                # Flatten features
                features = torch.flatten(features, 1)
            return features
        except Exception as e:
            logger.error(f"Error during deep feature extraction: {e}")
            return None

    def validate_deep_features(self, ref_patch: np.ndarray, curr_patch: np.ndarray) -> Tuple[bool, float]:
        """
        Validates if the current patch contains a real physical change or just lighting/shadow.
        Compares features of reference patch and current patch.
        Returns:
            - validated: True if it is a real foreign object (low similarity),
                         False if it is likely a shadow/light shift (high similarity).
            - similarity: Cosine similarity score (0.0 to 1.0)
        """
        if self.feature_extractor is None:
            # Fallback if model not loaded: assume it is a real object to avoid false negatives
            return True, 0.0

        ref_feat = self._extract_deep_features(ref_patch)
        curr_feat = self._extract_deep_features(curr_patch)

        if ref_feat is None or curr_feat is None:
            return True, 0.0

        # Compute cosine similarity
        cos = torch.nn.CosineSimilarity(dim=1)
        similarity = float(cos(ref_feat, curr_feat).cpu().item())
        
        # High similarity means the semantic structures are identical (just different lighting/shadow).
        # Low similarity means the shape or structure has changed (a new object is present).
        validated = similarity < self.config.deep_feature_threshold
        
        logger.info(f"Deep Feature validation: similarity={similarity:.3f}, threshold={self.config.deep_feature_threshold}, validated={validated}")
        return validated, similarity

    def validate_region(self, cand: dict, ref_img: np.ndarray, warped_img: np.ndarray, ignore_mask: Optional[np.ndarray] = None) -> Tuple[bool, float, str]:
        """
        Validates a candidate region based on:
        1. Boundary proximity (reject if too close to border)
        2. Ignore mask overlap (reject if overlapping user ignore zone)
        3. Deep Learning validation (Cosine similarity check or YOLO object check)
        Returns:
            - is_valid: bool
            - confidence_score: float
            - rejection_reason: str
        """
        x, y, w, h = cand["x"], cand["y"], cand["width"], cand["height"]
        img_h, img_w = warped_img.shape[:2]

        # 1. Boundary filter
        border = self.config.ignore_border_pixels
        if x < border or y < border or (x + w) > (img_w - border) or (y + h) > (img_h - border):
            return False, 0.0, "BOUNDARY_EXCLUSION"

        # 2. Ignore mask overlap filter
        if ignore_mask is not None:
            # Crop the candidate region mask and the ignore mask
            cand_mask = cand["mask"]
            # Ignore mask: values should be 0 for ignore zones and 255 for inspect zones
            # We calculate what percentage of the candidate's pixels lie in the ignore zone (value == 0)
            overlap_pixels = np.sum((cand_mask > 0) & (ignore_mask == 0))
            cand_pixels = np.sum(cand_mask > 0)
            
            if cand_pixels > 0:
                overlap_ratio = float(overlap_pixels) / float(cand_pixels)
                if overlap_ratio > 0.15: # If more than 15% of the object is in an ignore zone
                    logger.info(f"Region rejected by ignore mask overlap: {overlap_ratio * 100:.1f}% ignored.")
                    return False, 0.0, "IGNORE_MASK_OVERLAP"

        # 3. Deep Learning validation
        if self.config.enable_dl_validation:
            # Extract patches
            # Add a small padding to the patch to capture local context
            pad = 5
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(img_w, x + w + pad)
            y_max = min(img_h, y + h + pad)
            
            ref_patch = ref_img[y_min:y_max, x_min:x_max]
            curr_patch = warped_img[y_min:y_max, x_min:x_max]

            if ref_patch.size == 0 or curr_patch.size == 0:
                return True, 0.5, "PASSED_WITH_EMPTY_PATCH"

            if self.config.dl_validator_type == "deep_features":
                is_valid, similarity = self.validate_deep_features(ref_patch, curr_patch)
                # Compute confidence: lower similarity = higher confidence of foreign object
                confidence = max(0.0, min(1.0, 1.0 - similarity))
                
                # Boost confidence if similarity is very low
                if confidence > 0.4:
                    confidence = 0.5 + 0.5 * confidence # map [0.4, 1.0] -> [0.7, 1.0]
                
                if not is_valid:
                    return False, confidence, "DEEP_FEATURE_SIMILARITY_TOO_HIGH"
                else:
                    return True, confidence, "PASSED_DEEP_FEATURE_VALIDATION"

            elif self.config.dl_validator_type == "yolo" and self.yolo_model is not None:
                # YOLO validation
                # Run YOLO on the warped image
                try:
                    # Run YOLO on the patch or entire image
                    results = self.yolo_model(curr_patch, verbose=False, conf=self.config.yolo_conf)
                    
                    # If YOLO finds any class of object inside the patch
                    # (since we cropped around the candidate, any object detected here is a positive validation)
                    yolo_detected = False
                    max_conf = 0.0
                    for r in results:
                        if len(r.boxes) > 0:
                            yolo_detected = True
                            max_conf = float(r.boxes.conf.cpu().numpy().max())
                            break
                            
                    if yolo_detected:
                        logger.info(f"YOLOv8 confirmed object in patch with confidence {max_conf:.3f}")
                        return True, max_conf, "YOLO_CONFIRMED"
                    else:
                        # If YOLO doesn't detect it, but it was found by classical difference,
                        # since YOLO only detects COCO classes, we shouldn't discard it immediately.
                        # However, we can lower its confidence score
                        logger.info("YOLOv8 did not detect any known class in patch. Keeping with lower confidence.")
                        return True, 0.45, "PASSED_CLASSICAL_ONLY"
                except Exception as e:
                    logger.error(f"YOLO validation failed: {e}")
                    return True, 0.5, "PASSED_FALLBACK"
        
        # Rule-based fallback if DL is disabled or failed
        # Calculate a simple structural similarity value on the patch
        # using normalized correlation coefficient
        try:
            ref_gray_p = cv2.cvtColor(ref_patch, cv2.COLOR_BGR2GRAY)
            curr_gray_p = cv2.cvtColor(curr_patch, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(curr_gray_p, ref_gray_p, cv2.TM_CCOEFF_NORMED)
            ncc = float(res[0][0])
            confidence = 1.0 - max(0.0, ncc)
            
            # Simple threshold on template matching
            if ncc > 0.90:
                logger.info(f"Rejected by template matching similarity: {ncc:.3f}")
                return False, confidence, "HIGH_TEMPLATE_MATCH_SIMILARITY"
                
            return True, confidence, "PASSED_CLASSICAL_TEMPLATE_MATCH"
        except Exception as e:
            logger.error(f"Fallback validation failed: {e}")
            return True, 0.5, "PASSED_WITHOUT_VALIDATION"
        
        return True, 0.8, "PASSED_DEFAULT"
