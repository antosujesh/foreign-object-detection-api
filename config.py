import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for directory in [UPLOADS_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class PreprocessingConfig(BaseModel):
    crop_inspection_area: bool = Field(default=False, description="Whether to crop a specific inspection area")
    crop_coords: list[int] = Field(default=[0, 0, 0, 0], description="Bounding box [ymin, xmin, ymax, xmax] to crop if crop_inspection_area is true")
    resize_resolution: tuple[int, int] = Field(default=(1024, 1024), description="Standard resolution to resize images for inspection")
    apply_clahe: bool = Field(default=True, description="Apply CLAHE for contrast normalization")
    denoise_strength: float = Field(default=3.0, description="Denoise strength for Bilateral filter")

class AlignmentConfig(BaseModel):
    method: str = Field(default="SIFT", description="Feature detection method: SIFT, ORB, AKAZE")
    max_features: int = Field(default=2000, description="Maximum number of features to detect")
    ransac_threshold: float = Field(default=5.0, description="RANSAC outlier threshold")
    min_matches: int = Field(default=10, description="Minimum number of matches to attempt alignment")
    min_align_score: float = Field(default=0.4, description="Minimum acceptable alignment score (ratio of inliers or correlation)")

class DifferenceConfig(BaseModel):
    weight_ssim: float = Field(default=0.35, description="Weight of SSIM difference map")
    weight_abs_diff: float = Field(default=0.25, description="Weight of Absolute intensity difference map")
    weight_lab_diff: float = Field(default=0.25, description="Weight of LAB Delta-E color difference map")
    weight_edge_diff: float = Field(default=0.15, description="Weight of Canny edge difference map")
    
    # Thresholding
    threshold_method: str = Field(default="OTSU", description="Threshold method: OTSU, ADAPTIVE, STATIC")
    static_threshold_val: int = Field(default=30, description="Static threshold value (0-255) if method is STATIC")
    adaptive_block_size: int = Field(default=25, description="Block size for adaptive thresholding")
    adaptive_c: int = Field(default=5, description="C constant subtracted from mean in adaptive thresholding")
    
    # Morphological operations
    morph_kernel_size: int = Field(default=5, description="Kernel size for morphology operations")
    open_iterations: int = Field(default=1, description="Number of opening operations to remove noise")
    close_iterations: int = Field(default=2, description="Number of closing operations to bridge gaps")

class RegionConfig(BaseModel):
    min_area: int = Field(default=400, description="Minimum area in pixels for a valid candidate region")
    max_area: int = Field(default=500000, description="Maximum area in pixels for a valid candidate region")
    min_solidity: float = Field(default=0.5, description="Minimum solidity (contour_area / convex_hull_area)")
    min_circularity: float = Field(default=0.2, description="Minimum circularity (4*pi*area / perimeter^2)")
    max_aspect_ratio: float = Field(default=5.0, description="Maximum aspect ratio (width/height or height/width)")
    merge_distance_threshold: int = Field(default=75, description="Distance in pixels to merge close contours")

class ValidationConfig(BaseModel):
    enable_dl_validation: bool = Field(default=True, description="Enable deep learning validation of candidate regions")
    dl_validator_type: str = Field(default="yolo", description="DL validation method: 'deep_features' (ResNet/MobileNet cosine similarity) or 'yolo' (YOLOv8 object recognition)")
    cnn_model_name: str = Field(default="mobilenet_v3_small", description="PyTorch model for deep feature extraction: mobilenet_v3_small, resnet18")
    deep_feature_threshold: float = Field(default=0.75, description="Maximum feature similarity score above which difference is considered shadow/lighting change (and rejected)")
    yolo_model_path: str = Field(default="yolov8n.pt", description="Path to YOLOv8 model weights (will download if not present)")
    yolo_conf: float = Field(default=0.60, description="Confidence threshold for YOLO validation")
    ignore_border_pixels: int = Field(default=10, description="Number of pixels from border to ignore detections")

class DetectionConfig(BaseModel):
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    difference: DifferenceConfig = DifferenceConfig()
    region: RegionConfig = RegionConfig()
    validation: ValidationConfig = ValidationConfig()

DEFAULT_CONFIG = DetectionConfig()
