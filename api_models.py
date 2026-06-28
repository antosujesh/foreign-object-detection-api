"""
Pydantic response/request models for all API endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    error_code: str = Field(..., description="Machine-readable error code")
    detail: str = Field(..., description="Human-readable error message")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service health status")
    version: str = Field("1.0.0", description="API version")
    engine: str = Field("HYBRID_COLOR+DIFFMAP", description="Detection engine identifier")
    message: str = Field(..., description="Human-readable status message")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class ConfigResponse(BaseModel):
    config: Dict[str, Any] = Field(..., description="Active detection configuration")
    source: str = Field(..., description="Where the config came from: 'default' or 'custom'")


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------
class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    area: int
    confidence: float
    label: str
    method: str


class DetectionResult(BaseModel):
    status: str = Field(..., description="'FOD_DETECTED' or 'NO_FOD'")
    accuracy_mode: str
    similarity_score: float = Field(..., description="SSIM similarity score 0-100")
    alignment_score: float = Field(..., description="Image alignment quality 0-1")
    objects: int = Field(..., description="Number of detected foreign objects")
    detections: List[BoundingBox]
    output_image: str = Field(..., description="Relative URL to marked image")
    difference_map: str = Field(..., description="Relative URL to heatmap image")
    mask_image: str = Field(..., description="Relative URL to binary mask")
    output_image_base64: Optional[str] = Field(None, description="Base64 encoded output image")
    processing_time_ms: int
    
class Base64DetectionRequest(BaseModel):
    reference_images: List[str] = Field(..., description="List of base64 encoded reference images")
    current_image: str = Field(..., description="Base64 encoded live inspection image")
    ignore_mask: Optional[str] = Field(None, description="Optional base64 encoded mask")
    min_area: Optional[int] = None
    threshold_method: Optional[str] = None
    enable_dl_validation: Optional[bool] = None
    weight_ssim: Optional[float] = None
    weight_abs_diff: Optional[float] = None
    weight_lab_diff: Optional[float] = None
    weight_edge_diff: Optional[float] = None
    min_align_score: Optional[float] = None
    deep_feature_threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# Job models (async inspection)
# ---------------------------------------------------------------------------
class JobSubmitResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field("PENDING", description="Initial job status")
    message: str = Field(..., description="Human-readable submission message")
    poll_url: str = Field(..., description="URL to poll for job status")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="PENDING | PROCESSING | DONE | FAILED")
    submitted_at: str
    completed_at: Optional[str] = None
    processing_time_ms: Optional[int] = None
    result: Optional[DetectionResult] = None
    error: Optional[str] = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    submitted_at: str
    completed_at: Optional[str] = None
    objects: Optional[int] = None
    fod_status: Optional[str] = None


class JobListResponse(BaseModel):
    total: int
    jobs: List[JobSummary]


class PurgeResponse(BaseModel):
    deleted: int
    message: str
