"""
AERO-EYE FOD — FastAPI Application Entry Point
================================================
Mounts all versioned API routers under /api/v1/ and serves
the static web UI at the root path.

Endpoints:
  GET  /                           → Web UI (static/index.html)
  GET  /api/v1/health              → Health check
  GET  /api/v1/config              → Get active config
  PUT  /api/v1/config              → Update active config
  POST /api/v1/config/reset        → Reset config to defaults
  POST /api/v1/detect              → Synchronous FOD detection
  POST /api/v1/inspect             → Async FOD inspection (returns job_id)
  GET  /api/v1/inspect/{job_id}    → Poll job status + result
  GET  /api/v1/inspect/{job_id}/images/{type}  → Serve output images
  DELETE /api/v1/inspect/{job_id}  → Delete a job
  GET  /api/v1/jobs                → List all jobs
  DELETE /api/v1/jobs              → Purge old completed jobs

Legacy:
  POST /detect-fod                 → Backward-compat alias for /api/v1/detect
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import UPLOADS_DIR, OUTPUT_DIR
from routers import health, config, detect, inspect, jobs

# ---------------------------------------------------------------------------
# App Metadata
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AERO-EYE FOD — Industrial Foreign Object Detection API",
    description=(
        "Production-grade REST API for detecting foreign objects in industrial inspection images. "
        "Uses a hybrid pipeline: **Color Anomaly Detection** + **Image Alignment** + "
        "**Classical Difference Maps** + **Deep Learning validation (MobileNetV3 / ResNet18 cosine similarity)**.\n\n"
        "### Quick Start\n"
        "1. `POST /api/v1/inspect` — submit an inspection job (non-blocking)\n"
        "2. `GET  /api/v1/inspect/{job_id}` — poll until `status == DONE`\n"
        "3. `GET  /api/v1/inspect/{job_id}/images/marked` — view annotated output\n\n"
        "Or use `POST /api/v1/detect` for a single blocking call."
    ),
    version="1.0.0",
    contact={"name": "AERO-EYE FOD Team"},
    license_info={"name": "Proprietary"},
    openapi_tags=[
        {"name": "System",                "description": "Health check and service info"},
        {"name": "Configuration",         "description": "Read and update detection configuration"},
        {"name": "Synchronous Detection", "description": "Blocking single-call FOD detection"},
        {"name": "Inspection Jobs",       "description": "Async job submission and result polling"},
        {"name": "Job History",           "description": "List and manage past inspection jobs"},
    ],
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Static file mounts
# ---------------------------------------------------------------------------
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)),  name="static")

# ---------------------------------------------------------------------------
# API Routers
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(config.router)
app.include_router(detect.router)
app.include_router(inspect.router)
app.include_router(jobs.router)

# ---------------------------------------------------------------------------
# Root — serve web UI
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "AERO-EYE FOD API v1.0.0 — visit /docs for Swagger UI"}
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# Legacy backward-compat route
# ---------------------------------------------------------------------------
from fastapi import File, Form, UploadFile, HTTPException
from typing import List, Optional
import copy
import time
import cv2

from config import DEFAULT_CONFIG
from detector import FODDetector
from routers.config import get_active_config
from utils import clean_old_files


def _save_legacy(upload_file: UploadFile, prefix: str) -> Path:
    ts = int(time.time() * 1000)
    suffix = Path(upload_file.filename or "img.jpg").suffix or ".jpg"
    dest = UPLOADS_DIR / f"{prefix}_{ts}{suffix}"
    dest.write_bytes(upload_file.file.read())
    upload_file.file.seek(0)
    return dest


@app.post("/detect-fod", tags=["Legacy"], summary="[DEPRECATED] Use POST /api/v1/detect instead")
async def legacy_detect_fod(
    reference_images: List[UploadFile] = File(...),
    current_image: UploadFile = File(...),
    ignore_mask: Optional[UploadFile] = File(None),
    min_area: Optional[int] = Form(None),
    threshold_method: Optional[str] = Form(None),
    enable_dl_validation: Optional[bool] = Form(None),
    weight_ssim: Optional[float] = Form(None),
    weight_abs_diff: Optional[float] = Form(None),
    weight_lab_diff: Optional[float] = Form(None),
    weight_edge_diff: Optional[float] = Form(None),
    min_align_score: Optional[float] = Form(None),
    deep_feature_threshold: Optional[float] = Form(None),
):
    """Backward-compatible alias — delegates to the same logic as /api/v1/detect."""
    clean_old_files(UPLOADS_DIR, max_age_seconds=7200)
    clean_old_files(OUTPUT_DIR, max_age_seconds=7200)

    config = copy.deepcopy(get_active_config())
    if min_area is not None:             config.region.min_area = min_area
    if threshold_method is not None:     config.difference.threshold_method = threshold_method
    if enable_dl_validation is not None: config.validation.enable_dl_validation = enable_dl_validation
    if weight_ssim is not None:          config.difference.weight_ssim = weight_ssim
    if weight_abs_diff is not None:      config.difference.weight_abs_diff = weight_abs_diff
    if weight_lab_diff is not None:      config.difference.weight_lab_diff = weight_lab_diff
    if weight_edge_diff is not None:     config.difference.weight_edge_diff = weight_edge_diff
    if min_align_score is not None:      config.alignment.min_align_score = min_align_score
    if deep_feature_threshold is not None: config.validation.deep_feature_threshold = deep_feature_threshold

    try:
        ref_cv_images = []
        for i, rf in enumerate(reference_images):
            saved = _save_legacy(rf, f"ref_{i}")
            img = cv2.imread(str(saved))
            if img is None:
                raise HTTPException(status_code=400, detail=f"Invalid reference image at index {i}.")
            ref_cv_images.append(img)

        curr_saved = _save_legacy(current_image, "curr")
        curr_cv = cv2.imread(str(curr_saved))
        if curr_cv is None:
            raise HTTPException(status_code=400, detail="Invalid current image.")

        ignore_mask_cv = None
        if ignore_mask is not None:
            mask_saved = _save_legacy(ignore_mask, "mask")
            ignore_mask_cv = cv2.imread(str(mask_saved), cv2.IMREAD_GRAYSCALE)

        detector = FODDetector(config)
        return detector.detect(
            ref_images=ref_cv_images,
            curr_image=curr_cv,
            ignore_mask=ignore_mask_cv,
            output_prefix=curr_saved.stem,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
