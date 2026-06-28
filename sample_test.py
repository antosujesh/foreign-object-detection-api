# -*- coding: utf-8 -*-
"""
sample_test.py — End-to-end API test with synthetic sample data.
=================================================================
Generates realistic synthetic reference + current images (with a
simulated foreign object), then exercises every API endpoint and
prints a formatted results report.

Usage:
    py sample_test.py
"""
from __future__ import annotations

import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import requests

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


# ============================================================
# Helpers
# ============================================================

def color(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"

def ok(text: str)   -> str: return color(f"  [PASS] {text}", 32)
def err(text: str)  -> str: return color(f"  [FAIL] {text}", 31)
def info(text: str) -> str: return color(f"  [INFO] {text}", 36)
def head(text: str) -> str: return color(f"\n{'='*55}\n  {text}\n{'='*55}", 33)


def check_server():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        print(err(f"Server not reachable at {BASE_URL}: {e}"))
        return False


# ============================================================
# Synthetic Image Generator
# ============================================================

def make_reference_image(w=640, h=480) -> np.ndarray:
    """
    Simulate a clean industrial underbody scan:
    - Dark metallic gradient background
    - Structural lines (chassis ribs, pipes)
    - No foreign objects
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background (dark grey metallic)
    for y in range(h):
        v = int(40 + 20 * (y / h))
        img[y, :] = (v, v, v)

    # Horizontal structural ribs
    for y_pos in [80, 160, 240, 320, 400]:
        cv2.line(img, (0, y_pos), (w, y_pos), (80, 80, 80), 8)

    # Vertical pipe-like structures
    for x_pos in [120, 280, 440]:
        cv2.rectangle(img, (x_pos - 10, 0), (x_pos + 10, h), (60, 65, 60), -1)

    # Bolt-hole markers
    for bx in [100, 260, 420]:
        for by in [100, 200, 350]:
            cv2.circle(img, (bx, by), 12, (30, 30, 30), -1)
            cv2.circle(img, (bx, by), 12, (90, 90, 90), 2)

    # Light noise (camera sensor noise)
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def make_current_image_with_fod(ref: np.ndarray) -> np.ndarray:
    """
    Take the reference and add a simulated orange safety cone (FOD).
    Also add slight illumination change to test shadow rejection.
    """
    curr = ref.copy()

    # Slight global brightness shift (lighting change)
    curr = np.clip(curr.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    # Draw orange safety cone / FOD object
    # Orange in BGR: (0, 165, 255)
    cone_x, cone_y = 350, 180
    cone_pts = np.array([
        [cone_x,       cone_y - 60],
        [cone_x - 30,  cone_y + 40],
        [cone_x + 30,  cone_y + 40],
    ], dtype=np.int32)
    cv2.fillPoly(curr, [cone_pts], (0, 140, 255))   # orange body
    cv2.polylines(curr, [cone_pts], True, (0, 80, 180), 2)

    # White stripe on cone
    stripe_pts = np.array([
        [cone_x - 15, cone_y + 10],
        [cone_x + 15, cone_y + 10],
        [cone_x + 18, cone_y + 22],
        [cone_x - 18, cone_y + 22],
    ], dtype=np.int32)
    cv2.fillPoly(curr, [stripe_pts], (220, 220, 220))

    # Cone base (dark trapezoid)
    base_pts = np.array([
        [cone_x - 35, cone_y + 40],
        [cone_x + 35, cone_y + 40],
        [cone_x + 40, cone_y + 55],
        [cone_x - 40, cone_y + 55],
    ], dtype=np.int32)
    cv2.fillPoly(curr, [base_pts], (20, 20, 20))

    return curr


def make_ignore_mask(h=480, w=640) -> np.ndarray:
    """
    Binary mask: white = inspect, black = ignore.
    Ignores the left strip (e.g., UI overlay area).
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[:, :60] = 0   # Ignore left 60px column
    return mask


def encode_jpg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def encode_png(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ============================================================
# Test Runner
# ============================================================

results: list[dict] = []


def run_test(name: str, fn):
    try:
        data = fn()
        results.append({"name": name, "status": "PASS", "data": data})
        print(ok(f"{name}"))
        if data:
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    v_str = json.dumps(v, indent=6)[:400]
                else:
                    v_str = str(v)[:200]
                print(f"       {color(k, 90)}: {v_str}")
    except Exception as e:
        results.append({"name": name, "status": "FAIL", "error": str(e)})
        print(err(f"{name} — {e}"))
        traceback.print_exc()


# ============================================================
# Individual Tests
# ============================================================

REF_IMG   = make_reference_image()
CURR_IMG  = make_current_image_with_fod(REF_IMG)
MASK_IMG  = make_ignore_mask()

REF_BYTES  = encode_jpg(REF_IMG)
CURR_BYTES = encode_jpg(CURR_IMG)
MASK_BYTES = encode_png(MASK_IMG)


def test_health():
    r = requests.get(f"{API_BASE}/health")
    r.raise_for_status()
    d = r.json()
    assert d["status"] == "ok", f"Unexpected status: {d}"
    return {"status": d["status"], "version": d["version"], "engine": d["engine"]}


def test_get_config():
    r = requests.get(f"{API_BASE}/config")
    r.raise_for_status()
    d = r.json()
    assert "config" in d
    cfg = d["config"]
    return {
        "source": d["source"],
        "threshold_method": cfg["difference"]["threshold_method"],
        "enable_dl_validation": cfg["validation"]["enable_dl_validation"],
        "deep_feature_threshold": cfg["validation"]["deep_feature_threshold"],
    }


def test_update_config():
    payload = {
        "difference": {"weight_ssim": 0.40, "threshold_method": "OTSU"},
        "validation": {"deep_feature_threshold": 0.82},
    }
    r = requests.put(f"{API_BASE}/config", json=payload)
    r.raise_for_status()
    d = r.json()
    cfg = d["config"]
    assert cfg["difference"]["weight_ssim"] == 0.40
    return {
        "source": d["source"],
        "weight_ssim": cfg["difference"]["weight_ssim"],
        "deep_feature_threshold": cfg["validation"]["deep_feature_threshold"],
    }


def test_reset_config():
    r = requests.post(f"{API_BASE}/config/reset")
    r.raise_for_status()
    d = r.json()
    assert d["source"] == "default"
    return {"source": d["source"]}


_job_id = None


def test_submit_async_job():
    global _job_id
    files = {
        "reference_images": ("reference.jpg", REF_BYTES,  "image/jpeg"),
        "current_image":    ("current.jpg",   CURR_BYTES, "image/jpeg"),
        "ignore_mask":      ("mask.png",       MASK_BYTES, "image/png"),
    }
    data = {
        "enable_dl_validation": "true",
        "deep_feature_threshold": "0.82",
        "threshold_method": "OTSU",
    }
    r = requests.post(f"{API_BASE}/inspect", files=files, data=data)
    assert r.status_code == 202, f"Expected 202, got {r.status_code}: {r.text}"
    d = r.json()
    _job_id = d["job_id"]
    return {
        "job_id": _job_id,
        "status": d["status"],
        "poll_url": d["poll_url"],
    }


def test_poll_job_until_done():
    assert _job_id, "No job_id from previous test"
    for attempt in range(30):
        r = requests.get(f"{API_BASE}/inspect/{_job_id}")
        r.raise_for_status()
        d = r.json()
        status = d["status"]
        print(f"       [{attempt+1}] Polling... status={status}")
        if status == "DONE":
            res = d.get("result", {})
            return {
                "final_status": status,
                "fod_status": res.get("status"),
                "objects": res.get("objects"),
                "similarity_score": res.get("similarity_score"),
                "alignment_score": res.get("alignment_score"),
                "processing_time_ms": d.get("processing_time_ms"),
                "detections": res.get("detections", []),
            }
        elif status == "FAILED":
            raise RuntimeError(f"Job failed: {d.get('error')}")
        time.sleep(2)
    raise TimeoutError("Job did not complete within 60s")


def test_get_marked_image():
    assert _job_id, "No job_id"
    r = requests.get(f"{API_BASE}/inspect/{_job_id}/images/marked")
    assert r.status_code == 200, f"Got {r.status_code}"
    assert r.headers["content-type"].startswith("image/")
    size_kb = len(r.content) // 1024
    return {"content_type": r.headers["content-type"], "size_kb": f"{size_kb} KB"}


def test_get_diff_image():
    assert _job_id, "No job_id"
    r = requests.get(f"{API_BASE}/inspect/{_job_id}/images/diff")
    assert r.status_code == 200
    size_kb = len(r.content) // 1024
    return {"content_type": r.headers["content-type"], "size_kb": f"{size_kb} KB"}


def test_get_mask_image():
    assert _job_id, "No job_id"
    r = requests.get(f"{API_BASE}/inspect/{_job_id}/images/mask")
    assert r.status_code == 200
    size_kb = len(r.content) // 1024
    return {"content_type": r.headers["content-type"], "size_kb": f"{size_kb} KB"}


def test_list_jobs():
    r = requests.get(f"{API_BASE}/jobs")
    r.raise_for_status()
    d = r.json()
    return {"total": d["total"], "first_job_status": d["jobs"][0]["status"] if d["jobs"] else "N/A"}


def test_list_jobs_filter_done():
    r = requests.get(f"{API_BASE}/jobs?status=DONE&limit=5")
    r.raise_for_status()
    d = r.json()
    return {"total_done": d["total"], "returned": len(d["jobs"])}


def test_synchronous_detect():
    files = {
        "reference_images": ("reference.jpg", REF_BYTES,  "image/jpeg"),
        "current_image":    ("current.jpg",   CURR_BYTES, "image/jpeg"),
    }
    data = {"enable_dl_validation": "false"}
    r = requests.post(f"{API_BASE}/detect", files=files, data=data)
    r.raise_for_status()
    d = r.json()
    return {
        "status": d["status"],
        "objects": d["objects"],
        "similarity_score": d["similarity_score"],
        "processing_time_ms": d["processing_time_ms"],
    }


def test_legacy_detect_fod():
    files = {
        "reference_images": ("reference.jpg", REF_BYTES,  "image/jpeg"),
        "current_image":    ("current.jpg",   CURR_BYTES, "image/jpeg"),
    }
    r = requests.post(f"{BASE_URL}/detect-fod", files=files)
    r.raise_for_status()
    d = r.json()
    return {"status": d["status"], "objects": d["objects"]}


def test_delete_job():
    assert _job_id, "No job_id"
    r = requests.delete(f"{API_BASE}/inspect/{_job_id}")
    assert r.status_code == 200
    d = r.json()
    return {"deleted": d["deleted"], "job_id": d["job_id"][:8] + "..."}


def test_purge_jobs():
    r = requests.delete(f"{API_BASE}/jobs?max_age_seconds=60")
    r.raise_for_status()
    d = r.json()
    return {"deleted": d["deleted"], "message": d["message"]}


def test_not_found_job():
    r = requests.get(f"{API_BASE}/inspect/nonexistent-job-id")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    return {"http_status": r.status_code, "detail": r.json()["detail"][:60]}


# ============================================================
# Main
# ============================================================

TESTS = [
    ("Health Check — GET /api/v1/health",                   test_health),
    ("Get Config — GET /api/v1/config",                     test_get_config),
    ("Update Config — PUT /api/v1/config",                  test_update_config),
    ("Reset Config — POST /api/v1/config/reset",            test_reset_config),
    ("Submit Async Job — POST /api/v1/inspect",             test_submit_async_job),
    ("Poll Job Until DONE — GET /api/v1/inspect/{id}",      test_poll_job_until_done),
    ("Get Marked Image — GET /api/v1/inspect/{id}/images/marked", test_get_marked_image),
    ("Get Diff Heatmap — GET /api/v1/inspect/{id}/images/diff",   test_get_diff_image),
    ("Get Mask Image — GET /api/v1/inspect/{id}/images/mask",     test_get_mask_image),
    ("List All Jobs — GET /api/v1/jobs",                    test_list_jobs),
    ("List DONE Jobs — GET /api/v1/jobs?status=DONE",       test_list_jobs_filter_done),
    ("Sync Detect — POST /api/v1/detect",                   test_synchronous_detect),
    ("Legacy Detect — POST /detect-fod",                    test_legacy_detect_fod),
    ("Delete Job — DELETE /api/v1/inspect/{id}",            test_delete_job),
    ("Purge Old Jobs — DELETE /api/v1/jobs",                test_purge_jobs),
    ("404 Not Found — GET /api/v1/inspect/bad-id",          test_not_found_job),
]


def main():
    print(head("AERO-EYE FOD — Full API Test Suite"))
    print(info(f"Target: {BASE_URL}"))

    if not check_server():
        print(err("Start the server first: uvicorn app:app --reload"))
        sys.exit(1)

    print(info("Generating synthetic sample images..."))
    print(info(f"  Reference: {REF_BYTES.__len__() // 1024} KB clean industrial scan"))
    print(info(f"  Current:   {CURR_BYTES.__len__() // 1024} KB scan + orange safety cone (FOD)"))
    print(info(f"  Mask:      {MASK_BYTES.__len__() // 1024} KB ignore-zone mask\n"))

    for name, fn in TESTS:
        run_test(name, fn)

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(head(f"Results: {passed}/{len(results)} passed  |  {failed} failed"))

    if failed:
        print(err("Failed tests:"))
        for r in results:
            if r["status"] == "FAIL":
                print(f"   • {r['name']}: {r['error']}")
    else:
        print(ok("All tests passed! API is fully operational."))


if __name__ == "__main__":
    main()
