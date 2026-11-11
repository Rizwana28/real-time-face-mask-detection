"""FastAPI application for AegisMask AI Real-Time Face Mask Detection Platform.

Provides REST and WebSocket endpoints for:
- Live browser webcam streaming and real-time inference
- Static image upload & batch analysis
- Video file processing
- Pre-loaded interactive demonstration samples
- Session audit logging & CSV compliance export
- Interactive Single-Page Dashboard
"""

from __future__ import annotations

import base64
import csv
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.detector import Detection, FaceMaskPipeline, draw_hud

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize pipeline — graceful: server starts even if models are absent
try:
    pipeline = FaceMaskPipeline(models_dir=MODELS_DIR)
except Exception as _exc:
    print(f"[AegisMask] WARNING: Pipeline not ready — {_exc}")
    pipeline = None  # type: ignore[assignment]

app = FastAPI(
    title="AegisMask AI - Real-Time Face Mask Detection Platform",
    description="Enterprise-grade face mask detection, compliance analytics, and safety monitoring",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session audit logs
audit_logs: list[dict[str, Any]] = []


def _record_audit_events(detections: list[Detection], source: str = "webcam") -> None:
    """Record non-compliant detections in the session audit log."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for d in detections:
        if d.label in ("without_mask", "mask_weared_incorrect"):
            audit_logs.append({
                "id": len(audit_logs) + 1,
                "timestamp": now_str,
                "source": source,
                "violation_type": d.label,
                "confidence": round(d.confidence * 100, 1),
                "box": d.box,
                "status": "NON_COMPLIANT" if d.label == "without_mask" else "IMPROPER_WEAR",
            })
            # Keep latest 500 records
            if len(audit_logs) > 500:
                audit_logs.pop(0)


def _require_pipeline():
    """Raise 503 when ONNX models are not yet loaded."""
    if pipeline is None or not pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Upload face_detection_yunet_2023mar.onnx and mask_detector.onnx to the models/ folder and restart.",
        )


def _compute_summary(detections: list[Detection]) -> dict[str, Any]:
    total = len(detections)
    with_mask = sum(1 for d in detections if d.label == "with_mask")
    without_mask = sum(1 for d in detections if d.label == "without_mask")
    incorrect = sum(1 for d in detections if d.label == "mask_weared_incorrect")

    compliance_rate = round((with_mask / total * 100.0), 1) if total > 0 else 100.0

    return {
        "total_faces": total,
        "with_mask": with_mask,
        "without_mask": without_mask,
        "mask_weared_incorrect": incorrect,
        "compliance_score": compliance_rate,
        "compliance_status": (
            "OPTIMAL" if compliance_rate >= 90 else
            "MODERATE" if compliance_rate >= 60 else
            "CRITICAL"
        ),
    }


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Return model readiness and server telemetry."""
    return {
        "status": "online",
        "service": "AegisMask AI Intelligence Engine",
        "model_ready": pipeline is not None and pipeline.is_ready,
        "detector": "OpenCV YuNet (ONNX)",
        "classifier": "MobileNetV2 Face Mask (ONNX)",
        "supported_classes": FaceMaskPipeline.LABELS if pipeline else [],
        "audit_events_count": len(audit_logs),
    }


@app.get("/api/samples")
def list_samples() -> list[dict[str, Any]]:
    """List available sample images for interactive preview."""
    samples = [
        {
            "id": "pic1.jpeg",
            "title": "Multiple Protected Individuals",
            "description": "Two individuals wearing surgical face masks properly",
            "expected": "with_mask",
            "url": "/samples/pic1.jpeg",
        },
        {
            "id": "mask_sample.jpg",
            "title": "Close-up Masked Portrait",
            "description": "Single individual wearing a protective medical mask",
            "expected": "with_mask",
            "url": "/samples/mask_sample.jpg",
        },
        {
            "id": "lena.jpg",
            "title": "Unmasked Portrait (Violation)",
            "description": "Standard portrait benchmark without protective mask",
            "expected": "without_mask",
            "url": "/samples/lena.jpg",
        },
    ]
    return [s for s in samples if (SAMPLES_DIR / s["id"]).is_file()]


@app.get("/samples/{filename}")
def get_sample_file(filename: str):
    file_path = SAMPLES_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(file_path)


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    nms: float = Form(0.35),
    draw_landmarks: bool = Form(True),
):
    """Run face mask detection on an uploaded image file."""
    _require_pipeline()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file or format")

    t0 = time.monotonic()
    detections = pipeline.detect(image, conf_threshold=confidence, nms_threshold=nms)
    elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)

    _record_audit_events(detections, source=file.filename or "upload")
    summary = _compute_summary(detections)

    rendered = draw_hud(
        image,
        detections,
        draw_landmarks=draw_landmarks,
        draw_hud_banner=True,
    )

    _, encoded_jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_image = "data:image/jpeg;base64," + base64.b64encode(encoded_jpg).decode("utf-8")

    return {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "summary": summary,
        "detections": [d.to_dict() for d in detections],
        "annotated_image": b64_image,
    }


@app.post("/api/detect/sample/{sample_id}")
def detect_sample(
    sample_id: str,
    confidence: float = 0.5,
    nms: float = 0.35,
    draw_landmarks: bool = True,
):
    """Run detection directly on a pre-packaged sample image."""
    _require_pipeline()
    file_path = SAMPLES_DIR / sample_id
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")

    image = cv2.imread(str(file_path))
    if image is None:
        raise HTTPException(status_code=500, detail="Could not read sample image")

    t0 = time.monotonic()
    detections = pipeline.detect(image, conf_threshold=confidence, nms_threshold=nms)
    elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)

    _record_audit_events(detections, source=sample_id)
    summary = _compute_summary(detections)

    rendered = draw_hud(
        image,
        detections,
        draw_landmarks=draw_landmarks,
        draw_hud_banner=True,
    )

    _, encoded_jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_image = "data:image/jpeg;base64," + base64.b64encode(encoded_jpg).decode("utf-8")

    return {
        "success": True,
        "sample_id": sample_id,
        "processing_time_ms": elapsed_ms,
        "summary": summary,
        "detections": [d.to_dict() for d in detections],
        "annotated_image": b64_image,
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """High-speed WebSocket endpoint for live client webcam inference."""
    await websocket.accept()
    if pipeline is None or not pipeline.is_ready:
        await websocket.send_json({"error": "Models not loaded. Please add ONNX models to models/ and restart."})
        await websocket.close()
        return
    try:
        while True:
            # Client sends base64 JPEG string or JSON with settings & image
            data = await websocket.receive_json()
            image_data = data.get("image")
            if not image_data:
                continue

            # Strip base64 header if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Failed to decode frame"})
                continue

            conf = float(data.get("confidence", 0.5))
            nms = float(data.get("nms", 0.35))
            draw_pts = bool(data.get("draw_landmarks", True))
            client_fps = data.get("fps")

            t0 = time.monotonic()
            detections = pipeline.detect(frame, conf_threshold=conf, nms_threshold=nms)
            elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)

            _record_audit_events(detections, source="live_webcam")
            summary = _compute_summary(detections)

            fps_val = float(client_fps) if client_fps else (1000.0 / max(elapsed_ms, 1.0))
            rendered = draw_hud(
                frame,
                detections,
                fps=fps_val,
                draw_landmarks=draw_pts,
                draw_hud_banner=True,
            )

            _, encoded_jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_out = "data:image/jpeg;base64," + base64.b64encode(encoded_jpg).decode("utf-8")

            await websocket.send_json({
                "success": True,
                "latency_ms": elapsed_ms,
                "summary": summary,
                "detections": [d.to_dict() for d in detections],
                "frame": b64_out,
            })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


@app.get("/api/audit-logs")
def get_audit_logs() -> list[dict[str, Any]]:
    """Return all recorded compliance violations and incident telemetry."""
    return list(reversed(audit_logs))


@app.post("/api/audit-logs/clear")
def clear_audit_logs() -> dict[str, str]:
    audit_logs.clear()
    return {"message": "Audit log cleared successfully"}


@app.get("/api/audit-logs/export")
def export_audit_logs():
    """Export compliance violation audit logs as a downloadable CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Source", "Violation Type", "Confidence (%)", "Status", "Bounding Box"])

    for item in audit_logs:
        writer.writerow([
            item["id"],
            item["timestamp"],
            item["source"],
            item["violation_type"],
            item["confidence"],
            item["status"],
            str(item["box"]),
        ])

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=compliance_audit_{int(time.time())}.csv"
        },
    )


# Mount static assets
if (STATIC_DIR / "css").is_dir() or (STATIC_DIR / "js").is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return HTMLResponse("<h1>AegisMask AI Server is running</h1>")

