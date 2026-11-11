"""Integration tests for AegisMask AI FastAPI endpoints and pipeline."""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from core.detector import FaceMaskPipeline, draw_hud

client = TestClient(app)


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_ready"] is True
    assert "YuNet" in data["detector"]


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 2
    ids = [s["id"] for s in samples]
    assert "lena.jpg" in ids
    assert "mask_sample.jpg" in ids


def test_detect_sample_unmasked():
    response = client.post("/api/detect/sample/lena.jpg")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["detections"]) >= 1
    assert data["detections"][0]["label"] == "without_mask"
    assert data["summary"]["without_mask"] >= 1
    assert "data:image/jpeg;base64," in data["annotated_image"]


def test_detect_sample_masked():
    response = client.post("/api/detect/sample/mask_sample.jpg")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["detections"]) >= 1
    assert data["detections"][0]["label"] == "with_mask"
    assert data["summary"]["with_mask"] >= 1


def test_detect_image_upload():
    # Create an in-memory test image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    files = {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}
    response = client.post("/api/detect/image", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["summary"]["total_faces"] == 0


def test_audit_logs_and_export():
    # Audit logs should have entries from previous sample detection tests
    response = client.get("/api/audit-logs")
    assert response.status_code == 200
    logs = response.json()
    assert isinstance(logs, list)

    # Test export
    export_resp = client.get("/api/audit-logs/export")
    assert export_resp.status_code == 200
    assert "text/csv" in export_resp.headers["content-type"]
    assert "ID,Timestamp,Source" in export_resp.text


def test_dashboard_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "AegisMask" in response.text or "AEGIS-MASK AI" in response.text

