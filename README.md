<div align="center">

# 🛡️ AegisMask AI

**Real-time face-mask detection platform powered by YuNet + MobileNetV2 ONNX**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-brightgreen.svg)](https://opencv.org/)

https://real-time-face-mask-detection-yl6e.onrender.com/


A production-grade web dashboard for real-time face-mask compliance monitoring.
Runs fully offline — no cloud calls at inference time.

</div>

---

## ✨ Features

| Capability | Detail |
|---|---|
| 🎯 **Face Detection** | YuNet (ONNX) — sub-millisecond on CPU |
| 😷 **Mask Classification** | MobileNetV2 (ONNX) — 3 classes: `with_mask`, `without_mask`, `mask_weared_incorrect` |
| 📷 **Live Webcam Stream** | WebSocket-based real-time HUD overlay in the browser |
| 🖼️ **Image Upload** | Drag-and-drop single-image inference |
| 📊 **Compliance Dashboard** | KPI cards, live timeline chart, incident audit log |
| 🔔 **Audible Alarm** | Web Audio API chime on mask violations |
| 📥 **CSV Export** | Download full audit log as CSV |
| 🧪 **Full Test Suite** | 10 pytest tests — no camera, GUI, or model weights needed |

---

## 🗂️ Project Layout

```
real-time-face-mask-detection/
├── app.py                    # FastAPI backend (REST + WebSocket)
├── mask_detection.py         # CLI tool (image / video / webcam)
├── core/
│   ├── __init__.py
│   └── detector.py           # Pipeline: YuNet → MobileNetV2, draw_hud, YOLO fallback
├── static/
│   ├── index.html            # Dashboard UI
│   ├── css/dashboard.css
│   └── js/dashboard.js
├── models/                   # ONNX model weights (not committed — see below)
│   ├── face_detection_yunet_2023mar.onnx
│   └── mask_detector.onnx
├── samples/                  # Test images
├── tests/
│   └── test_api.py
├── labels.example.txt
├── requirements.txt
└── pyproject.toml
```

> **Model weights are NOT committed** (large binary files). Download them once — see [Model Setup](#-model-setup) below.

---

## ⚡ Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/Rizwana28/real-time-face-mask-detection.git
cd real-time-face-mask-detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2 — Model Setup

Download the two ONNX models and place them in a `models/` folder:

| File | Source |
|---|---|
| `face_detection_yunet_2023mar.onnx` | [opencv_zoo – YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `mask_detector.onnx` | [daodung/face-mask-detection-onnx](https://github.com/daodung/face-mask-detection-onnx) |

```bash
mkdir models
# Then copy / move the downloaded .onnx files into models/
```

### 3 — Run the web dashboard

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

### 4 — Run the CLI tool

```bash
# Webcam (live)
python mask_detection.py

# Image file
python mask_detection.py --source photo.jpg --output output/result.jpg

# Video file
python mask_detection.py --source input.mp4 --output output/result.mp4

# Legacy YOLO weights (optional)
python mask_detection.py \
  --weights models/mask-yolo.weights \
  --config  models/mask-yolo.cfg \
  --labels  labels.example.txt
```

Press **Q** to exit live preview.

### 5 — Run tests

```bash
python -m pytest -v
```

All 10 tests pass without a camera, model weights, or display.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Server health + model readiness |
| `GET` | `/api/samples` | List available sample images |
| `POST` | `/api/infer/sample/{name}` | Run inference on a sample |
| `POST` | `/api/infer/upload` | Upload & infer an arbitrary image |
| `GET` | `/api/audit` | Retrieve audit-log entries |
| `GET` | `/api/audit/csv` | Download audit log as CSV |
| `WS` | `/ws/stream` | WebSocket live webcam stream |

Full interactive docs: **http://localhost:8000/docs**

---

## 🚀 Deployment

### Option A — Railway (recommended, free tier)

1. Push this repo to GitHub (already done ✅).
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**.
3. Select `real-time-face-mask-detection`.
4. Set the **Start Command**:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
5. Add a **Volume** (or object storage) and mount models there, or bundle small ONNX files.
6. Click **Deploy** — Railway gives you a public HTTPS URL instantly.

### Option B — Render (free tier)

1. [render.com](https://render.com) → **New Web Service → Connect GitHub repo**.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Upload models via Render **Disk** (persistent storage).

### Option C — Google Cloud Run (serverless)

```bash
gcloud run deploy aegismask \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

### Option D — Docker (self-host / VPS)

```dockerfile
# Dockerfile (add to project root)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t aegismask .
docker run -p 8000:8000 -v $(pwd)/models:/app/models aegismask
```

---

## 🎓 Team Presentation Guide

See [PRESENTATION.md](PRESENTATION.md) for the full 4-member presentation script.

---

## 📄 License

MIT © 2024 Rizwana28 / AkashMs24
