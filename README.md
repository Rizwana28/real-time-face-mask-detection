<div align="center">

# 🛡️ AegisMask AI 

**Real-time face-mask detection platform powered by YuNet + MobileNetV2 ONNX**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-brightgreen.svg)](https://opencv.org/)




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

PLEASE DEPLOY IT IN RENDER OR VERCEL OR RAILWAY

---

## 📄 License

MIT © 2024 Rizwana28 / AkashMs24
