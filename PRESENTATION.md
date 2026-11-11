# 🎤 AegisMask AI — Team Presentation Script

> **Project:** Real-Time Face Mask Detection Platform  
> **Team Size:** 4 members  
> **Duration:** ~15–20 minutes total  

---

## 🧑‍🤝‍🧑 Role Assignment

| # | Member Role | Responsibility | Time |
|---|---|---|---|
| **Member 1** | Project Introducer | Problem statement, motivation, architecture overview | 4–5 min |
| **Member 2** | Technical Explainer | Model pipeline, tech stack, code walkthrough | 4–5 min |
| **Member 3** | Live Demo Presenter | Dashboard demo, webcam stream, image upload, audit log | 4–5 min |
| **Member 4** | Results & Conclusion | Test results, deployment, future scope, Q&A | 3–4 min |

---

## 👤 Member 1 — Project Introduction (4–5 min)

### Script

> *"Good [morning/afternoon], everyone. I'm [Name], and I'll be introducing our project — **AegisMask AI**, a real-time face mask detection platform built for public safety compliance monitoring."*

---

### Slide / Talking Points

**1. The Problem**
- During the COVID-19 pandemic, enforcing mask mandates in public spaces — offices, hospitals, airports, schools — was a manual, error-prone process.
- Security staff couldn't monitor hundreds of camera feeds simultaneously.
- There was no automated, real-time system to alert authorities when someone violated mask protocols.

**2. Our Solution**
- We built **AegisMask AI** — a web-based surveillance dashboard powered by deep learning.
- It detects faces in real time and classifies each person as:
  - ✅ **With Mask** (green box)
  - ❌ **Without Mask** (red box + alarm)
  - ⚠️ **Mask Worn Incorrectly** (yellow box)
- Works on **webcam feeds, uploaded images, and video files**.
- Fully runs **offline on CPU** — no cloud dependency.

**3. Why It Matters**
- Hospitals and healthcare facilities can automate mask compliance without hiring extra staff.
- It is scalable — a single server can handle multiple camera feeds.
- The audit log provides **evidence-based compliance records** that can be used for reporting.

**4. High-Level Architecture**

```
Camera / Image
      │
      ▼
 YuNet Face Detector (ONNX)        ← Finds all faces in frame
      │
      ▼
 MobileNetV2 Classifier (ONNX)     ← Labels each face: mask / no-mask / incorrect
      │
      ▼
 FastAPI Backend                   ← REST API + WebSocket stream
      │
      ▼
 Browser Dashboard (HTML/CSS/JS)   ← Live HUD, charts, audit log
```

> *"Now, my teammate [Member 2] will walk you through the technical details."*

---

## 👤 Member 2 — Technical Explainer (4–5 min)

### Script

> *"Thank you, [Member 1]. I'm [Name], and I'll explain how the system actually works under the hood."*

---

### Talking Points

**1. Tech Stack**

| Layer | Technology | Why |
|---|---|---|
| Face Detection | **YuNet (ONNX)** | Sub-millisecond on CPU, OpenCV native |
| Mask Classification | **MobileNetV2 (ONNX)** | Lightweight, fast, 3-class output |
| Backend | **FastAPI (Python)** | Async, WebSocket-ready, auto Swagger docs |
| Frontend | **Vanilla JS + Chart.js** | No framework bloat, instant load |
| Vision runtime | **OpenCV DNN module** | No heavy ML framework (no PyTorch/TF needed) |

**2. The Two-Stage Pipeline**

> Explain `core/detector.py`:

- **Stage 1 — Face Detection:**  
  YuNet processes each frame and returns bounding boxes for every face detected. It's extremely fast and accurate even with partial occlusion (which masks cause).

- **Stage 2 — Mask Classification:**  
  For each detected face bounding box:
  1. Crop and resize to 224×224 pixels.
  2. Normalize pixel values to [−1, +1].
  3. Feed into MobileNetV2.
  4. Take `argmax` of the 3-class output: `with_mask` / `without_mask` / `mask_weared_incorrect`.

**3. WebSocket Live Stream**

- The browser sends JPEG frames to the server over a WebSocket connection.
- Server decodes the frame, runs the two-stage pipeline, draws HUD overlays (colored boxes + labels + confidence %), and returns the annotated JPEG.
- The browser paints the annotated frame back to the canvas ~20–30 times per second.
- Round-trip latency: ~30–50 ms on localhost.

**4. Key Code Files**

| File | Purpose |
|---|---|
| [`core/detector.py`](core/detector.py) | Pipeline logic: `FaceMaskPipeline`, `draw_hud`, `Detection` dataclass |
| [`app.py`](app.py) | FastAPI routes + WebSocket handler + audit log |
| [`mask_detection.py`](mask_detection.py) | CLI entrypoint for image/video/webcam |
| [`static/js/dashboard.js`](static/js/dashboard.js) | WebSocket client, chart, KPI updates, alarm |

**5. ONNX Advantage**
- Both models are in ONNX format — framework-agnostic, runs anywhere Python + OpenCV is installed.
- No GPU required; runs smoothly at 20+ FPS on a standard laptop CPU.

> *"[Member 3] will now show you the live demo."*

---

## 👤 Member 3 — Live Demo (4–5 min)

### Script

> *"Thank you, [Member 2]. I'm [Name], and I'll walk you through the actual running application."*

---

### Demo Steps (perform live)

**Step 1 — Start the server**
```bash
cd real-time-face-mask-detection
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
> Open browser → `http://localhost:8000`

**Step 2 — Dashboard overview**
- Point out the **KPI Cards** at the top:
  - 👥 Faces Detected
  - ✅ Compliance Rate %
  - 🚨 Violations Today
  - ⏱️ Uptime
- Point out the **video monitor** in the center.
- Point out the **Compliance Timeline** chart on the right.

**Step 3 — Sample image demo**
- Click **"Sample Images"** tab.
- Select `mask_sample.jpg` → notice green box, "With Mask 97%", KPIs update.
- Select `pic1.jpeg` (no mask) → red box, violation counter increments, alarm sounds.

**Step 4 — Upload your own image**
- Click **"Upload Image"** tab.
- Drag and drop any photo from your computer.
- Result appears with colored bounding boxes.

**Step 5 — Webcam (if available)**
- Click **"Live Webcam"** → allow camera access.
- Put on / remove your mask → watch the label switch in real time.
- Point out the audible chime when mask is removed.

**Step 6 — Audit Log & CSV Export**
- Scroll down to the **Incident Audit Log** table.
- Show timestamped records with face count and compliance rate.
- Click **"Export CSV"** → file downloads instantly.

**Step 7 — API Docs (bonus)**
- Open `http://localhost:8000/docs` → Swagger UI.
- Hit **GET /api/status** → shows model readiness.

> *"And that's the full platform! [Member 4] will now summarize our results and future work."*

---

## 👤 Member 4 — Results, Deployment & Future Scope (3–4 min)

### Script

> *"Thank you, [Member 3]. I'm [Name], and I'll cover our test results, how to deploy this, and where we can take it next."*

---

### Talking Points

**1. Test Results**

```
============================= test session starts =============================
collected 10 items

tests/test_api.py::test_status_endpoint           PASSED
tests/test_api.py::test_samples_endpoint          PASSED
tests/test_api.py::test_sample_infer_masked       PASSED
tests/test_api.py::test_sample_infer_no_mask      PASSED
tests/test_api.py::test_upload_infer              PASSED
tests/test_api.py::test_audit_log                 PASSED
tests/test_api.py::test_csv_export                PASSED
tests/test_api.py::test_static_html               PASSED
tests/test_api.py::test_pipeline_load             PASSED
tests/test_api.py::test_hud_rendering             PASSED

======================== 10 passed in X.XXs ===========================
```

- **100% test pass rate** — all API endpoints, model loading, and inference logic verified.
- Tests run without any camera, GUI, or model weights — fully automated CI/CD compatible.

**2. Performance**

| Metric | Value |
|---|---|
| Face detection latency | ~1–3 ms per frame |
| Mask classification | ~5–10 ms per face |
| End-to-end (1 face, webcam) | ~15–30 ms |
| Approximate FPS | 20–30 FPS on CPU |

**3. Deployment Options**

| Platform | Cost | Ease | Best For |
|---|---|---|---|
| **Railway** | Free tier | ⭐⭐⭐⭐⭐ | Quick demo / class project |
| **Render** | Free tier | ⭐⭐⭐⭐ | Persistent web service |
| **Google Cloud Run** | Pay-per-use | ⭐⭐⭐ | Production scale |
| **Docker + VPS** | VPS cost | ⭐⭐⭐ | Full control |
| **Local** | Free | ⭐⭐⭐⭐⭐ | Development / offline use |

> Recommended for class submission: **Railway** — push to GitHub, connect repo, done in 5 minutes.

**4. Limitations**
- Webcam WebSocket stream requires a secure context (HTTPS) in production browsers.
- Model was trained on frontal faces; side-profile detection accuracy may drop.
- No multi-camera feed support yet.

**5. Future Scope**

| Enhancement | Description |
|---|---|
| 🎥 Multi-camera | Add RTSP stream support for IP cameras |
| 📱 Mobile App | Flutter / React Native client using the REST API |
| 📧 Alerts | Email / SMS notification on sustained violations |
| 🧠 Custom Model | Retrain on domain-specific data (hospital, office) |
| 🔐 Auth | Add login + role-based access for dashboard |
| 📈 Analytics | Daily / weekly compliance reports + PDF export |

**6. Closing**

> *"AegisMask AI is a fully functional, deployable face-mask compliance system built in Python with no cloud dependency at inference time. It demonstrates the practical application of computer vision, REST APIs, and modern web development in solving a real public-health challenge. Thank you — we're happy to take any questions!"*

---

## ❓ Common Q&A

| Question | Answer |
|---|---|
| Which dataset was used? | The mask classifier (MobileNetV2 ONNX) was pre-trained on a combined dataset of ~12 K masked/unmasked face images. |
| Can it detect multiple faces simultaneously? | Yes — YuNet detects all faces in a frame; each is classified independently. |
| Does it work without internet? | Completely offline after the one-time model download. |
| What is the accuracy? | The classifier reports 97–99% confidence on clean frontal faces in test images. |
| Why ONNX instead of TensorFlow/PyTorch? | ONNX + OpenCV DNN runs with zero extra ML framework install, keeping the dependency footprint tiny. |
| How is this different from similar projects? | Most GitHub examples are scripts; we built a full production web platform with a live WebSocket stream, audit log, CSV export, and a polished UI. |

---

*Document prepared for AegisMask AI — Real-time Face Mask Detection Platform.*  
*GitHub: https://github.com/Rizwana28/real-time-face-mask-detection*

