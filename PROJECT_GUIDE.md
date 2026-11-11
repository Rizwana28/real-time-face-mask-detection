# 🛡️ AegisMask AI — Final Project Guide & 4-Member Presentation Script

**Project Title:** Real-Time Face Mask Detection & Compliance Monitoring System  
**Repository:** [github.com/Rizwana28/real-time-face-mask-detection](https://github.com/Rizwana28/real-time-face-mask-detection)  
**Live URL:** [real-time-face-mask-detection-yl6e.onrender.com](https://real-time-face-mask-detection-yl6e.onrender.com)  

---

## 📌 1. Simple Summary of What We Built

We transformed a basic script into a **complete, production-grade AI web application**:
1. **AI Vision Engine:** Uses a 2-stage lightweight pipeline:
   - **YuNet (ONNX):** Ultra-fast face detector that finds any face in under 5 milliseconds.
   - **MobileNetV2 (ONNX):** Classifies each detected face into **With Mask**, **Without Mask**, or **Incorrect Mask**.
2. **FastAPI Web Server:** High-performance async backend that handles image upload inference, preloaded demo samples, and **real-time WebSocket video streaming** (over secure `wss://`).
3. **Professional Black & White Dashboard:** Engineered with an enterprise minimalist UI (`#0a0a0a` dark mode, typography in Inter & JetBrains Mono, strictly no flashy AI gradients).
4. **Live Telemetry & Auditing:** Real-time compliance score (%), incident audit log, downloadable CSV export, and audio warning alerts.
5. **Automated Testing & Cloud Deployment:** 10 out of 10 automated unit tests passing, containerized with Docker, and deployed live on Render.

---

## 👥 2. 4-Member Team Presentation Division

---

### 👤 Student 1: Introduction, Problem Statement & Motivation
* **Role:** Sets the stage, explains the real-world importance, and gives the big picture.
* **Duration:** ~3 to 4 minutes.

#### 🎙️ Speech Script:
> "Good morning/afternoon respected professors and evaluators. Today, our team is presenting **AegisMask AI** — an automated, real-time face mask detection and safety compliance platform.
>
> **The Problem:**  
> In healthcare facilities, cleanrooms, airports, and public transport hubs, enforcing hygiene protocols like wearing face masks is critical. Relying on manual human security guards is inconsistent, expensive, and puts guards at direct risk of exposure.
>
> **Our Solution:**  
> We engineered an end-to-end intelligent platform that connects to any camera or video feed. It automatically recognizes faces, checks whether masks are worn correctly, tracks violation statistics in real-time, and saves verifiable incident logs for safety audits.
>
> **Core Highlights:**  
> - Runs completely offline without expensive cloud AI APIs.  
> - Designed with a high-performance, distraction-free corporate dashboard.  
> - Delivers real-time inference at 25+ frames per second on standard everyday hardware.
>
> Now, my teammate [Student 2's Name] will explain the AI architecture and how the machine learning pipeline works."

---

### 👤 Student 2: System Architecture & AI Pipeline
* **Role:** Explains how the machine learning and backend work under the hood.
* **Duration:** ~4 minutes.

#### 🎙️ Speech Script:
> "Thank you, [Student 1's Name]. I will walk you through the technical architecture and AI pipeline.
>
> **The Two-Stage Computer Vision Pipeline:**  
> Instead of using heavy, monolithic neural networks that struggle on CPUs, we implemented a decoupled two-stage pipeline:
> 
> 1. **Stage 1 — Face Detection (OpenCV YuNet ONNX):**  
>    YuNet scans the input frame and predicts bounding boxes and facial landmarks. It is specifically optimized to detect faces even when partially occluded by masks or hats.
> 
> 2. **Stage 2 — Mask Classification (MobileNetV2 ONNX):**  
>    Each detected face region is cropped, normalized to the range $[-1, +1]$, and fed into a deep MobileNetV2 classifier. It calculates class probabilities across three categories:  
>    - `with_mask` (Compliant)  
>    - `without_mask` (Violation)  
>    - `mask_weared_incorrect` (Improper wear)  
>
> **Backend & Streaming Technology:**  
> - Built with **FastAPI** in Python for async request processing.  
> - For live video, standard HTTP requests create too much latency. We used bidirectional **WebSockets** over `wss://`. The client captures frames, transmits compressed JPEG payloads, and the server returns the annotated frame with sub-30 millisecond latency.
>
> Next, [Student 3's Name] will demonstrate the live system in action."

---

### 👤 Student 3: Live Application Demonstration
* **Role:** Hands-on product demonstration (screen share / live browser).
* **Duration:** ~4 to 5 minutes.

#### 🎙️ Demo Checklist & Script:
> "Thank you, [Student 2's Name]. Let's look at the live running platform.
>
> *(Open the browser at `https://real-time-face-mask-detection-yl6e.onrender.com`)*
>
> **1. Dashboard Overview:**  
> Notice our clean, monochromatic interface designed like a security operations room. At the top and left, we have our key metrics:  
> - Compliance Score percentage  
> - Total Faces, Safe count, and Violation alerts.  
>
> **2. Testing Pre-loaded Demo Samples:**  
> *(Click on 'Demo Samples' tab)*  
> - When I select `mask_sample.jpg`, the system immediately detects the face, tags it with a white/gray bounding box labeled 'With Mask (99%)', and updates the compliance score to 100%.  
> - When I switch to an unmasked photo (`lena.jpg`), the system marks the violation, increments the alert counter, and flags the incident.
>
> **3. Testing Image Upload:**  
> *(Click 'Upload Image' tab, drag and drop an image)*  
> - Any photo uploaded from disk is processed instantly by our API endpoint `/api/detect/image`.
>
> **4. Live Webcam Stream:**  
> *(Click 'Live Webcam' tab and allow camera)*  
> - Through WebSocket streaming, we observe real-time detection on my camera with low latency.  
> - Notice the audio alarm toggle and the landmark tracking.
>
> **5. Audit Logging & Compliance Export:**  
> - Every violation is captured in our Incident Log table below.  
> - When management needs a compliance report, clicking **'Export CSV'** downloads the full audit trail with timestamps and coordinates.
>
> Now, [Student 4's Name] will conclude with testing, deployment, and future enhancements."

---

### 👤 Student 4: Testing, Cloud Deployment & Conclusion
* **Role:** Discusses testing reliability, production deployment, and project wrap-up.
* **Duration:** ~3 minutes.

#### 🎙️ Speech Script:
> "Thank you, [Student 3's Name]. I will cover our validation results, cloud deployment, and future scope.
>
> **Quality Assurance & Testing:**  
> Reliability was a primary requirement for this project:  
> - We created an automated test suite with **pytest**.  
> - All **10 integration and unit tests pass cleanly**, verifying API health, image detection accuracy, CSV exports, and HUD rendering logic without needing physical cameras.
>
> **Cloud Deployment:**  
> - Containerized with **Docker** for complete environment consistency.  
> - Deployed live on **Render Cloud** using automatic continuous deployment linked to our GitHub repository. Both ONNX models are embedded directly, ensuring zero downtime and no missing dependency errors.
>
> **Future Scope:**  
> 1. Supporting RTSP streams from multi-camera CCTV networks.  
> 2. Automated email or SMS notifications for facility managers when violation spikes occur.  
> 3. Edge-device deployment to hardware like Raspberry Pi or NVIDIA Jetson for embedded gate controls.
>
> In conclusion, AegisMask AI is a robust, responsive, and production-ready safety monitoring system. We now invite any questions from the panel. Thank you!"

---

## 💡 3. Quick Viva / Examiner Questions & Answers

| # | Question Evaluators Might Ask | Simple & Accurate Answer |
|---|---|---|
| **1** | **Why did you use ONNX models instead of PyTorch or TensorFlow directly?** | ONNX models run directly inside OpenCV's native DNN module without needing heavy frameworks (PyTorch is ~2 GB, ONNX runtime/OpenCV is lightweight). It provides faster CPU inference with minimal RAM usage. |
| **2** | **Why a 2-stage pipeline instead of single-stage YOLO?** | YuNet specializes in detecting occluded faces with high precision, while MobileNetV2 specializes in fine-grained mask classification. This modular approach allows updating either model independently and runs efficiently on CPU. |
| **3** | **How does the live webcam avoid lag?** | Frames are transmitted over a persistent WebSocket (`wss://`) as compressed JPEG data instead of polling HTTP requests. Server inference takes ~15–25 ms, achieving smooth real-time video. |
| **4** | **Where are the models hosted?** | Both models are committed directly inside the `models/` directory in the repository, so the Docker container on Render builds and runs self-contained without downloading external files at runtime. |
| **5** | **Is user data safe?** | Yes. All image processing happens in volatile memory; frames are never stored to disk unless explicitly designated for security audit logs. |
