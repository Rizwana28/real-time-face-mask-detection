/**
 * AegisMask AI - Frontend Intelligence & Telemetry Controller
 * Handles:
 * - High-speed WebSocket live webcam streaming
 * - Instant sample image evaluation
 * - Drag-and-drop file inference
 * - Rolling compliance time-series canvas chart
 * - Web Audio API synthesized security alarm
 * - Session audit log management and CSV export
 */

class AegisMaskDashboard {
  constructor() {
    this.mode = "webcam"; // "webcam" | "sample" | "upload"
    this.ws = null;
    this.stream = null;
    this.videoElement = document.createElement("video");
    this.videoElement.autoplay = true;
    this.videoElement.playsInline = true;
    this.videoElement.muted = true;

    this.offscreenCanvas = document.createElement("canvas");
    this.offscreenCtx = this.offscreenCanvas.getContext("2d");

    this.isStreaming = false;
    this.audioEnabled = false;
    this.alarmActive = false;
    this.lastAlarmTime = 0;

    // Time-series chart history: 60 data points (1 per second)
    this.timelineHistory = Array(60).fill(100.0);
    this.chartTimer = null;

    // Audio context for alarm
    this.audioCtx = null;

    this.initDOM();
    this.initEventListeners();
    this.initChart();
    this.fetchStatus();
    this.fetchSamples();
    this.fetchAuditLogs();

    // Start with webcam by default
    this.startWebcam();
  }

  initDOM() {
    this.monitorImg = document.getElementById("monitor-img");
    this.monitorPlaceholder = document.getElementById("monitor-placeholder");
    this.alarmFlash = document.getElementById("alarm-flash");

    // KPIs
    this.kpiCompliance = document.getElementById("kpi-compliance");
    this.kpiComplianceBadge = document.getElementById("kpi-compliance-badge");
    this.kpiTotal = document.getElementById("kpi-total");
    this.kpiSafe = document.getElementById("kpi-safe");
    this.kpiDanger = document.getElementById("kpi-danger");
    this.kpiWarn = document.getElementById("kpi-warn");

    // HUD Telemetry
    this.hudLatency = document.getElementById("hud-latency");
    this.hudFps = document.getElementById("hud-fps");
    this.hudSafe = document.getElementById("hud-safe");
    this.hudDanger = document.getElementById("hud-danger");
    this.hudStatus = document.getElementById("hud-status");

    // Controls
    this.confSlider = document.getElementById("conf-slider");
    this.confValue = document.getElementById("conf-value");
    this.nmsSlider = document.getElementById("nms-slider");
    this.nmsValue = document.getElementById("nms-value");
    this.toggleLandmarks = document.getElementById("toggle-landmarks");
    this.toggleAudio = document.getElementById("toggle-audio");

    // Chart
    this.chartCanvas = document.getElementById("timeline-chart");
    this.chartCtx = this.chartCanvas ? this.chartCanvas.getContext("2d") : null;

    // Incident Table
    this.incidentTbody = document.getElementById("incident-tbody");
    this.exportBtn = document.getElementById("btn-export-log");
    this.clearBtn = document.getElementById("btn-clear-log");

    // Tabs
    this.tabWebcam = document.getElementById("tab-webcam");
    this.tabSamples = document.getElementById("tab-samples");
    this.tabUpload = document.getElementById("tab-upload");

    // Panels
    this.samplePanel = document.getElementById("sample-panel");
    this.uploadPanel = document.getElementById("upload-panel");
    this.controlsPanel = document.getElementById("controls-panel");
  }

  initEventListeners() {
    // Mode tabs
    this.tabWebcam.addEventListener("click", () => this.switchMode("webcam"));
    this.tabSamples.addEventListener("click", () => this.switchMode("sample"));
    this.tabUpload.addEventListener("click", () => this.switchMode("upload"));

    // Sliders
    this.confSlider.addEventListener("input", (e) => {
      this.confValue.textContent = `${Math.round(e.target.value * 100)}%`;
    });
    this.nmsSlider.addEventListener("input", (e) => {
      this.nmsValue.textContent = `${Math.round(e.target.value * 100)}%`;
    });

    // Audio alarm toggle
    this.toggleAudio.addEventListener("change", (e) => {
      this.audioEnabled = e.target.checked;
      if (this.audioEnabled && !this.audioCtx) {
        this.initAudio();
      }
    });

    // Export & Clear log
    if (this.exportBtn) {
      this.exportBtn.addEventListener("click", () => {
        window.location.href = "/api/audit-logs/export";
      });
    }
    if (this.clearBtn) {
      this.clearBtn.addEventListener("click", async () => {
        await fetch("/api/audit-logs/clear", { method: "POST" });
        this.fetchAuditLogs();
      });
    }

    // Drag and drop upload
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    if (dropzone && fileInput) {
      dropzone.addEventListener("click", () => fileInput.click());
      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
      dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
      dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });
      fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }
  }

  switchMode(mode) {
    this.mode = mode;
    [this.tabWebcam, this.tabSamples, this.tabUpload].forEach((tab) => tab.classList.remove("active"));

    if (mode === "webcam") {
      this.tabWebcam.classList.add("active");
      this.samplePanel.style.display = "none";
      this.uploadPanel.style.display = "none";
      this.startWebcam();
    } else if (mode === "sample") {
      this.tabSamples.classList.add("active");
      this.stopWebcam();
      this.samplePanel.style.display = "block";
      this.uploadPanel.style.display = "none";
      // Auto run first sample
      this.runSample("pic1.jpeg");
    } else if (mode === "upload") {
      this.tabUpload.classList.add("active");
      this.stopWebcam();
      this.samplePanel.style.display = "none";
      this.uploadPanel.style.display = "block";
    }
  }

  /* ----------------- Web Audio API Alarm ----------------- */
  initAudio() {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContext();
    } catch (e) {
      console.warn("Web Audio API not supported:", e);
    }
  }

  playViolationAlarm() {
    if (!this.audioEnabled) return;
    const now = Date.now();
    if (now - this.lastAlarmTime < 1500) return; // Debounce alarm
    this.lastAlarmTime = now;

    if (!this.audioCtx) this.initAudio();
    if (!this.audioCtx) return;

    if (this.audioCtx.state === "suspended") {
      this.audioCtx.resume();
    }

    try {
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      // Clean two-tone security chime
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, this.audioCtx.currentTime); // A5
      osc.frequency.setValueAtTime(440, this.audioCtx.currentTime + 0.15); // A4

      gain.gain.setValueAtTime(0.2, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.4);

      osc.start();
      osc.stop(this.audioCtx.currentTime + 0.45);
    } catch (err) {
      console.warn("Error playing audio alarm:", err);
    }
  }

  /* ----------------- Live Webcam & WebSocket Streaming ----------------- */
  async startWebcam() {
    if (this.isStreaming) return;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      this.videoElement.srcObject = this.stream;
      await this.videoElement.play();

      this.offscreenCanvas.width = this.videoElement.videoWidth || 640;
      this.offscreenCanvas.height = this.videoElement.videoHeight || 480;

      this.isStreaming = true;
      this.monitorPlaceholder.style.display = "none";
      this.monitorImg.style.display = "block";

      this.connectWebSocket();
    } catch (err) {
      console.warn("Webcam access error or denied:", err);
      this.monitorPlaceholder.style.display = "flex";
      this.monitorPlaceholder.querySelector("p").textContent =
        "Webcam unavailable or permission denied. Please select a Demo Sample or Upload an Image.";
    }
  }

  stopWebcam() {
    this.isStreaming = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
  }

  connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.hudStatus.textContent = "CONNECTED (LIVE)";
      this.hudStatus.style.color = "var(--color-safe)";
      this.streamLoop();
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.frame) {
          this.monitorImg.src = data.frame;
        }
        if (data.summary) {
          this.updateKPIs(data.summary);
          this.updateHUD(data.summary, data.latency_ms);
        }
      } catch (e) {
        console.error("WS Parse Error:", e);
      }
    };

    this.ws.onclose = () => {
      this.hudStatus.textContent = "DISCONNECTED";
      this.hudStatus.style.color = "var(--color-danger)";
      if (this.isStreaming && this.mode === "webcam") {
        setTimeout(() => this.connectWebSocket(), 2000);
      }
    };
  }

  streamLoop() {
    if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    if (this.videoElement.readyState >= 2) {
      this.offscreenCtx.drawImage(
        this.videoElement,
        0,
        0,
        this.offscreenCanvas.width,
        this.offscreenCanvas.height
      );
      const dataUrl = this.offscreenCanvas.toDataURL("image/jpeg", 0.7);

      const payload = {
        image: dataUrl,
        confidence: parseFloat(this.confSlider.value),
        nms: parseFloat(this.nmsSlider.value),
        draw_landmarks: this.toggleLandmarks.checked,
      };
      this.ws.send(JSON.stringify(payload));
    }

    // Target ~15 FPS transmission
    setTimeout(() => this.streamLoop(), 65);
  }

  /* ----------------- Samples & File Upload ----------------- */
  async fetchSamples() {
    try {
      const res = await fetch("/api/samples");
      const samples = await res.json();
      const listContainer = document.getElementById("sample-list");
      if (!listContainer) return;
      listContainer.innerHTML = "";

      samples.forEach((s) => {
        const btn = document.createElement("button");
        btn.className = "sample-btn";
        btn.innerHTML = `
          <div class="sample-badge ${s.expected === "with_mask" ? "mask" : "nomask"}">
            ${s.expected === "with_mask" ? "SAFE" : "VIOLATION"}
          </div>
          <div class="sample-details">
            <h4>${s.title}</h4>
            <p>${s.description}</p>
          </div>
        `;
        btn.addEventListener("click", () => {
          document.querySelectorAll(".sample-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          this.runSample(s.id);
        });
        listContainer.appendChild(btn);
      });
    } catch (e) {
      console.warn("Could not fetch samples:", e);
    }
  }

  async runSample(sampleId) {
    this.monitorPlaceholder.style.display = "none";
    this.monitorImg.style.display = "block";

    try {
      const conf = this.confSlider.value;
      const nms = this.nmsSlider.value;
      const landmarks = this.toggleLandmarks.checked;

      const res = await fetch(
        `/api/detect/sample/${sampleId}?confidence=${conf}&nms=${nms}&draw_landmarks=${landmarks}`,
        { method: "POST" }
      );
      const data = await res.json();
      if (data.annotated_image) {
        this.monitorImg.src = data.annotated_image;
      }
      if (data.summary) {
        this.updateKPIs(data.summary);
        this.updateHUD(data.summary, data.processing_time_ms);
      }
      this.fetchAuditLogs();
    } catch (err) {
      console.error("Error running sample:", err);
    }
  }

  async handleFileUpload(file) {
    this.monitorPlaceholder.style.display = "none";
    this.monitorImg.style.display = "block";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("confidence", this.confSlider.value);
    formData.append("nms", this.nmsSlider.value);
    formData.append("draw_landmarks", this.toggleLandmarks.checked);

    try {
      const res = await fetch("/api/detect/image", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.annotated_image) {
        this.monitorImg.src = data.annotated_image;
      }
      if (data.summary) {
        this.updateKPIs(data.summary);
        this.updateHUD(data.summary, data.processing_time_ms);
      }
      this.fetchAuditLogs();
    } catch (err) {
      console.error("Upload detection error:", err);
    }
  }

  /* ----------------- Telemetry, KPIs & Alarms ----------------- */
  updateKPIs(summary) {
    this.kpiCompliance.textContent = `${summary.compliance_score}%`;
    this.kpiComplianceBadge.textContent = summary.compliance_status;
    this.kpiTotal.textContent = summary.total_faces;
    this.kpiSafe.textContent = summary.with_mask;
    this.kpiDanger.textContent = summary.without_mask;
    this.kpiWarn.textContent = summary.mask_weared_incorrect;

    // Trigger Alarm if violation present
    const hasViolation = summary.without_mask > 0 || summary.mask_weared_incorrect > 0;
    if (hasViolation) {
      this.alarmFlash.classList.add("active");
      this.playViolationAlarm();
    } else {
      this.alarmFlash.classList.remove("active");
    }

    // Push into timeline chart
    this.timelineHistory.push(summary.compliance_score);
    if (this.timelineHistory.length > 60) {
      this.timelineHistory.shift();
    }
  }

  updateHUD(summary, latencyMs) {
    if (this.hudLatency && latencyMs) {
      this.hudLatency.textContent = `${Math.round(latencyMs)} ms`;
    }
    if (this.hudSafe) this.hudSafe.textContent = summary.with_mask;
    if (this.hudDanger) this.hudDanger.textContent = summary.without_mask;
  }

  /* ----------------- Rolling Timeline Canvas Chart ----------------- */
  initChart() {
    if (!this.chartCanvas) return;
    // Resize canvas
    this.chartCanvas.width = this.chartCanvas.parentElement.clientWidth || 600;
    this.chartCanvas.height = 140;

    window.addEventListener("resize", () => {
      this.chartCanvas.width = this.chartCanvas.parentElement.clientWidth || 600;
    });

    setInterval(() => this.renderChart(), 1000);
  }

  renderChart() {
    if (!this.chartCtx) return;
    const ctx = this.chartCtx;
    const w = this.chartCanvas.width;
    const h = this.chartCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let y = 0; y <= 100; y += 25) {
      const py = h - (y / 100) * (h - 20) - 10;
      ctx.beginPath();
      ctx.moveTo(30, py);
      ctx.lineTo(w, py);
      ctx.stroke();

      ctx.fillStyle = "#64748b";
      ctx.font = "9px monospace";
      ctx.fillText(`${y}%`, 4, py + 3);
    }

    // Safety threshold line (90%)
    const safeY = h - (90 / 100) * (h - 20) - 10;
    ctx.strokeStyle = "rgba(16, 185, 129, 0.3)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(30, safeY);
    ctx.lineTo(w, safeY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Plot compliance data
    const points = this.timelineHistory;
    const step = (w - 40) / Math.max(points.length - 1, 1);

    ctx.beginPath();
    points.forEach((val, i) => {
      const x = 35 + i * step;
      const y = h - (val / 100) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    // Stroke line
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Area fill gradient
    ctx.lineTo(35 + (points.length - 1) * step, h - 10);
    ctx.lineTo(35, h - 10);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(99, 102, 241, 0.25)");
    grad.addColorStop(1, "rgba(99, 102, 241, 0.0)");
    ctx.fillStyle = grad;
    ctx.fill();
  }

  /* ----------------- Incident Audit Log ----------------- */
  async fetchAuditLogs() {
    try {
      const res = await fetch("/api/audit-logs");
      const logs = await res.json();
      if (!this.incidentTbody) return;

      if (logs.length === 0) {
        this.incidentTbody.innerHTML = `
          <tr>
            <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 24px;">
              No compliance incidents recorded in current session. All monitored faces compliant.
            </td>
          </tr>
        `;
        return;
      }

      this.incidentTbody.innerHTML = "";
      logs.slice(0, 15).forEach((item) => {
        const row = document.createElement("tr");
        const isViolation = item.violation_type === "without_mask";
        row.innerHTML = `
          <td>#${item.id}</td>
          <td>${item.timestamp}</td>
          <td>
            <span class="badge-tag ${isViolation ? "violation" : "warning"}">
              ${isViolation ? "VIOLATION (NO MASK)" : "INCORRECT POSITION"}
            </span>
          </td>
          <td>${item.confidence}%</td>
          <td><code>(${item.box.join(", ")})</code></td>
        `;
        this.incidentTbody.appendChild(row);
      });
    } catch (e) {
      console.warn("Could not fetch audit logs:", e);
    }
  }

  async fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      const el = document.getElementById("active-model-name");
      if (el) {
        el.textContent = `${data.detector} + ${data.classifier}`;
      }
    } catch (e) {
      console.warn("Could not fetch status:", e);
    }
  }
}

// Instantiate dashboard when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.dashboard = new AegisMaskDashboard();
});

