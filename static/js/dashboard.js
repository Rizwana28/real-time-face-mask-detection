/**
 * AegisMask — Dashboard Controller
 * Handles: WebSocket webcam stream, sample inference,
 * file upload inference, timeline chart, audit log.
 */

class Dashboard {
  constructor() {
    this.mode = 'webcam';
    this.ws = null;
    this.stream = null;
    this.isStreaming = false;
    this.audioEnabled = false;
    this.audioCtx = null;
    this.lastAlarmAt = 0;

    // Offscreen canvas for frame capture
    this.captureCanvas = document.createElement('canvas');
    this.captureCtx = this.captureCanvas.getContext('2d');
    this.videoEl = document.createElement('video');
    this.videoEl.autoplay = true;
    this.videoEl.playsInline = true;
    this.videoEl.muted = true;

    // Timeline history (60 s)
    this.timeline = Array(60).fill(100);

    this._bindDOM();
    this._bindEvents();
    this._initChart();
    this._fetchStatus();
    this._fetchSamples();
    this._fetchAuditLogs();
    this._startWebcam();

    // Refresh audit log every 5 s
    setInterval(() => this._fetchAuditLogs(), 5000);
  }

  /* ================================================================ DOM */

  _bindDOM() {
    this.monitorImg         = document.getElementById('monitor-img');
    this.monitorPlaceholder = document.getElementById('monitor-placeholder');
    this.alarmFlash         = document.getElementById('alarm-flash');

    this.kpiCompliance      = document.getElementById('kpi-compliance');
    this.kpiBadge           = document.getElementById('kpi-compliance-badge');
    this.kpiTotal           = document.getElementById('kpi-total');
    this.kpiSafe            = document.getElementById('kpi-safe');
    this.kpiDanger          = document.getElementById('kpi-danger');
    this.kpiWarn            = document.getElementById('kpi-warn');

    this.hudLatency         = document.getElementById('hud-latency');
    this.hudSafe            = document.getElementById('hud-safe');
    this.hudDanger          = document.getElementById('hud-danger');
    this.hudStatus          = document.getElementById('hud-status');

    this.confSlider         = document.getElementById('conf-slider');
    this.confValue          = document.getElementById('conf-value');
    this.nmsSlider          = document.getElementById('nms-slider');
    this.nmsValue           = document.getElementById('nms-value');
    this.chkLandmarks       = document.getElementById('toggle-landmarks');
    this.chkAudio           = document.getElementById('toggle-audio');

    this.chartCanvas        = document.getElementById('timeline-chart');
    this.chartCtx           = this.chartCanvas ? this.chartCanvas.getContext('2d') : null;

    this.incidentTbody      = document.getElementById('incident-tbody');
    this.btnExport          = document.getElementById('btn-export-log');
    this.btnClear           = document.getElementById('btn-clear-log');

    this.tabWebcam          = document.getElementById('tab-webcam');
    this.tabSamples         = document.getElementById('tab-samples');
    this.tabUpload          = document.getElementById('tab-upload');
    this.samplePanel        = document.getElementById('sample-panel');
    this.uploadPanel        = document.getElementById('upload-panel');
  }

  /* ============================================================= EVENTS */

  _bindEvents() {
    this.tabWebcam.addEventListener('click',  () => this._switchMode('webcam'));
    this.tabSamples.addEventListener('click', () => this._switchMode('sample'));
    this.tabUpload.addEventListener('click',  () => this._switchMode('upload'));

    this.confSlider.addEventListener('input', e =>
      this.confValue.textContent = `${Math.round(e.target.value * 100)}%`);
    this.nmsSlider.addEventListener('input', e =>
      this.nmsValue.textContent = `${Math.round(e.target.value * 100)}%`);

    this.chkAudio.addEventListener('change', e => {
      this.audioEnabled = e.target.checked;
      if (this.audioEnabled && !this.audioCtx) this._initAudio();
    });

    this.btnExport && this.btnExport.addEventListener('click', () => {
      window.location.href = '/api/audit-logs/export';
    });
    this.btnClear && this.btnClear.addEventListener('click', async () => {
      await fetch('/api/audit-logs/clear', { method: 'POST' });
      this._fetchAuditLogs();
    });

    // Drag-and-drop upload
    const dropzone  = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());
      dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
      dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
      dropzone.addEventListener('drop', e => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) this._uploadFile(e.dataTransfer.files[0]);
      });
      fileInput.addEventListener('change', e => {
        if (e.target.files.length) this._uploadFile(e.target.files[0]);
      });
    }
  }

  /* ======================================================= MODE SWITCH */

  _switchMode(mode) {
    this.mode = mode;
    [this.tabWebcam, this.tabSamples, this.tabUpload].forEach(t => t.classList.remove('active'));

    if (mode === 'webcam') {
      this.tabWebcam.classList.add('active');
      this.samplePanel.style.display = 'none';
      this.uploadPanel.style.display = 'none';
      this._startWebcam();
    } else if (mode === 'sample') {
      this.tabSamples.classList.add('active');
      this._stopWebcam();
      this.samplePanel.style.display = 'block';
      this.uploadPanel.style.display = 'none';
    } else {
      this.tabUpload.classList.add('active');
      this._stopWebcam();
      this.samplePanel.style.display = 'none';
      this.uploadPanel.style.display = 'block';
    }
  }

  /* ====================================================== AUDIO ALARM */

  _initAudio() {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AC();
    } catch (_) {}
  }

  _alarm() {
    if (!this.audioEnabled) return;
    const now = Date.now();
    if (now - this.lastAlarmAt < 1500) return;
    this.lastAlarmAt = now;
    if (!this.audioCtx) this._initAudio();
    if (!this.audioCtx) return;
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
    try {
      const osc  = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, this.audioCtx.currentTime);
      osc.frequency.setValueAtTime(440, this.audioCtx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.15, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.4);
      osc.start();
      osc.stop(this.audioCtx.currentTime + 0.45);
    } catch (_) {}
  }

  /* ======================================================= WEBCAM / WS */

  async _startWebcam() {
    if (this.isStreaming) return;
    this._setStatus('CONNECTING');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      this.videoEl.srcObject = this.stream;
      await this.videoEl.play();
      this.captureCanvas.width  = this.videoEl.videoWidth  || 640;
      this.captureCanvas.height = this.videoEl.videoHeight || 480;
      this.isStreaming = true;
      this._showMonitor();
      this._connectWS();
    } catch (err) {
      this._setStatus('NO CAMERA');
      this._showPlaceholder('Camera unavailable or permission denied. Use Demo Samples or Upload an image.');
    }
  }

  _stopWebcam() {
    this.isStreaming = false;
    if (this.ws) { this.ws.close(); this.ws = null; }
    if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
    this._setStatus('IDLE');
  }

  _connectWS() {
    // Always use wss:// on https:// (required by browsers)
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url   = `${proto}//${location.host}/ws/stream`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this._setStatus('LIVE');
      this._streamLoop();
    };

    this.ws.onmessage = evt => {
      try {
        const d = JSON.parse(evt.data);
        if (d.error) {
          // Model not ready — show friendly message, don't crash
          this._setStatus('MODEL LOADING');
          return;
        }
        if (d.frame) this.monitorImg.src = d.frame;
        if (d.summary) {
          this._updateKPIs(d.summary);
          if (d.latency_ms) this.hudLatency.textContent = `${Math.round(d.latency_ms)} ms`;
        }
      } catch (_) {}
    };

    this.ws.onerror = () => this._setStatus('WS ERROR');

    this.ws.onclose = () => {
      this._setStatus('DISCONNECTED');
      if (this.isStreaming && this.mode === 'webcam') {
        setTimeout(() => this._connectWS(), 2000);
      }
    };
  }

  _streamLoop() {
    if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (this.videoEl.readyState >= 2) {
      this.captureCtx.drawImage(this.videoEl, 0, 0,
        this.captureCanvas.width, this.captureCanvas.height);
      this.ws.send(JSON.stringify({
        image:          this.captureCanvas.toDataURL('image/jpeg', 0.7),
        confidence:     parseFloat(this.confSlider.value),
        nms:            parseFloat(this.nmsSlider.value),
        draw_landmarks: this.chkLandmarks.checked,
      }));
    }
    // ~15 FPS
    setTimeout(() => this._streamLoop(), 67);
  }

  /* ====================================================== SAMPLES */

  async _fetchSamples() {
    try {
      const res     = await fetch('/api/samples');
      const samples = await res.json();
      const list    = document.getElementById('sample-list');
      if (!list) return;
      list.innerHTML = '';
      samples.forEach(s => {
        const btn = document.createElement('button');
        btn.className = 'sample-btn';
        btn.innerHTML = `
          <div class="sample-tag">${s.expected === 'with_mask' ? 'COMPLIANT' : 'VIOLATION'}</div>
          <h4>${s.title}</h4>
          <p>${s.description}</p>`;
        btn.addEventListener('click', () => {
          document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          this._runSample(s.id);
        });
        list.appendChild(btn);
      });
    } catch (_) {}
  }

  async _runSample(id) {
    this._setStatus('PROCESSING');
    this._showMonitor();
    try {
      const conf = this.confSlider.value;
      const nms  = this.nmsSlider.value;
      const lmk  = this.chkLandmarks.checked;
      const res  = await fetch(
        `/api/detect/sample/${id}?confidence=${conf}&nms=${nms}&draw_landmarks=${lmk}`,
        { method: 'POST' }
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        this._setStatus('ERROR');
        this._showPlaceholder(err.detail || 'Detection failed. Models may not be loaded.');
        return;
      }
      const d = await res.json();
      if (d.annotated_image) this.monitorImg.src = d.annotated_image;
      if (d.summary) {
        this._updateKPIs(d.summary);
        if (d.processing_time_ms) this.hudLatency.textContent = `${Math.round(d.processing_time_ms)} ms`;
      }
      this._setStatus('READY');
      this._fetchAuditLogs();
    } catch (err) {
      this._setStatus('ERROR');
      this._showPlaceholder('Request failed: ' + err.message);
    }
  }

  /* ====================================================== UPLOAD */

  async _uploadFile(file) {
    this._setStatus('PROCESSING');
    this._showMonitor();
    const fd = new FormData();
    fd.append('file', file);
    fd.append('confidence', this.confSlider.value);
    fd.append('nms',        this.nmsSlider.value);
    fd.append('draw_landmarks', this.chkLandmarks.checked);
    try {
      const res = await fetch('/api/detect/image', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        this._setStatus('ERROR');
        this._showPlaceholder(err.detail || 'Upload detection failed. Models may not be loaded.');
        return;
      }
      const d = await res.json();
      if (d.annotated_image) this.monitorImg.src = d.annotated_image;
      if (d.summary) {
        this._updateKPIs(d.summary);
        if (d.processing_time_ms) this.hudLatency.textContent = `${Math.round(d.processing_time_ms)} ms`;
      }
      this._setStatus('READY');
      this._fetchAuditLogs();
    } catch (err) {
      this._setStatus('ERROR');
      this._showPlaceholder('Upload failed: ' + err.message);
    }
  }

  /* ====================================================== KPIs */

  _updateKPIs(s) {
    this.kpiCompliance.textContent = `${s.compliance_score}%`;
    this.kpiBadge.textContent      = s.compliance_status;
    this.kpiTotal.textContent      = s.total_faces;
    this.kpiSafe.textContent       = s.with_mask;
    this.kpiDanger.textContent     = s.without_mask;
    this.kpiWarn.textContent       = s.mask_weared_incorrect;
    this.hudSafe.textContent       = s.with_mask;
    this.hudDanger.textContent     = s.without_mask;

    const hasViolation = s.without_mask > 0 || s.mask_weared_incorrect > 0;
    if (hasViolation) {
      this.alarmFlash.classList.add('active');
      this._alarm();
      setTimeout(() => this.alarmFlash.classList.remove('active'), 500);
    }

    this.timeline.push(s.compliance_score);
    if (this.timeline.length > 60) this.timeline.shift();
  }

  /* ====================================================== CHART */

  _initChart() {
    if (!this.chartCanvas) return;
    const resize = () => {
      this.chartCanvas.width = this.chartCanvas.parentElement.clientWidth || 268;
    };
    resize();
    window.addEventListener('resize', resize);
    setInterval(() => this._drawChart(), 1000);
  }

  _drawChart() {
    const ctx = this.chartCtx;
    if (!ctx) return;
    const W = this.chartCanvas.width;
    const H = this.chartCanvas.height;
    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#1e1e1e';
    ctx.lineWidth = 1;
    [0, 25, 50, 75, 100].forEach(v => {
      const y = H - 8 - (v / 100) * (H - 16);
      ctx.beginPath(); ctx.moveTo(28, y); ctx.lineTo(W, y); ctx.stroke();
      ctx.fillStyle = '#444';
      ctx.font = '8px monospace';
      ctx.fillText(`${v}`, 2, y + 3);
    });

    // 90% threshold
    const ty = H - 8 - (90 / 100) * (H - 16);
    ctx.strokeStyle = '#333';
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(28, ty); ctx.lineTo(W, ty); ctx.stroke();
    ctx.setLineDash([]);

    // Line
    const pts  = this.timeline;
    const step = (W - 36) / Math.max(pts.length - 1, 1);
    ctx.beginPath();
    pts.forEach((v, i) => {
      const x = 32 + i * step;
      const y = H - 8 - (v / 100) * (H - 16);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Fill
    ctx.lineTo(32 + (pts.length - 1) * step, H - 8);
    ctx.lineTo(32, H - 8);
    ctx.closePath();
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, 'rgba(255,255,255,0.08)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fill();
  }

  /* ==================================================== AUDIT LOG */

  async _fetchAuditLogs() {
    try {
      const res  = await fetch('/api/audit-logs');
      const logs = await res.json();
      if (!this.incidentTbody) return;
      if (!logs.length) {
        this.incidentTbody.innerHTML =
          '<tr><td colspan="4" style="text-align:center;padding:20px 0;color:var(--text-dim)">No incidents recorded</td></tr>';
        return;
      }
      this.incidentTbody.innerHTML = '';
      logs.slice(0, 20).forEach(item => {
        const tr = document.createElement('tr');
        const isViol = item.violation_type === 'without_mask';
        // Short timestamp: HH:MM:SS
        const ts = item.timestamp.split(' ')[1] || item.timestamp;
        tr.innerHTML = `
          <td>#${item.id}</td>
          <td style="font-family:var(--mono)">${ts}</td>
          <td><span class="badge">${isViol ? 'NO MASK' : 'INCORRECT'}</span></td>
          <td style="font-family:var(--mono)">${item.confidence}%</td>`;
        this.incidentTbody.appendChild(tr);
      });
    } catch (_) {}
  }

  /* ====================================================== STATUS / API */

  async _fetchStatus() {
    try {
      const res  = await fetch('/api/status');
      const data = await res.json();
      const el   = document.getElementById('active-model-name');
      if (el) el.textContent = `${data.detector} + ${data.classifier}`;
    } catch (_) {}
  }

  /* ====================================================== HELPERS */

  _showMonitor() {
    this.monitorImg.style.display         = 'block';
    this.monitorPlaceholder.style.display = 'none';
  }

  _showPlaceholder(msg) {
    this.monitorImg.style.display         = 'none';
    this.monitorPlaceholder.style.display = 'flex';
    const p = this.monitorPlaceholder.querySelector('p');
    if (p) p.textContent = msg;
  }

  _setStatus(text) {
    if (this.hudStatus) this.hudStatus.textContent = text;
  }
}

document.addEventListener('DOMContentLoaded', () => { window.app = new Dashboard(); });
