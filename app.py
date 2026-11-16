"""Face-mask detection web app (FastAPI backend + browser front end).

The visitor's BROWSER captures the camera / screen / video / photo and this server runs the YOLO model,
so one link works on a PC, a laptop or a phone, locally or hosted (e.g. Hugging Face Spaces).

    python app.py                  # opens http://127.0.0.1:7860
    python app.py --port 8000      # other port
    python app.py --host 0.0.0.0   # reachable on your network (camera then needs https, see README)

Needs weights/best.pt (or best.pt next to this file).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
_cfg = HERE / ".yolo_config" if os.access(HERE, os.W_OK) else Path(tempfile.gettempdir()) / "yolo_config"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_cfg))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WEB_DIR = HERE / "web"
WEIGHT_CANDIDATES = [HERE / "weights" / "best.pt", HERE / "best.pt"]
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "300")) * 1024 * 1024
MAX_VIDEO_FRAMES = 30000
KEEP_SECONDS = 3600

# BGR colours used when exporting annotated videos (same palette as the front end)
COLORS_BGR = {"with_mask": (153, 211, 52), "mask_weared_incorrect": (36, 191, 251), "without_mask": (133, 113, 251)}
PRETTY = {"with_mask": "Mask OK", "mask_weared_incorrect": "Mask incorrect", "without_mask": "No mask"}


# ----------------------------------------------------------------------------- model
def find_weights() -> Path:
    for p in WEIGHT_CANDIDATES:
        if p.exists():
            return p
    sys.exit("[error] Model weights not found. Expected weights/best.pt (or best.pt next to app.py).")


def pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


MODEL = YOLO(str(find_weights()))
NAMES = {int(k): v for k, v in MODEL.names.items()}
DEVICE = pick_device()
try:
    TRAIN_IMGSZ = int(MODEL.overrides.get("imgsz", 512))
except Exception:
    TRAIN_IMGSZ = 512
LOCK = threading.Lock()  # one inference at a time -> safe with several visitors
EXTRA = {"half": True} if DEVICE != "cpu" else {}


def infer(bgr, conf: float, imgsz: int):
    """Run the model on a BGR image. Returns (list of detections in pixel coords, milliseconds)."""
    t0 = time.perf_counter()
    with LOCK:
        res = MODEL.predict(bgr, imgsz=imgsz, conf=conf, iou=0.5, agnostic_nms=True, device=DEVICE,
                            verbose=False, **EXTRA)[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, cf):
            dets.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                         "conf": round(float(s), 3), "cls": int(c), "name": NAMES.get(int(c), str(c))})
    return dets, (time.perf_counter() - t0) * 1000.0


def clamp_params(conf, imgsz):
    conf = min(max(float(conf), 0.05), 0.95)
    imgsz = int(round(min(max(int(imgsz), 160), 1280) / 32.0)) * 32
    return conf, imgsz


# ----------------------------------------------------------------------------- app
app = FastAPI(title="Face Mask Detection", docs_url=None, redoc_url=None)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return {"ok": True, "device": DEVICE, "classes": [NAMES[i] for i in sorted(NAMES)], "imgsz": TRAIN_IMGSZ}


@app.post("/api/detect")
async def api_detect(request: Request, conf: float = 0.4, imgsz: int = 512):
    """Body = one JPEG/PNG frame. Returns detections in that frame's pixel coordinates."""
    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty body: send an image (JPEG or PNG) as the request body.")
    if len(data) > MAX_FRAME_BYTES:
        raise HTTPException(413, "Image too large (max 8 MB).")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the image.")
    conf, imgsz = clamp_params(conf, imgsz)
    dets, ms = await run_in_threadpool(infer, img, conf, imgsz)
    h, w = img.shape[:2]
    return {"w": w, "h": h, "ms": round(ms, 1), "detections": dets}


# ----------------------------------------------------------------------------- video export jobs
def work_dir() -> Path:
    for base in (HERE / "outputs" / "web_videos", Path(tempfile.gettempdir()) / "maskvid"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".w"
            probe.write_text("ok")
            probe.unlink()
            return base
        except Exception:
            continue
    raise RuntimeError("No writable folder for video processing")


JOBS: dict = {}
JOBS_LOCK = threading.Lock()


def draw_boxes(img, dets):
    """Draw boxes, labels and a summary bar on a BGR image (in place)."""
    h, w = img.shape[:2]
    thick = max(2, round(min(h, w) / 280))
    fscale = max(0.45, min(h, w) / 900.0)
    counts = {"with_mask": 0, "mask_weared_incorrect": 0, "without_mask": 0}
    for d in dets:
        name = d["name"]
        counts[name] = counts.get(name, 0) + 1
        col = COLORS_BGR.get(name, (200, 200, 200))
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        label = f"{PRETTY.get(name, name)} {d['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
        ty = y1 - 4 if y1 - th - base - 6 > 0 else y2 + th + base + 4
        cv2.rectangle(img, (x1, ty - th - base - 2), (x1 + tw + 8, ty + 2), col, -1)
        cv2.putText(img, label, (x1 + 4, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (20, 20, 20), 1, cv2.LINE_AA)
    bar = int(30 * max(1.0, fscale * 1.1))
    over = img.copy()
    cv2.rectangle(over, (0, 0), (w, bar), (20, 20, 20), -1)
    cv2.addWeighted(over, 0.65, img, 0.35, 0, img)
    text = (f"OK {counts['with_mask']}   Incorrect {counts['mask_weared_incorrect']}   "
            f"No mask {counts['without_mask']}")
    cv2.putText(img, text, (10, int(bar * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * max(1.0, fscale * 1.1),
                (255, 255, 255), 1, cv2.LINE_AA)


def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to browser-playable H.264 with a bundled or system ffmpeg. Returns False if impossible."""
    exe = None
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True, timeout=1800)
        return dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def run_video_job(job_id: str, src: Path, conf: float, imgsz: int, every: int):
    job = JOBS[job_id]
    cap = None
    writer = None
    raw = src.with_name(f"{job_id}_raw.mp4")
    out = src.with_name(f"{job_id}_annotated.mp4")
    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("Could not open the video (unsupported format?).")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 < fps < 121 else 25.0
        job.update(state="processing", total=total)
        n, cached, tot = 0, [], {"with_mask": 0, "mask_weared_incorrect": 0, "without_mask": 0}
        scale = 1.0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or n >= MAX_VIDEO_FRAMES:
                break
            if writer is None:
                h0, w0 = frame.shape[:2]
                scale = min(1.0, 960.0 / max(h0, w0))
                size = (int(w0 * scale) // 2 * 2, int(h0 * scale) // 2 * 2)
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError("Could not create the output video.")
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
            if n % every == 0:
                cached, _ = infer(small, conf, imgsz)
                for d in cached:
                    tot[d["name"]] = tot.get(d["name"], 0) + 1
            draw_boxes(small, cached)
            writer.write(small)
            n += 1
            job["frames"] = n
            job["progress"] = round(n / total, 3) if total else 0.0
            if job.get("cancel"):
                raise RuntimeError("Cancelled")
        if n == 0:
            raise RuntimeError("The video has no readable frames.")
        writer.release()
        writer = None
        job.update(state="encoding", progress=0.99)
        final = out if to_h264(raw, out) else raw
        job.update(state="done", progress=1.0, path=str(final), seconds=round(time.time() - t0, 1),
                   playable=(final == out))
    except Exception as exc:  # noqa: BLE001
        job.update(state="error", message=str(exc))
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        try:
            src.unlink(missing_ok=True)
            if job.get("path") != str(raw):
                raw.unlink(missing_ok=True)
        except Exception:
            pass


def prune_jobs():
    now = time.time()
    with JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if now - v["created"] > KEEP_SECONDS]:
            p = JOBS.pop(jid).get("path")
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass


@app.post("/api/video")
async def api_video(request: Request, conf: float = 0.4, imgsz: int = 512, every: int = 2, name: str = "video.mp4"):
    """Body = the raw video file. Starts a background job that returns an annotated MP4."""
    prune_jobs()
    if sum(1 for v in JOBS.values() if v["state"] in ("queued", "processing", "encoding")) >= 2:
        raise HTTPException(429, "The server is busy with other videos. Try again in a minute.")
    conf, imgsz = clamp_params(conf, imgsz)
    every = min(max(int(every), 1), 10)
    job_id = uuid.uuid4().hex[:12]
    ext = Path(name).suffix.lower()
    ext = ext if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"} else ".mp4"
    src = work_dir() / f"{job_id}_in{ext}"
    size = 0
    try:
        with open(src, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Video too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB).")
                f.write(chunk)
    except HTTPException:
        src.unlink(missing_ok=True)
        raise
    if size == 0:
        src.unlink(missing_ok=True)
        raise HTTPException(400, "Empty upload.")
    JOBS[job_id] = {"state": "queued", "progress": 0.0, "frames": 0, "total": 0, "created": time.time(),
                    "name": Path(name).stem or "video"}
    threading.Thread(target=run_video_job, args=(job_id, src, conf, imgsz, every), daemon=True).start()
    return {"job": job_id}


@app.get("/api/video/{job_id}")
def api_video_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    return {k: job.get(k) for k in ("state", "progress", "frames", "total", "message", "seconds", "playable")}


@app.get("/api/video/{job_id}/download")
def api_video_download(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("state") != "done" or not Path(job.get("path", "")).exists():
        raise HTTPException(404, "Result not available.")
    return FileResponse(job["path"], media_type="video/mp4", filename=f"{job['name']}_masks.mp4")


@app.delete("/api/video/{job_id}")
def api_video_cancel(job_id: str):
    job = JOBS.get(job_id)
    if job:
        job["cancel"] = True
    return {"ok": True}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="127.0.0.1 = this PC only (the camera works on localhost without https)")
    ap.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    args = ap.parse_args()

    if not (WEB_DIR / "index.html").exists():
        sys.exit(f"[error] Front end missing: {WEB_DIR / 'index.html'}")
    # warm-up so the first live frame is not slow
    infer(np.zeros((240, 320, 3), np.uint8), 0.4, TRAIN_IMGSZ)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"[info] Model ready on '{DEVICE}'. Open {url}  (Ctrl+C to stop)", flush=True)
    if not (args.no_browser or os.environ.get("SPACE_ID")):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
