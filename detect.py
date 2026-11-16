"""Real-time face-mask detection for images, folders, videos and webcams.

Classes: with_mask (green) | mask_weared_incorrect (orange) | without_mask (red)

Examples:
    python detect.py --source 0                      # webcam (index 0)
    python detect.py --source photo.jpg              # single image
    python detect.py --source my_images/             # folder of images (+ results.csv)
    python detect.py --source clip.mp4 --save        # video, annotated copy saved
    python detect.py --source 0 --save               # record the webcam session
    python detect.py --source screen                 # live detection on your screen (monitor 1)
    python detect.py --source screen:2               # second monitor
    python detect.py --source screen:100,100,800,600 # only this region: left,top,width,height

Webcam keys:  q / ESC = quit   s = save snapshot   p = pause/resume
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common import (COLOR_BY_NAME, DEFAULT_COLOR, DEFAULT_WEIGHTS, IMAGE_EXTS, PRETTY, ROOT, VIDEO_EXTS,
                    is_cuda, pick_device)

WINDOW = "Face Mask Detection  (q = quit, s = snapshot, p = pause)"


# ----------------------------------------------------------------------------- drawing
def draw_detections(frame, result, names, fps=None):
    """Draw boxes + summary banner on `frame` (in place). Returns {class_name: count}."""
    h, w = frame.shape[:2]
    scale = max(0.5, min(h, w) / 900.0)
    thick = max(1, int(round(2 * scale)))
    counts = {n: 0 for n in names.values()}

    boxes = result.boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            name = names.get(int(c), str(c))
            counts[name] = counts.get(name, 0) + 1
            color = COLOR_BY_NAME.get(name, DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            label = f"{PRETTY.get(name, name)} {s:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, thick)
            ty = y1 - 4 if y1 - th - base - 6 > 0 else y2 + th + base + 4
            cv2.rectangle(frame, (x1, ty - th - base - 2), (x1 + tw + 6, ty + 2), color, -1)
            cv2.putText(frame, label, (x1 + 3, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * scale, (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)

    ok, bad_inc, bad_no = counts.get("with_mask", 0), counts.get("mask_weared_incorrect", 0), counts.get("without_mask", 0)
    total = ok + bad_inc + bad_no
    if total == 0:
        status, scol = "No faces detected", (120, 120, 120)
    elif bad_inc or bad_no:
        status, scol = "ALERT: mask problem", (0, 0, 220)
    else:
        status, scol = "All masks OK", (0, 160, 0)

    bar_h = int(34 * max(1.0, scale))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.circle(frame, (int(bar_h * 0.5), bar_h // 2), int(bar_h * 0.22), scol, -1)
    text = f"{status}  |  OK: {ok}  Incorrect: {bad_inc}  No mask: {bad_no}"
    if fps is not None:
        text += f"  |  {fps:4.1f} FPS"
    x0 = int(bar_h * 0.95)
    fscale = 0.6 * max(1.0, scale * 0.9)
    while fscale > 0.25 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)[0][0] > w - x0 - 6:
        fscale -= 0.05  # shrink so the banner always fits the image width
    cv2.putText(frame, text, (x0, int(bar_h * 0.5 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)[0][1] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), 1, cv2.LINE_AA)
    return counts


# ----------------------------------------------------------------------------- io helpers
def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe on Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise IOError(f"Could not encode image for {path}")
    buf.tofile(str(path))


class ScreenGrabber:
    """Screen capture with the same tiny interface as cv2.VideoCapture (read / isOpened / release / get)."""

    def __init__(self, spec: str = "", max_width: int = 1280):
        self.max_width, self.sct, self.mon, self.ok = max_width, None, None, False
        try:
            import mss
        except ImportError:
            print("[error] Screen capture needs the 'mss' package:  pip install mss")
            return
        try:
            self.sct = mss.mss()
            mons = self.sct.monitors  # [0] = all monitors together, 1..n = single monitors
            spec = (spec or "").strip()
            if "," in spec:
                left, top, width, height = [int(float(v)) for v in spec.split(",")]
                if width < 10 or height < 10:
                    raise ValueError("region width/height too small")
                self.mon = {"left": left, "top": top, "width": width, "height": height}
            else:
                idx = int(spec) if spec else 1
                if not 1 <= idx < len(mons):
                    print(f"[error] Monitor {idx} not found. Available monitors: 1..{len(mons) - 1}")
                    return
                self.mon = dict(mons[idx])
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[error] Could not start screen capture: {exc}")

    def isOpened(self):
        return self.ok

    def read(self):
        try:
            shot = self.sct.grab(self.mon)
            frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])  # BGRA -> BGR
        except Exception:  # noqa: BLE001
            return False, None
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        return True, frame

    def get(self, prop):
        return 0

    def release(self):
        try:
            if self.sct is not None:
                self.sct.close()
        except Exception:  # noqa: BLE001
            pass


def open_capture(src, width, height):
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(src)
        if cap.isOpened():
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(src)
    return cap


class Display:
    """cv2.imshow wrapper that degrades gracefully on headless OpenCV builds."""

    def __init__(self, enabled: bool, small: bool = False):
        self.small = small
        # Headless Linux (no X/Wayland): OpenCV's Qt backend hard-aborts, so never try to open a window.
        if enabled and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[warn] No display detected -> running without a preview window (annotated output is saved).")
            enabled = False
        self.enabled = enabled
        self._created = False

    def show(self, img) -> int:
        if not self.enabled:
            return -1
        try:
            if not self._created:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                self._created = True
                if self.small:  # keep the preview out of the way so it does not fill the captured screen
                    try:
                        cv2.resizeWindow(WINDOW, 560, 320)
                        cv2.moveWindow(WINDOW, 20, 20)
                        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
                    except cv2.error:
                        pass
            cv2.imshow(WINDOW, img)
            return cv2.waitKey(1) & 0xFF
        except cv2.error:
            print("[warn] This OpenCV build has no GUI support -> running without a preview window. "
                  "Fix:  pip uninstall opencv-python-headless -y && pip install opencv-python")
            self.enabled = False
            return -1

    def closed(self) -> bool:
        if not (self.enabled and self._created):
            return False
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return False

    def wait_key(self, ms=0) -> int:
        if not self.enabled:
            return -1
        try:
            return cv2.waitKey(ms) & 0xFF
        except cv2.error:
            return -1

    def close(self):
        if self.enabled and self._created:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ----------------------------------------------------------------------------- pipeline
class Detector:
    def __init__(self, weights, device, imgsz, conf, iou):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.device = pick_device(device)
        if not imgsz:  # default: use the size the model was trained at
            try:
                imgsz = int(self.model.overrides.get("imgsz", 640))
            except Exception:
                imgsz = 640
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        # FP16 only on CUDA; the kwarg is omitted elsewhere (newer ultralytics warns when it is passed at all)
        self.extra = {"half": True} if is_cuda(self.device) else {}
        # warm-up so the first real frame is not slow
        self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), imgsz=self.imgsz,
                           device=self.device, verbose=False, **self.extra)

    def __call__(self, frame):
        return self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, agnostic_nms=True,
                                  device=self.device, verbose=False, **self.extra)[0]


def run_images(det, paths, out_dir, display, save, args):
    rows = []
    for i, p in enumerate(paths, 1):
        img = read_image(p)
        if img is None:
            print(f"[warn] cannot read {p}, skipped")
            continue
        res = det(img)
        counts = draw_detections(img, res, det.names)
        print(f"[{i}/{len(paths)}] {p.name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        rows.append([p.name] + [counts.get(n, 0) for n in det.names.values()])
        if save:
            write_image(out_dir / f"{p.stem}_pred{p.suffix if p.suffix.lower() in IMAGE_EXTS else '.jpg'}", img)
        if display.enabled:
            display.show(img)
            key = display.wait_key(0 if len(paths) == 1 else 700)
            if key in (ord("q"), 27) or display.closed():
                break
    if len(paths) > 1 and save and rows:
        csv_path = out_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image"] + list(det.names.values()))
            w.writerows(rows)
        print(f"[info] Summary table: {csv_path}")
    if save:
        print(f"[info] Annotated images saved in: {out_dir}")


def run_stream(det, source, out_dir, display, save, args, is_webcam, label=None):
    cap = source if hasattr(source, "read") and not isinstance(source, (str, int)) else open_capture(
        source, args.cam_width, args.cam_height)
    if not cap.isOpened():
        hint = ("Is another app using the camera? Try --source 1 (another camera index) and check "
                "Windows Settings > Privacy > Camera." if is_webcam else
                ("Use screen, screen:2 (monitor 2) or screen:left,top,width,height." if label == "screen"
                 else "Check the file path / URL."))
        sys.exit(f"[error] Could not open source '{label or source}'. {hint}")

    writer, fail, n, paused = None, 0, 0, False
    fps, last_t, snap_i = None, time.perf_counter(), 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    last_frame = None
    print("[info] Running. Press q / ESC in the window to quit." if display.enabled else "[info] Running (no window).")
    try:
        while True:
            if paused and last_frame is not None:
                display.show(last_frame)
                key = display.wait_key(30)
                if key == ord("p"):
                    paused = False
                elif key in (ord("q"), 27):
                    break
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                fail += 1
                if not is_webcam or fail > 30:
                    if is_webcam:
                        print("[error] Lost the camera feed.")
                    break
                time.sleep(0.03)
                continue
            fail = 0
            if is_webcam and args.mirror:
                frame = cv2.flip(frame, 1)

            res = det(frame)
            now = time.perf_counter()
            inst = 1.0 / max(now - last_t, 1e-6)
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            last_t = now
            draw_detections(frame, res, det.names, fps)
            last_frame = frame
            n += 1

            if save:
                if writer is None:
                    h, w = frame.shape[:2]
                    src_fps = cap.get(cv2.CAP_PROP_FPS)
                    src_fps = src_fps if 1 < src_fps < 120 else 25.0
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / f"{label or ('webcam' if is_webcam else Path(str(source)).stem)}_{stamp}.mp4"
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                    print(f"[info] Recording to {path}")
                writer.write(frame)

            key = display.show(frame)
            if key in (ord("q"), 27) or display.closed():
                break
            if key == ord("p"):
                paused = True
            if key == ord("s"):
                snap_i += 1
                snap = out_dir / f"snapshot_{stamp}_{snap_i}.jpg"
                write_image(snap, frame)
                print(f"[info] Snapshot saved: {snap}")
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print(f"[done] Processed {n} frames" + (f" at ~{fps:.1f} FPS" if fps else ""))


def classify_source(s: str):
    if s.lower() == "screen" or s.lower().startswith("screen:"):
        return "screen", s[7:] if ":" in s else ""
    if s.isdigit():
        return "webcam", int(s)
    if s.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return "video", s
    p = Path(s).expanduser()
    if p.is_dir():
        return "folder", p
    if p.is_file():
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            return "image", p
        if ext in VIDEO_EXTS:
            return "video", str(p)
        sys.exit(f"[error] Unsupported file type '{ext}'.")
    sys.exit(f"[error] Source not found: {s}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="Webcam index, image, folder, video file, stream URL, or screen / screen:2 / screen:left,top,w,h (default 0)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=0, help="Inference size; 0 = size the model was trained at "
                                                          "(try 640 for tiny/far faces, 320-416 for speed)")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | mps")
    ap.add_argument("--save", action="store_true", help="Save annotated output")
    ap.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "detections"))
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror the webcam image")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    args = ap.parse_args(argv)

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"[error] Weights not found: {weights}\n"
                 "  -> Train first:  python prepare_dataset.py  then  python train.py\n"
                 "  -> or pass your own fine-tuned model:  --weights path/to/best.pt")

    kind, src = classify_source(args.source)
    det = Detector(weights, args.device, args.imgsz, args.conf, args.iou)
    print(f"[info] Model loaded ({weights.name}) on device '{det.device}', classes: {list(det.names.values())}")
    out_dir = Path(args.out)
    display = Display(enabled=not args.no_show, small=(kind == "screen"))
    try:
        if kind in ("image", "folder"):
            paths = [src] if kind == "image" else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not paths:
                sys.exit(f"[error] No images found in {src}")
            # always save when there is no window to look at, or when processing a whole folder
            run_images(det, paths, out_dir, display, args.save or not display.enabled or len(paths) > 1, args)
        elif kind == "screen":
            run_stream(det, ScreenGrabber(src), out_dir, display, args.save, args, is_webcam=False, label="screen")
        else:
            run_stream(det, src, out_dir, display, args.save, args, is_webcam=(kind == "webcam"))
    finally:
        display.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
