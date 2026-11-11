"""Real-time face-mask detection using OpenCV DNN and deep neural networks.

Supports:
- Out-of-the-box built-in high-accuracy detector (YuNet + MobileNetV2 ONNX).
- Custom/Darknet/ONNX YOLO models via --weights and --config.
- Modern Web Dashboard via --web.
- Image, video, and live webcam sources.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from core.detector import Detection, FaceMaskPipeline, draw_hud, load_labels

__all__ = [
    "Detection",
    "FaceMaskPipeline",
    "MaskDetector",
    "draw_detections",
    "draw_hud",
    "load_labels",
    "parse_yolo_outputs",
]


def _normalise_outputs(outputs: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(output) for output in outputs]
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    flattened = []
    for output in arrays:
        if output.ndim in (3, 4):
            output = output.reshape(-1, output.shape[-1])
        elif output.ndim == 1:
            output = output.reshape(1, -1)
        flattened.append(output)
    return np.concatenate(flattened, axis=0)


def parse_yolo_outputs(
    outputs: Iterable[np.ndarray],
    frame_shape: Sequence[int],
    labels: Sequence[str],
    confidence_threshold: float,
    nms_threshold: float,
) -> list[Detection]:
    """Convert YOLO output tensors into detections using OpenCV NMS."""
    height, width = frame_shape[:2]
    candidates: list[tuple[str, float, tuple[int, int, int, int]]] = []
    rows = _normalise_outputs(outputs)

    for row in rows:
        if row.size < 6:
            continue
        objectness = float(row[4])
        class_scores = row[5:]
        if not class_scores.size:
            continue
        class_id = int(np.argmax(class_scores))
        confidence = objectness * float(class_scores[class_id])
        if confidence < confidence_threshold or class_id >= len(labels):
            continue

        center_x, center_y, box_width, box_height = row[:4]
        x = round((float(center_x) - float(box_width) / 2) * width)
        y = round((float(center_y) - float(box_height) / 2) * height)
        w = round(float(box_width) * width)
        h = round(float(box_height) * height)
        candidates.append((labels[class_id], confidence, (x, y, w, h)))

    if not candidates:
        return []

    boxes = [candidate[2] for candidate in candidates]
    scores = [candidate[1] for candidate in candidates]
    kept = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, nms_threshold)
    indices = np.asarray(kept).reshape(-1).tolist() if len(kept) else []
    return [Detection(*candidates[index]) for index in indices]


class MaskDetector:
    """Run a YOLO-compatible OpenCV DNN model on image frames."""

    def __init__(
        self,
        weights: Path,
        config: Path | None,
        labels: Sequence[str],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (416, 416),
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        if config is not None and not config.is_file():
            raise FileNotFoundError(f"Model config not found: {config}")
        if not labels:
            raise ValueError("At least one class label is required")
        if not 0 < confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if not 0 < nms_threshold <= 1:
            raise ValueError("NMS threshold must be between 0 and 1")

        self.labels = list(labels)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.network = (
            cv2.dnn.readNet(str(weights), str(config))
            if config is not None
            else cv2.dnn.readNet(str(weights))
        )
        self.output_layers = self._output_layers()

    def _output_layers(self) -> list[str]:
        names = self.network.getUnconnectedOutLayersNames()
        if names:
            return list(names)
        layer_names = self.network.getLayerNames()
        indices = self.network.getUnconnectedOutLayers()
        return [layer_names[int(index) - 1] for index in np.asarray(indices).reshape(-1)]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for one BGR frame."""
        if frame is None or frame.ndim != 3:
            raise ValueError("Expected a non-empty BGR image")
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, self.input_size, swapRB=True, crop=False
        )
        self.network.setInput(blob)
        outputs = self.network.forward(self.output_layers)
        return parse_yolo_outputs(
            outputs,
            frame.shape,
            self.labels,
            self.confidence_threshold,
            self.nms_threshold,
        )


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Draw detections on a copy of a frame."""
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    for detection in detections:
        x, y, box_width, box_height = detection.box
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        right = max(x + 1, min(x + box_width, width - 1))
        bottom = max(y + 1, min(y + box_height, height - 1))

        lbl = detection.label.lower()
        if "without" in lbl or "no_mask" in lbl:
            color = (0, 0, 220)       # Red
        elif "incorrect" in lbl or "warn" in lbl:
            color = (0, 165, 255)     # Amber/Orange
        else:
            color = (0, 190, 0)       # Green

        cv2.rectangle(rendered, (x, y), (right, bottom), color, 2)
        caption = f"{detection.label}: {detection.confidence:.0%}"
        baseline = max(20, y)
        cv2.rectangle(
            rendered,
            (x, baseline - 22),
            (x + 8 + len(caption) * 9, baseline),
            color,
            -1,
        )
        cv2.putText(
            rendered,
            caption,
            (x + 4, baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional custom YOLO weights/model file (if omitted, uses built-in high-accuracy models)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YOLO config file, required for Darknet models",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Text file containing class names (default: built-in classes)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to an image/video file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save processed image/video to this path",
    )
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--size", type=int, default=416, help="Square input size for YOLO model")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the interactive web dashboard and API",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web dashboard host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Web dashboard port (default 8000)")
    parser.add_argument(
        "--classic-draw",
        action="store_true",
        help="Use simple classic bounding box renderer instead of modern HUD",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.web:
        print(f"[*] Starting AegisMask Web Dashboard on http://{args.host}:{args.port}")
        import uvicorn
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
        return 0

    # Determine detector: custom YOLO or built-in pipeline
    if args.weights is not None:
        if args.labels is None:
            default_labels = Path(__file__).resolve().parent / "labels.example.txt"
            labels = load_labels(default_labels) if default_labels.is_file() else ["with_mask", "without_mask"]
        else:
            labels = load_labels(args.labels)
        detector = MaskDetector(
            args.weights,
            args.config,
            labels,
            args.confidence,
            args.nms,
            (args.size, args.size),
        )
    else:
        pipeline = FaceMaskPipeline(
            conf_threshold=args.confidence,
            nms_threshold=args.nms,
        )
        detector = pipeline

    source = _source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    is_image = isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }
    writer = None
    if is_image:
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {source}")
        frames = [frame]
    else:
        frames = iter(lambda: capture.read()[1], None)

    started = time.monotonic()
    processed = 0
    render_fn = draw_detections if args.classic_draw else draw_hud

    try:
        for frame in frames:
            if frame is None:
                break
            detections = detector.detect(frame)
            processed += 1
            fps = processed / max(time.monotonic() - started, 1e-6) if not is_image else None

            if args.classic_draw:
                result = draw_detections(frame, detections)
                if not is_image:
                    cv2.putText(
                        result,
                        f"FPS: {fps:.1f}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
            else:
                result = draw_hud(frame, detections, fps=fps)

            if args.output and writer is None and not is_image:
                video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                size = (result.shape[1], result.shape[0])
                args.output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video_fps,
                    size,
                )
            if writer is not None:
                writer.write(result)
            elif args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.output), result):
                    raise RuntimeError(f"Unable to write output: {args.output}")

            if not args.no_display:
                cv2.imshow("AegisMask AI - Face Mask Detection", result)
                if cv2.waitKey(1 if not is_image else 0) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = _build_parser()
    try:
        return run(parser.parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
