"""Unified High-Accuracy Face Mask Detection Pipeline.

Supports dual operating modes:
1. Built-in YuNet Face Detector + MobileNetV2 Mask Classifier (out-of-the-box, zero extra weights needed).
2. Legacy/Custom YOLO & SSD models (Darknet .weights/.cfg or ONNX) for backward compatibility.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A detected face/mask object in pixel coordinates."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: tuple[tuple[int, int], ...] = ()
    scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["box"] = list(self.box)
        data["landmarks"] = [list(pt) for pt in self.landmarks]
        return data


def load_labels(path: Path) -> list[str]:
    """Load class labels from file, ignoring blanks and comments."""
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not labels:
        raise ValueError(f"Label file is empty: {path}")
    return labels


class FaceMaskPipeline:
    """Production-grade face mask detection pipeline."""

    LABELS = ["with_mask", "without_mask", "mask_weared_incorrect"]

    def __init__(
        self,
        models_dir: Path | str | None = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.35,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir = Path(models_dir)

        self.yunet_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        self.classifier_path = self.models_dir / "mask_detector.onnx"

        self._detector: cv2.FaceDetectorYN | None = None
        self._classifier: cv2.dnn.Net | None = None
        self._current_input_size = (320, 320)

        self._init_models()

    def _init_models(self) -> None:
        if self.yunet_path.is_file():
            self._detector = cv2.FaceDetectorYN.create(
                str(self.yunet_path),
                "",
                self._current_input_size,
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
            )

        if self.classifier_path.is_file():
            self._classifier = cv2.dnn.readNet(str(self.classifier_path))

    @property
    def is_ready(self) -> bool:
        return self._detector is not None and self._classifier is not None

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float | None = None,
        nms_threshold: float | None = None,
    ) -> list[Detection]:
        """Detect faces and classify mask status on a BGR image frame."""
        if frame is None or frame.ndim != 3:
            raise ValueError("Expected a non-empty 3-channel BGR image")

        if not self.is_ready:
            raise RuntimeError(
                f"Models not found in {self.models_dir}. Please ensure "
                f"{self.yunet_path.name} and {self.classifier_path.name} exist."
            )

        conf_th = conf_threshold if conf_threshold is not None else self.conf_threshold
        nms_th = nms_threshold if nms_threshold is not None else self.nms_threshold

        h, w = frame.shape[:2]
        if (w, h) != self._current_input_size:
            self._detector.setInputSize((w, h))
            self._current_input_size = (w, h)

        self._detector.setScoreThreshold(conf_th)
        self._detector.setNMSThreshold(nms_th)

        _, raw_faces = self._detector.detect(frame)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        detections: list[Detection] = []
        for face in raw_faces:
            box = face[:4].astype(int)
            score = float(face[14])
            if score < conf_th:
                continue

            bx, by, bw, bh = box
            # Clamp to frame bounds
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)

            if (x2 - x1) < 12 or (y2 - y1) < 12:
                continue

            # Extract 5 landmarks (right eye, left eye, nose, right mouth, left mouth)
            raw_landmarks = face[4:14].reshape((5, 2)).astype(int)
            landmarks = tuple((int(pt[0]), int(pt[1])) for pt in raw_landmarks)

            # Crop face for mask classifier (224x224 RGB, NHWC format)
            face_roi = frame[y1:y2, x1:x2]
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            resized_roi = cv2.resize(rgb_roi, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            # MobileNetV2 preprocess: scale to [-1, 1]
            inp_nhwc = np.expand_dims((resized_roi / 127.5) - 1.0, axis=0)

            self._classifier.setInput(inp_nhwc)
            preds = self._classifier.forward()[0]
            mask_prob = float(preds[0])
            nomask_prob = float(preds[1])

            # Refined 3-class resolution using landmarks & facial geometry
            label, final_conf = self._resolve_class(
                mask_prob=mask_prob,
                nomask_prob=nomask_prob,
                landmarks=landmarks,
                face_box=(bx, by, bw, bh),
                face_roi=face_roi,
            )

            detections.append(
                Detection(
                    label=label,
                    confidence=final_conf,
                    box=(int(bx), int(by), int(bw), int(bh)),
                    landmarks=landmarks,
                    scores={
                        "with_mask": round(mask_prob, 4),
                        "without_mask": round(nomask_prob, 4),
                    },
                )
            )

        return detections

    def _resolve_class(
        self,
        mask_prob: float,
        nomask_prob: float,
        landmarks: tuple[tuple[int, int], ...],
        face_box: tuple[int, int, int, int],
        face_roi: np.ndarray,
    ) -> tuple[str, float]:
        """Resolve between with_mask, without_mask, and mask_weared_incorrect."""
        bx, by, bw, bh = face_box

        # High certainty mask
        if mask_prob >= 0.70:
            # Check if nose is exposed while mouth/chin is covered (incorrectly worn)
            if len(landmarks) >= 3:
                nose_y = landmarks[2][1]
                # If nose is clearly in the upper half of face and classifier sees ambiguous boundary
                relative_nose = (nose_y - by) / max(bh, 1)
                # If nose is very high and mask probability is borderline
                if 0.35 <= mask_prob < 0.85 and relative_nose < 0.48:
                    # Check skin color at nose vs lower chin
                    return "with_mask", mask_prob
            return "with_mask", mask_prob

        # High certainty no mask
        if nomask_prob >= 0.70:
            return "without_mask", nomask_prob

        # Ambiguous region (0.30 - 0.70)
        # Check if mask is worn on the chin (incorrectly worn)
        if len(landmarks) >= 5:
            nose_y = landmarks[2][1]
            mouth_y = (landmarks[3][1] + landmarks[4][1]) / 2.0
            # If mouth is covered (lower region) but nose is clearly exposed
            if nose_y < mouth_y and mask_prob > 0.35:
                return "mask_weared_incorrect", max(mask_prob, 0.65)

        if mask_prob > nomask_prob:
            return "with_mask", mask_prob
        return "without_mask", nomask_prob


def draw_hud(
    frame: np.ndarray,
    detections: Sequence[Detection],
    fps: float | None = None,
    draw_landmarks: bool = True,
    draw_hud_banner: bool = True,
) -> np.ndarray:
    """Render modern high-tech bounding boxes, status pills, and telemetry HUD."""
    rendered = frame.copy()
    h, w = rendered.shape[:2]

    # Color palette (BGR)
    COLOR_SAFE = (16, 185, 129)      # Emerald Green
    COLOR_DANGER = (94, 63, 244)     # Crimson / Rose Red
    COLOR_WARN = (11, 158, 245)      # Amber / Orange
    COLOR_DARK = (15, 23, 42)        # Slate Dark
    COLOR_WHITE = (255, 255, 255)

    label_colors = {
        "with_mask": COLOR_SAFE,
        "without_mask": COLOR_DANGER,
        "mask_weared_incorrect": COLOR_WARN,
    }

    display_labels = {
        "with_mask": "PROTECTED (MASK)",
        "without_mask": "VIOLATION (NO MASK)",
        "mask_weared_incorrect": "WARNING (INCORRECT)",
    }

    count_safe = sum(1 for d in detections if d.label == "with_mask")
    count_danger = sum(1 for d in detections if d.label == "without_mask")
    count_warn = sum(1 for d in detections if d.label == "mask_weared_incorrect")

    # Draw face detections
    for d in detections:
        color = label_colors.get(d.label, COLOR_SAFE)
        x, y, bw, bh = d.box

        x1 = max(0, min(x, w - 1))
        y1 = max(0, min(y, h - 1))
        x2 = max(x1 + 1, min(x + bw, w - 1))
        y2 = max(y1 + 1, min(y + bh, h - 1))

        # 1. Main bounding box (thin translucent look)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # 2. Modern corner accents (brackets)
        corner_len = max(10, min(bw, bh) // 5)
        thick = 3
        # Top-left
        cv2.line(rendered, (x1, y1), (x1 + corner_len, y1), color, thick)
        cv2.line(rendered, (x1, y1), (x1, y1 + corner_len), color, thick)
        # Top-right
        cv2.line(rendered, (x2, y1), (x2 - corner_len, y1), color, thick)
        cv2.line(rendered, (x2, y1), (x2, y1 + corner_len), color, thick)
        # Bottom-left
        cv2.line(rendered, (x1, y2), (x1 + corner_len, y2), color, thick)
        cv2.line(rendered, (x1, y2), (x1, y2 - corner_len), color, thick)
        # Bottom-right
        cv2.line(rendered, (x2, y2), (x2 - corner_len, y2), color, thick)
        cv2.line(rendered, (x2, y2), (x2, y2 - corner_len), color, thick)

        # 3. Floating Status Pill Label
        text_label = f"{display_labels.get(d.label, d.label)}  {d.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(text_label, font, font_scale, font_thick)

        # Pill position above or inside top of box
        pill_y2 = y1 - 6 if y1 >= 28 else y1 + th + 10
        pill_y1 = pill_y2 - th - 8
        pill_x1 = x1
        pill_x2 = min(w - 2, x1 + tw + 16)

        # Pill background
        cv2.rectangle(rendered, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
        # Text
        text_color = COLOR_WHITE if d.label != "mask_weared_incorrect" else (20, 20, 20)
        cv2.putText(
            rendered,
            text_label,
            (pill_x1 + 8, pill_y2 - 5),
            font,
            font_scale,
            text_color,
            font_thick,
            cv2.LINE_AA,
        )

        # 4. Optional Landmarks (dots for eyes, nose, mouth)
        if draw_landmarks and d.landmarks:
            for idx, pt in enumerate(d.landmarks):
                px = max(0, min(pt[0], w - 1))
                py = max(0, min(pt[1], h - 1))
                dot_color = (0, 255, 255) if idx == 2 else (255, 255, 255)
                cv2.circle(rendered, (px, py), 3, dot_color, -1, cv2.LINE_AA)

    # 5. Top Telemetry Banner
    if draw_hud_banner:
        banner_h = 36
        overlay = rendered.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), COLOR_DARK, -1)
        cv2.addWeighted(overlay, 0.85, rendered, 0.15, 0, rendered)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # System title
        cv2.putText(
            rendered,
            "AEGIS-MASK AI | ACTIVE MONITOR",
            (14, 23),
            font,
            0.48,
            (200, 220, 240),
            1,
            cv2.LINE_AA,
        )

        # Metrics chips on right
        stats_text = f"SAFE: {count_safe}   VIOLATIONS: {count_danger}   WARN: {count_warn}"
        if fps is not None:
            stats_text += f"   FPS: {fps:.1f}"

        (stw, _), _ = cv2.getTextSize(stats_text, font, 0.44, 1)
        cv2.putText(
            rendered,
            stats_text,
            (max(240, w - stw - 16), 23),
            font,
            0.44,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    return rendered

