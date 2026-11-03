"""Streamlit web frontend for the face-mask detector."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from mask_detection import MaskDetector, draw_detections


st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
)


def _write_upload(uploaded_file, suffix: str) -> Path:
    file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = uploaded_file if isinstance(uploaded_file, bytes) else uploaded_file.getbuffer()
    file.write(content)
    file.close()
    return Path(file.name)


def _detector(weights_file, config_file, labels_text: str, confidence: float, nms: float, size: int):
    labels = [line.strip() for line in labels_text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not labels:
        raise ValueError("Enter at least one class label.")
    weights_path = _write_upload(weights_file, Path(weights_file.name).suffix)
    config_path = (
        _write_upload(config_file, Path(config_file.name).suffix)
        if config_file is not None
        else None
    )
    return MaskDetector(weights_path, config_path, labels, confidence, nms, (size, size))


def _render_image(detector: MaskDetector, image_bytes: bytes) -> tuple[np.ndarray, list]:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded image could not be decoded.")
    detections = detector.detect(image)
    return cv2.cvtColor(draw_detections(image, detections), cv2.COLOR_BGR2RGB), detections


def _render_video(detector: MaskDetector, video_bytes: bytes, suffix: str) -> Path:
    input_path = _write_upload(video_bytes, suffix)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rendered = draw_detections(frame, detector.detect(frame))
            if writer is None:
                fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                size = (rendered.shape[1], rendered.shape[0])
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
                )
            writer.write(rendered)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if writer is None:
        raise ValueError("The uploaded video contained no readable frames.")
    return output_path


st.title("😷 Face Mask Detection")
st.caption("Upload a photo, camera snapshot, or video and run your YOLO model in the browser.")

with st.sidebar:
    st.header("Model")
    weights = st.file_uploader("YOLO weights or ONNX model", type=["weights", "onnx", "bin"])
    config = st.file_uploader("Darknet config (optional)", type=["cfg", "txt"])
    labels = st.text_area(
        "Class labels (one per line)",
        value="with_mask\nwithout_mask\nmask_weared_incorrect",
        height=110,
    )
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05)
    nms = st.slider("NMS threshold", 0.05, 0.95, 0.40, 0.05)
    input_size = st.selectbox("Input size", [320, 416, 512, 640], index=1)

if weights is None:
    st.info("Upload your trained model in the sidebar to begin.")
    st.markdown(
        "For Darknet models, upload both `.weights` and `.cfg`. "
        "For ONNX models, upload only the `.onnx` file."
    )
    st.stop()

try:
    detector = _detector(weights, config, labels, confidence, nms, input_size)
except (FileNotFoundError, ValueError, cv2.error) as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

tab_image, tab_video = st.tabs(["Image / camera", "Video"])
with tab_image:
    image_upload = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )
    camera_upload = st.camera_input("Or take a picture")
    image_file = image_upload or camera_upload
    if image_file is not None:
        try:
            rendered, detections = _render_image(detector, image_file.getvalue())
            st.image(rendered, use_container_width=True)
            st.success(f"Detected {len(detections)} object(s).")
            if detections:
                st.dataframe(
                    [
                        {"Class": item.label, "Confidence": f"{item.confidence:.1%}"}
                        for item in detections
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
        except (ValueError, cv2.error) as error:
            st.error(f"Could not process the image: {error}")

with tab_video:
    video_upload = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )
    if video_upload is not None and st.button("Process video", type="primary"):
        with st.spinner("Processing video..."):
            try:
                result_path = _render_video(
                    detector, video_upload.getvalue(), Path(video_upload.name).suffix
                )
                st.video(result_path.read_bytes())
                st.success("Video processed successfully.")
            except (ValueError, cv2.error) as error:
                st.error(f"Could not process the video: {error}")
