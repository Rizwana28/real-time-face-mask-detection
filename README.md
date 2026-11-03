# Real-time face-mask detection

A lightweight face-mask detector for webcam, image, and video input. It uses
OpenCV's DNN module and accepts any YOLO model whose output rows follow the
standard format:

`center_x center_y width height objectness class_1_score ... class_n_score`

Coordinates must be normalized to the input image. Model weights are deliberately
not committed; this avoids storing large binaries and lets you use your own
trained YOLO/SSD-compatible model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Create a labels file with one class per line. An example is included:

```text
with_mask
without_mask
mask_weared_incorrect
```

## Run

With Darknet YOLO files (`.weights` and `.cfg`):

```bash
python mask_detection.py \
  --weights models/mask-yolo.weights \
  --config models/mask-yolo.cfg \
  --labels labels.example.txt
```

Use an image or video instead of the default webcam (`0`):

```bash
python mask_detection.py --weights models/mask-yolo.weights \
  --config models/mask-yolo.cfg --labels labels.example.txt \
  --source input.mp4 --output output/result.mp4

python mask_detection.py --weights models/mask-yolo.onnx \
  --labels labels.example.txt --source photo.jpg --output output/photo.jpg \
  --no-display
```

Press `q` to stop a live preview. Adjust `--confidence`, `--nms`, and `--size`
for the model and hardware. For an ONNX model, omit `--config`; for Darknet
models, provide the matching `.cfg`.

## Test

```bash
python -m pytest
```

The tests cover output decoding, confidence filtering, non-maximum suppression,
and safe rendering. No camera, model weights, or GUI is required to run them.

## Project layout

| Path | Purpose |
| --- | --- |
| `mask_detection.py` | CLI, model loading, inference, rendering |
| `labels.example.txt` | Example class names |
| `tests/` | Unit tests for inference utilities |
| `requirements.txt` | Runtime dependencies |
