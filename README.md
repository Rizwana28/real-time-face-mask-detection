# Real-Time Face Mask Detection (YOLO11, fine-tuned)

Built with Ultralytics YOLO11, trained on the Kaggle Face Mask Detection dataset.

Detects whether people in **images, folders, videos and live webcam feeds** wear a mask correctly.

| Class | Colour | Meaning |
|---|---|---|
| `with_mask` | green | mask worn correctly |
| `mask_weared_incorrect` | orange | mask on, but under nose / on chin |
| `without_mask` | red | no mask |

**Model:** Ultralytics YOLO11n (COCO-pretrained) fine-tuned on the Kaggle *Face Mask Detection* dataset
(853 images, 4 072 labelled faces). YOLO11n is small and fast: real-time on a GPU, ~10-20 FPS on a normal laptop CPU.
Use `--model yolo11s.pt` in `train.py` for higher accuracy.

---

## QUICK START (trained model is already included)

`weights/best.pt` is already fine-tuned on the face-mask dataset, so you can detect immediately:

1. Install Python 3.10-3.13 (tick **Add python.exe to PATH**). Unzip this folder to **D: or E:** (not C:).
2. Double-click **setup.bat** (one-time, installs PyTorch + OpenCV + Ultralytics).
3. Double-click **web.bat** for the browser app (webcam, screen, video, photo), **webcam.bat** for a desktop webcam window, **screen.bat** to detect on your screen, or **demo.bat** for the sample photos.
   Or run: `python detect.py --source your_photo.jpg`

Annotated sample results are already in `outputs/demo/`.

**Included model results** (YOLO11n fine-tuned 20 epochs, held-out test split, 85 images):

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| with_mask | 0.96 | 0.90 | 0.95 |
| mask_weared_incorrect | 0.79 | 0.43 | 0.49 |
| without_mask | 0.84 | 0.57 | 0.68 |
| **All** | **0.86** | **0.63** | **0.71** |

The "incorrect mask" class has only ~120 training examples in this dataset, so it is the weakest.
For better accuracy, retrain longer / on a GPU (sections 3-4 below) and tune `--conf` (lower = catches more faces, higher = fewer false alarms).
Training curves and confusion matrix: `outputs/train/face_mask/`.

---

## WEB APP (MaskSense): webcam, screen, video and photo in your browser

`app.py` starts a polished browser app. The **browser** captures the camera / screen, so it works with a laptop's
built-in camera, a USB camera on a PC, or a phone, and it can be hosted online for a live link.

**Run it:** double-click **web.bat** (or `python app.py`). Your browser opens http://127.0.0.1:7860 with four tabs:

| Tab | What it does |
|---|---|
| **Webcam** | Click *Start camera*, allow access. Pick another camera in the drop-down. Mirror, labels and alert sound are in Settings. |
| **Screen** | Click *Share screen* and pick a window, tab or whole screen (Zoom call, YouTube...). Desktop Chrome / Edge / Firefox. |
| **Video** | Choose or drag in a video file: detection runs live as it plays (play, pause, seek). *Export annotated video* makes a downloadable MP4 with the boxes burned in. |
| **Photo** | Choose or drag in a photo. Moving the confidence slider updates the result instantly. |

The right-hand panel shows the % of faces masked correctly, counts per class, an alert when someone has no mask or wears it
wrongly, latency and FPS. The camera icon saves a PNG snapshot.

**Screen capture without a browser (desktop window):** double-click **screen.bat**, or
`python detect.py --source screen` (`screen:2` = second monitor, `screen:100,100,800,600` = only that region).
A small always-on-top preview window shows the result; press **q** in it to quit. Tip: use a second monitor or a region so the
preview does not cover what you capture.

**Permanent live link (Hugging Face Spaces, free):**
- Easiest: `python deploy_hf.py --space YOUR_USERNAME/face-mask-detection --token hf_xxxxx`
  (token: huggingface.co -> Settings -> Access Tokens -> New token, type *Write*).
- Or manually: create a new Space with SDK **Docker** and upload everything inside the `hf_space/` folder.
Open the app at `https://YOUR_USERNAME-face-mask-detection.hf.space` (the direct address is best for camera and screen sharing).
The first build takes several minutes. Free Spaces run on CPU, so live video is a few frames per second; use a GPU Space for smooth real-time.
Other hosts (Render, Railway, a VM) work too: run `python app.py --host 0.0.0.0` behind https (browsers only allow the camera on https or localhost).

---

## 1. Setup details (Windows)  - keep this folder on D: or E:, NOT C: (C: is almost full)

1. Install Python 3.10-3.13 from python.org (tick **Add python.exe to PATH**).
2. Unzip this project to e.g. `D:\face_mask_detection`.
3. Double-click **setup.bat**. (Creates `.venv`, installs PyTorch + dependencies. It auto-picks CUDA if you have an NVIDIA GPU. Temp files stay inside the folder.)

Linux / macOS: `bash setup.sh`, then `source .venv/bin/activate`.

## 2. Check your installation (optional, ~5 min)

```
.venv\Scripts\activate
python selftest.py
```
Prints `PASS` if conversion, training and detection all work on your machine (uses synthetic data, needs internet once for `yolo11n.pt`).

## 3. Retraining (optional): get the dataset

Option A (automatic): `python prepare_dataset.py` - downloads through `kagglehub`.
Needs a free Kaggle account; if it asks for credentials, create an API token at kaggle.com -> Settings -> API and set `KAGGLE_USERNAME` / `KAGGLE_KEY`.

Option B (manual, always works):
1. Open https://www.kaggle.com/datasets/andrewmvd/face-mask-detection and click **Download**.
2. `python prepare_dataset.py --source "D:\Downloads\archive.zip"`

This converts the Pascal-VOC XML labels to YOLO format and splits train / val / test (75 / 15 / 10).

## 4. Retraining (optional): fine-tune

```
python train.py                      # yolo11n, 40 epochs, early stopping
python train.py --model yolo11s.pt --epochs 60
python train.py --resume             # continue an interrupted run
```
Best weights are copied to `weights/best.pt`.

**No GPU?** Train free on Google Colab (Runtime -> GPU): upload this folder, run `pip install ultralytics`,
`python prepare_dataset.py --source archive.zip`, `python train.py`, then download `weights/best.pt` into your local `weights/` folder.
Training 40 epochs on a GPU takes about 10 minutes; on CPU expect 1-3 hours.

## 5. Evaluate

```
python evaluate.py          # precision / recall / mAP per class on the test split
```

## 6. Run detection

```
python detect.py --source 0                   # webcam  (q/ESC quit, s snapshot, p pause)
python detect.py --source 0 --save            # webcam + record to outputs/detections/
python detect.py --source photo.jpg           # one image
python detect.py --source my_images/          # folder -> annotated images + results.csv
python detect.py --source clip.mp4 --save     # video file
python detect.py --source rtsp://...          # IP camera / stream
python detect.py --source screen              # live on your screen (screen:2, screen:x,y,w,h)
```
Detection size defaults to the size the model was trained at (512). Speed-ups on slow PCs: `--imgsz 416` (or 320); for small / distant faces try `--imgsz 640`. Fewer false alarms: `--conf 0.5`.
Double-click **webcam.bat** for one-click webcam mode, **demo.bat** for the sample photos, or **quickstart.bat** (uses the included model; trains only if `weights/best.pt` is missing).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open source '0'` | Close Zoom/Teams/browser tabs using the camera, try `--source 1`, check Windows Privacy -> Camera |
| No preview window / "no GUI support" | `pip uninstall opencv-python-headless -y` then `pip install opencv-python` |
| Out of memory while training | `--batch 4` or `--imgsz 480` |
| Low FPS on CPU | `--imgsz 320`, or use a GPU |
| Disk full error | Keep project on D:/E:. Cache folders are already redirected inside the project |
| Kaggle download fails | Use Option B above |

## Files

`prepare_dataset.py` VOC->YOLO converter | `train.py` fine-tuning | `evaluate.py` metrics | `detect.py` image/video/webcam inference |
`app.py` + `web/` + `web.bat` browser app | `screen.bat` screen capture | `deploy_hf.py` + `hf_space/` live-link deployment | `demo.bat` sample demo | `selftest.py` installation check | `common.py` shared config | `setup.*`, `quickstart.bat`, `webcam.bat` helpers

Note: this is a computer-vision demo, not a medical or safety-certified system. Accuracy depends on lighting, camera angle and the dataset's limits.

## License and credits

Project code: MIT License (see `LICENSE`).
Third-party parts keep their own licences: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) is AGPL-3.0 (the trained `weights/best.pt` is derived from it),
and the dataset is the Kaggle "Face Mask Detection" set by andrewmvd. Check these terms before any commercial or closed-source use.
