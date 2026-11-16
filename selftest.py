"""End-to-end self-test: checks that your installation works WITHOUT downloading the real dataset.

It builds a tiny synthetic Pascal-VOC dataset, converts it, fine-tunes yolo11n for 2 epochs,
runs detection on an image and a video, and prints PASS/FAIL.
(Needs internet once to fetch the small pretrained yolo11n.pt.)

    python selftest.py
"""
import random
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

from common import ROOT

WORK = ROOT / ".selftest"
CLASSES = ["with_mask", "mask_weared_incorrect", "without_mask"]


def draw_face(img, cx, cy, r, kind, rng):
    skin = (int(150 + rng.random() * 40), int(170 + rng.random() * 40), int(210 + rng.random() * 30))
    cv2.circle(img, (cx, cy), r, skin, -1)
    for dx in (-r // 3, r // 3):
        cv2.circle(img, (cx + dx, cy - r // 5), max(2, r // 9), (30, 30, 30), -1)
    if kind == "with_mask":
        cv2.rectangle(img, (cx - r, cy), (cx + r, cy + r), (230, 200, 60), -1)
    elif kind == "mask_weared_incorrect":
        cv2.rectangle(img, (cx - r, cy + r // 2), (cx + r, cy + r), (230, 200, 60), -1)
    else:
        cv2.ellipse(img, (cx, cy + r // 2), (r // 3, r // 8), 0, 0, 180, (60, 60, 200), 2)


def make_voc(dst: Path, n=60, seed=0):
    rng = random.Random(seed)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "annotations").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        size = 320
        img = np.full((size, size, 3), 200, np.uint8)
        img[:] = [rng.randint(120, 230)] * 3
        objs = []
        for _ in range(rng.randint(1, 3)):
            r = rng.randint(30, 55)
            cx, cy = rng.randint(r + 5, size - r - 5), rng.randint(r + 5, size - r - 5)
            kind = rng.choice(CLASSES)
            draw_face(img, cx, cy, r, kind, rng)
            objs.append((kind, cx - r, cy - r, cx + r, cy + r))
        name = f"synthetic{i}"
        cv2.imwrite(str(dst / "images" / f"{name}.png"), img)
        xml = ["<annotation>", f"<filename>{name}.png</filename>",
               f"<size><width>{size}</width><height>{size}</height><depth>3</depth></size>"]
        for k, x1, y1, x2, y2 in objs:
            xml.append(f"<object><name>{k}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                       f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>")
        xml.append("</annotation>")
        (dst / "annotations" / f"{name}.xml").write_text("".join(xml))


def make_video(path: Path, frames=20):
    rng = random.Random(1)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 320))
    for i in range(frames):
        img = np.full((320, 320, 3), 180, np.uint8)
        draw_face(img, 100 + i * 5, 160, 45, "with_mask", rng)
        vw.write(img)
    vw.release()


def step(title):
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    import detect
    import prepare_dataset
    import train

    if WORK.exists():
        shutil.rmtree(WORK)
    try:
        step("1/4 build synthetic VOC dataset")
        make_voc(WORK / "voc")
        make_video(WORK / "clip.mp4")

        step("2/4 convert to YOLO format")
        ds = WORK / "yolo"
        assert prepare_dataset.main(["--source", str(WORK / "voc"), "--out", str(ds)]) == 0
        assert (ds / "data.yaml").exists()

        step("3/4 fine-tune 2 epochs (smoke test)")
        best = WORK / "best.pt"
        assert train.main(["--data", str(ds / "data.yaml"), "--epochs", "2", "--imgsz", "320", "--batch", "8",
                           "--workers", "0", "--name", "selftest", "--export-to", str(best)]) == 0
        assert best.exists()

        step("4/4 run detection on image + video")
        img = next((ds / "images" / "val").glob("*.png"))
        out = WORK / "out"
        assert detect.main(["--source", str(img), "--weights", str(best), "--imgsz", "320", "--no-show",
                            "--conf", "0.05", "--out", str(out)]) == 0
        assert detect.main(["--source", str(WORK / "clip.mp4"), "--weights", str(best), "--imgsz", "320",
                            "--no-show", "--save", "--conf", "0.05", "--out", str(out)]) == 0
        outputs = [p.name for p in out.iterdir()]
        assert any(n.endswith("_pred.png") for n in outputs), outputs
        assert any(n.endswith(".mp4") for n in outputs), outputs
        print("\nPASS  - training, conversion and detection all work on this machine.")
        return 0
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            return 0
        traceback.print_exc()
        print("\nFAIL  - see the error above.")
        return 1
    finally:
        # keep the folder small
        shutil.rmtree(WORK / "yolo", ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
