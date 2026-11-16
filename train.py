"""Fine-tune a COCO-pretrained YOLO model on the face-mask dataset.

Usage:
    python train.py                       # sensible defaults (yolo11n, 40 epochs)
    python train.py --epochs 60 --imgsz 640 --model yolo11s.pt
    python train.py --resume              # continue an interrupted run

The best checkpoint is copied to  weights/best.pt  (used by detect.py).
"""
import argparse
import shutil
import sys
from pathlib import Path

from common import DEFAULT_DATA_YAML, DEFAULT_WEIGHTS, ROOT, is_cuda, pick_device


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA_YAML))
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Pretrained checkpoint: yolo11n.pt (fast) | yolo11s.pt (more accurate) | yolov8n.pt ...")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto (16 on GPU, 8 on CPU)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=15, help="Early-stopping patience (epochs)")
    ap.add_argument("--cache", action="store_true", help="Cache images in RAM (faster, needs memory)")
    ap.add_argument("--name", default="face_mask")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export-to", default=str(DEFAULT_WEIGHTS), help="Where to copy the best weights")
    args = ap.parse_args(argv)

    data = Path(args.data)
    if not data.exists():
        sys.exit(f"[error] {data} not found. Run  python prepare_dataset.py  first.")

    from ultralytics import YOLO

    device = pick_device(args.device)
    batch = args.batch or (16 if is_cuda(device) else 8)
    print(f"[info] device={device} batch={batch} epochs={args.epochs} imgsz={args.imgsz} model={args.model}")
    if device == "cpu":
        print("[warn] Training on CPU is slow (hours). A GPU or free Google Colab GPU is much faster; "
              "copy the resulting best.pt into the weights/ folder.")

    project = ROOT / "outputs" / "train"
    if args.resume:
        last = project / args.name / "weights" / "last.pt"
        if not last.exists():
            sys.exit(f"[error] Nothing to resume: {last} not found.")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        weights_dir = ROOT / "weights"
        weights_dir.mkdir(exist_ok=True)
        model_path = Path(args.model)
        if not model_path.exists() and model_path.parent == Path("."):
            model_path = weights_dir / model_path.name  # download pretrained weights into weights/
        model = YOLO(str(model_path))
        model.train(
            data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=batch, device=device,
            workers=args.workers, patience=args.patience, cache=args.cache,
            project=str(project), name=args.name, exist_ok=True,
            pretrained=True, optimizer="auto", cos_lr=True, close_mosaic=5,
            fliplr=0.5, mosaic=1.0, plots=True, seed=0, verbose=True,
        )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        best = Path(model.trainer.save_dir) / "weights" / "last.pt"
    dest = Path(args.export_to)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    print(f"\n[done] Best weights copied to: {dest}")
    print("Next:  python detect.py --source 0      (webcam)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
