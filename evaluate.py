"""Evaluate trained weights on the held-out split and print per-class metrics.

Usage:
    python evaluate.py
    python evaluate.py --split val --weights weights/best.pt
"""
import argparse
import sys
from pathlib import Path

from common import DEFAULT_DATA_YAML, DEFAULT_WEIGHTS, ROOT, pick_device


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--data", default=str(DEFAULT_DATA_YAML))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=0, help="0 = size the model was trained at")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    if not Path(args.weights).exists():
        sys.exit(f"[error] Weights not found: {args.weights}. Run  python train.py  first.")
    if not Path(args.data).exists():
        sys.exit(f"[error] {args.data} not found. Run  python prepare_dataset.py  first.")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    imgsz = args.imgsz or int(model.overrides.get("imgsz", 640))
    m = model.val(data=args.data, split=args.split, imgsz=imgsz, device=pick_device(args.device),
                  project=str(ROOT / "outputs" / "eval"), name=args.split, exist_ok=True, plots=True, verbose=False)

    print(f"\nResults on '{args.split}' split")
    print(f"{'class':<24}{'precision':>10}{'recall':>10}{'mAP50':>10}{'mAP50-95':>10}")
    for i, cidx in enumerate(m.box.ap_class_index):
        print(f"{model.names[int(cidx)]:<24}{m.box.p[i]:>10.3f}{m.box.r[i]:>10.3f}"
              f"{m.box.ap50[i]:>10.3f}{m.box.ap[i]:>10.3f}")
    print(f"{'ALL':<24}{m.box.mp:>10.3f}{m.box.mr:>10.3f}{m.box.map50:>10.3f}{m.box.map:>10.3f}")
    print(f"\nPlots (confusion matrix, PR curve) saved in {ROOT / 'outputs' / 'eval' / args.split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
