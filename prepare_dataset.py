"""Download / convert the Face Mask Detection dataset (Pascal VOC) to YOLO format.

Default dataset: Kaggle "andrewmvd/face-mask-detection" (853 images, 3 classes:
with_mask, mask_weared_incorrect, without_mask).

Ways to supply the data (first match wins):
  1. --source <folder or .zip>      (e.g. the zip you downloaded manually from Kaggle)
  2. datasets/raw/                  (put the unzipped dataset or the zip here)
  3. automatic download through `kagglehub`

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --source C:/Downloads/face-mask-detection.zip
"""
import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from common import CLASS_NAMES, DEFAULT_DATA_YAML, IMAGE_EXTS, ROOT

ALIASES = {
    "with_mask": 0, "with mask": 0, "mask": 0, "mask_on": 0, "good": 0,
    "mask_weared_incorrect": 1, "mask_worn_incorrect": 1, "mask_weared_incorrectly": 1,
    "incorrect": 1, "improper": 1, "mask_incorrect": 1,
    "without_mask": 2, "without mask": 2, "no_mask": 2, "nomask": 2, "bad": 2,
}

KAGGLE_HANDLE = "andrewmvd/face-mask-detection"
MANUAL_HELP = f"""
Could not obtain the dataset automatically. Do this once, manually:
  1. Open https://www.kaggle.com/datasets/{KAGGLE_HANDLE}  (free Kaggle login)
  2. Click 'Download' -> you get a .zip file
  3. Run:   python prepare_dataset.py --source "<path to that zip>"
     (or put the zip / unzipped folders 'images' and 'annotations' into  {ROOT / 'datasets' / 'raw'})
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_source(arg_source: str | None) -> Path:
    raw_dir = ROOT / "datasets" / "raw"
    if arg_source:
        p = Path(arg_source).expanduser()
        if not p.exists():
            sys.exit(f"[error] --source path does not exist: {p}")
        return unpack_if_zip(p)

    if raw_dir.exists() and any(raw_dir.rglob("*.xml")):
        return raw_dir
    if raw_dir.exists():
        zips = list(raw_dir.glob("*.zip"))
        if zips:
            return unpack_if_zip(zips[0])

    log("[info] No local dataset found, trying Kaggle download via kagglehub ...")
    try:
        import kagglehub  # noqa: WPS433

        path = kagglehub.dataset_download(KAGGLE_HANDLE)
        return Path(path)
    except Exception as exc:  # network, auth, missing package ...
        log(f"[warn] kagglehub download failed: {exc}")
        sys.exit(MANUAL_HELP)


def unpack_if_zip(p: Path) -> Path:
    if p.is_file() and p.suffix.lower() == ".zip":
        dest = ROOT / "datasets" / "raw"
        dest.mkdir(parents=True, exist_ok=True)
        log(f"[info] Extracting {p.name} -> {dest}")
        with zipfile.ZipFile(p) as zf:
            zf.extractall(dest)
        return dest
    return p


def parse_voc(xml_path: Path):
    """Return list of (class_id, xmin, ymin, xmax, ymax) and a list of unknown class names."""
    root = ET.parse(xml_path).getroot()
    objs, unknown = [], []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip().lower()
        bb = o.find("bndbox")
        if bb is None:
            continue
        if name not in ALIASES:
            unknown.append(name)
            continue
        try:
            vals = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        objs.append((ALIASES[name], *vals))
    return objs, unknown


def to_yolo_line(cls, xmin, ymin, xmax, ymax, w, h):
    xmin, xmax = max(0.0, min(xmin, xmax)), min(float(w), max(xmin, xmax))
    ymin, ymax = max(0.0, min(ymin, ymax)), min(float(h), max(ymin, ymax))
    bw, bh = xmax - xmin, ymax - ymin
    if bw < 2 or bh < 2:
        return None
    xc, yc = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
    return f"{cls} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="Folder or .zip with VOC images + annotations")
    ap.add_argument("--out", default=str(DEFAULT_DATA_YAML.parent), help="Output YOLO dataset folder")
    ap.add_argument("--val", type=float, default=0.15, help="Validation fraction")
    ap.add_argument("--test", type=float, default=0.10, help="Test fraction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from PIL import Image  # installed together with ultralytics

    src = resolve_source(args.source)
    log(f"[info] Reading dataset from: {src}")

    images = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            images.setdefault(p.stem, p)
    xmls = sorted(src.rglob("*.xml"))
    if not xmls or not images:
        sys.exit(f"[error] Found {len(images)} images and {len(xmls)} XML files in {src}. "
                 "Expected Pascal-VOC style 'images' + 'annotations' folders.\n" + MANUAL_HELP)

    samples, unknown_all, missing = [], {}, 0
    for x in xmls:
        img_path = images.get(x.stem)
        if img_path is None:
            missing += 1
            continue
        try:
            objs, unknown = parse_voc(x)
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as exc:
            log(f"[warn] skipping {x.name}: {exc}")
            continue
        for u in unknown:
            unknown_all[u] = unknown_all.get(u, 0) + 1
        lines = [l for l in (to_yolo_line(c, a, b, d, e, w, h) for c, a, b, d, e in objs) if l]
        samples.append((img_path, lines))

    if not samples:
        sys.exit("[error] No usable image/annotation pairs found.")
    if missing:
        log(f"[warn] {missing} annotation files had no matching image (skipped).")
    if unknown_all:
        log(f"[warn] Unknown class names ignored: {unknown_all}")

    random.Random(args.seed).shuffle(samples)
    n = len(samples)
    n_test = max(1, round(n * args.test)) if n >= 10 else 0
    n_val = max(1, round(n * args.val)) if n >= 3 else 0
    splits = {
        "test": samples[:n_test],
        "val": samples[n_test:n_test + n_val],
        "train": samples[n_test + n_val:],
    }

    out = Path(args.out).resolve()
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    stats = {}
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts = [0] * len(CLASS_NAMES)
        for img_path, lines in items:
            shutil.copy2(img_path, out / "images" / split / img_path.name)
            (out / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            for l in lines:
                counts[int(l.split()[0])] += 1
        stats[split] = (len(items), counts)

    yaml_lines = [f"path: {out.as_posix()}", "train: images/train", "val: images/val"]
    if splits["test"]:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    log("\nDataset ready:")
    for split, (cnt, counts) in stats.items():
        per = ", ".join(f"{CLASS_NAMES[i]}={c}" for i, c in enumerate(counts))
        log(f"  {split:<5} {cnt:>5} images | {per}")
    log(f"\ndata.yaml -> {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
