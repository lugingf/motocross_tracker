"""Evaluate plate recognition against a labeled YOLO crop directory.

Each image is expected to have a sibling ``.txt`` YOLO label (``class cx cy w h``)
with class 0 = plate box and classes 1..10 = digits 0..9. The ground-truth plate
number is the digit classes sorted left-to-right by center-x. We then run the
real plate model through ``PlateReader.read_best`` and report exact-match accuracy.

Usage:
    ./myenv/bin/python utils/eval_plates.py \
        --images-dir data/datasets/plates_dataset/new_dataset \
        --plate-model data/models/yolov8n_plates_ft1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx_tracker.config import TrackerSettings  # noqa: E402
from mx_tracker.plate_reading import PlateReader  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ground_truth_number(label_path: Path) -> str | None:
    """Plate number from a YOLO label: digit classes sorted left-to-right."""
    digits: list[tuple[float, str]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        if class_id == 0:  # plate box, not a digit
            continue
        digit = class_id - 1
        if not (0 <= digit <= 9):
            continue
        digits.append((float(parts[1]), str(digit)))
    if not digits:
        return None
    digits.sort(key=lambda d: d[0])
    return "".join(d for _, d in digits)


def evaluate(images_dir: Path, plate_model_path: Path, plate_conf: float, device: str) -> int:
    from mx_tracker.models import get_device  # local import: pulls in torch/ultralytics
    from ultralytics import YOLO

    settings = TrackerSettings()
    settings.models.plate_conf = plate_conf
    model = YOLO(str(plate_model_path))
    model.to(get_device(device))
    reader = PlateReader(model, settings)

    cases = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        truth = ground_truth_number(label_path)
        if truth is None:
            continue
        cases.append((image_path, truth))

    if not cases:
        print(f"No labeled images found in {images_dir}")
        return 1

    correct = 0
    mismatches: list[tuple[str, str, str]] = []
    for image_path, truth in cases:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        # Join plate groups left-to-right so a two-bike crop matches a label
        # that spans both bikes, mirroring how a crossing is actually read.
        reads = sorted(reader.read_all(img), key=lambda r: r.x_center)
        predicted = "".join(r.text for r in reads if r.text)
        if predicted == truth:
            correct += 1
        else:
            mismatches.append((image_path.name, truth, predicted))

    total = len(cases)
    accuracy = correct / total if total else 0.0
    print(f"[eval] images={total} correct={correct} accuracy={accuracy:.1%}")
    if mismatches:
        print(f"[eval] {len(mismatches)} mismatches (truth -> predicted):")
        for name, truth, predicted in mismatches[:30]:
            print(f"  {name}: {truth!r} -> {predicted!r}")
        if len(mismatches) > 30:
            print(f"  ... and {len(mismatches) - 30} more")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate plate recognition on labeled crops")
    parser.add_argument("--images-dir", required=True, help="Directory of images with sibling YOLO .txt labels")
    parser.add_argument("--plate-model", required=True, help="Path to the plate model weights")
    parser.add_argument("--plate-conf", type=float, default=0.25, help="Confidence threshold for digits")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda:0/mps")
    args = parser.parse_args(argv)
    return evaluate(Path(args.images_dir), Path(args.plate_model), args.plate_conf, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
