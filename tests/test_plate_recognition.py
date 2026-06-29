"""Plate-number recognition tests.

Two layers:

1. Logic tests (always run): a fake plate model returns crafted boxes so we can
   verify how digit detections are assembled into a number — left-to-right
   ordering, confidence filtering, the plate_has_class digit offset, plate-box
   filtering, and the two-bike split. No weights, no images.

2. Model regression (skipped unless the real model + labeled dataset are
   present locally): runs the actual plate model over labeled crops and asserts
   an exact-match accuracy floor. The dataset lives under data/ (gitignored), so
   this runs locally and is skipped in clean checkouts / CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mx_tracker.config import TrackerSettings
from mx_tracker.plate_reading import detect_all_plate_numbers, detect_plate_number


# ---------------------------------------------------------------------------
# Fake plate model — mimics the slice of the ultralytics API we consume
# ---------------------------------------------------------------------------

class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeBoxes:
    def __init__(self, boxes: list[tuple]) -> None:
        # each box: (x1, y1, x2, y2, class_id, conf)
        self._n = len(boxes)
        self.xyxy = _FakeTensor(np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=float))
        self.cls = _FakeTensor(np.array([b[4] for b in boxes], dtype=float))
        self.conf = _FakeTensor(np.array([b[5] for b in boxes], dtype=float))

    def __len__(self) -> int:
        return self._n


class _FakeResult:
    def __init__(self, boxes: list[tuple]) -> None:
        self.boxes = _FakeBoxes(boxes)


class FakePlateModel:
    """Returns a fixed set of detections regardless of the input crop."""

    def __init__(self, boxes: list[tuple]) -> None:
        self._boxes = boxes

    def __call__(self, crop, verbose=False):  # noqa: ARG002 - mirrors YOLO signature
        return [_FakeResult(self._boxes)]


def _settings(plate_has_class: bool = True, plate_conf: float = 0.1) -> TrackerSettings:
    settings = TrackerSettings()
    settings.models.plate_has_class = plate_has_class
    settings.models.plate_conf = plate_conf
    settings.models.plate_class_id = 0
    settings.reads.min_digits = 1
    return settings


def _crop(width: int = 400, height: int = 120) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# digit class ids when plate_has_class=True: class_id = digit + 1
def _digit_box(x1: float, x2: float, digit: int, conf: float = 0.9, y1: float = 40, y2: float = 90) -> tuple:
    return (x1, y1, x2, y2, digit + 1, conf)


# ---------------------------------------------------------------------------
# detect_plate_number — single best reading
# ---------------------------------------------------------------------------

class TestDetectPlateNumber:
    def test_digits_assembled_left_to_right(self):
        # "133": digit 1 then 3 then 3, given out of order to prove sorting by x
        model = FakePlateModel([
            _digit_box(60, 80, 3),
            _digit_box(10, 30, 1),
            _digit_box(35, 55, 3),
        ])
        read = detect_plate_number(model, _crop(), _settings())
        assert read.text == "133"

    def test_low_confidence_digit_dropped(self):
        model = FakePlateModel([
            _digit_box(10, 30, 1, conf=0.9),
            _digit_box(35, 55, 2, conf=0.05),  # below plate_conf=0.1
        ])
        read = detect_plate_number(model, _crop(), _settings(plate_conf=0.1))
        assert read.text == "1"

    def test_plate_has_class_false_uses_class_as_digit(self):
        # No plate box; classes map directly to digits 0..9
        model = FakePlateModel([
            (10, 40, 30, 90, 5, 0.9),   # class 5 -> digit 5
            (35, 40, 55, 90, 7, 0.9),   # class 7 -> digit 7
        ])
        read = detect_plate_number(model, _crop(), _settings(plate_has_class=False))
        assert read.text == "57"

    def test_digits_outside_plate_box_filtered_out(self):
        # plate box covers x in [0, 100]; one digit inside, one far outside
        model = FakePlateModel([
            (0, 30, 100, 100, 0, 0.95),     # plate box (class 0)
            _digit_box(20, 40, 8),          # inside plate -> kept
            _digit_box(300, 320, 9),        # outside plate -> dropped
        ])
        read = detect_plate_number(model, _crop(), _settings())
        assert read.text == "8"

    def test_overlapping_boxes_deduped_keeping_higher_conf(self):
        # two boxes at nearly the same x: the higher-conf digit wins
        model = FakePlateModel([
            _digit_box(10, 30, 1, conf=0.6),
            _digit_box(12, 32, 7, conf=0.95),  # overlaps the first
        ])
        read = detect_plate_number(model, _crop(), _settings())
        assert read.text == "7"

    def test_no_detections_returns_empty_read(self):
        read = detect_plate_number(FakePlateModel([]), _crop(), _settings())
        assert read.text is None

    def test_respects_min_digits(self):
        model = FakePlateModel([_digit_box(10, 30, 5)])
        settings = _settings()
        settings.reads.min_digits = 2
        read = detect_plate_number(model, _crop(), settings)
        assert read.text is None


# ---------------------------------------------------------------------------
# detect_all_plate_numbers — two bikes in one crop split by a large gap
# ---------------------------------------------------------------------------

class TestDetectAllPlateNumbers:
    def test_two_bikes_split_into_two_numbers(self):
        # group A near x=10..55, group B far away near x=300..345 (gap >> width)
        model = FakePlateModel([
            _digit_box(10, 30, 1),
            _digit_box(35, 55, 2),
            _digit_box(300, 320, 8),
            _digit_box(325, 345, 9),
        ])
        reads = detect_all_plate_numbers(model, _crop(width=400), _settings())
        numbers = {r.text for r in reads}
        assert numbers == {"12", "89"}

    def test_single_group_returns_one_number(self):
        model = FakePlateModel([
            _digit_box(10, 30, 2),
            _digit_box(35, 55, 7),
        ])
        reads = detect_all_plate_numbers(model, _crop(), _settings())
        assert len(reads) == 1
        assert reads[0].text == "27"

    def test_x_center_set_for_ordering(self):
        model = FakePlateModel([
            _digit_box(10, 30, 1),
            _digit_box(300, 320, 9),
        ])
        reads = detect_all_plate_numbers(model, _crop(width=400), _settings())
        # left group has smaller x_center than right group
        by_text = {r.text: r.x_center for r in reads}
        assert by_text["1"] < by_text["9"]


# ---------------------------------------------------------------------------
# Real-model integration tests
#
# These need the plate weights (gitignored), so they skip in a clean checkout.
# The committed fixtures under tests/fixtures/plates/ make the inputs and
# expected outputs version-controlled, so a recognition regression shows up as
# a concrete per-crop failure on any machine that has the model.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLATE_MODEL = _REPO_ROOT / "data" / "models" / "yolov8n_plates_ft1.pt"
_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures" / "plates"
_MANIFEST = _FIXTURES_DIR / "expected.json"

_DATASET_DIR = _REPO_ROOT / "data" / "datasets" / "plates_dataset" / "new_dataset"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_ACCURACY_FLOOR = 0.95  # real model scores ~98% on the labeled set


def _load_manifest() -> list[dict]:
    if not _MANIFEST.exists():
        return []
    return json.loads(_MANIFEST.read_text())


def _build_reader():
    from ultralytics import YOLO

    from mx_tracker.models import get_device
    from mx_tracker.plate_reading import PlateReader

    model = YOLO(str(_PLATE_MODEL))
    model.to(get_device("auto"))
    return PlateReader(model, TrackerSettings())


def _ordered_join(reader, img) -> str:
    """Predicted number(s) as the plate groups joined left-to-right.

    Mirrors how a crossing is read: two bikes in one crop yield two groups,
    so the concatenation matches a label that spans both.
    """
    reads = sorted(reader.read_all(img), key=lambda r: r.x_center)
    return "".join(r.text for r in reads if r.text)


def _ground_truth_number(label_path: Path) -> str | None:
    digits: list[tuple[float, str]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        if class_id == 0:
            continue
        digit = class_id - 1
        if 0 <= digit <= 9:
            digits.append((float(parts[1]), str(digit)))
    if not digits:
        return None
    digits.sort(key=lambda d: d[0])
    return "".join(d for _, d in digits)


@pytest.fixture(scope="module")
def plate_reader():
    if not _PLATE_MODEL.exists():
        pytest.skip("plate model not available (data/ is gitignored)")
    return _build_reader()


@pytest.mark.skipif(not _MANIFEST.exists(), reason="no fixture manifest")
@pytest.mark.parametrize("entry", _load_manifest(), ids=[e["file"] for e in _load_manifest()])
def test_fixture_plate_recognition(entry, plate_reader):
    """Each committed crop must still read to its expected plate number(s)."""
    import cv2

    img = cv2.imread(str(_FIXTURES_DIR / entry["file"]))
    assert img is not None, f"fixture image unreadable: {entry['file']}"
    detected = sorted(r.text for r in plate_reader.read_all(img) if r.text)
    assert detected == sorted(entry["expected"]), (
        f"{entry['file']}: expected {sorted(entry['expected'])}, got {detected}"
    )


@pytest.mark.skipif(
    not (_DATASET_DIR.is_dir() and _PLATE_MODEL.exists()),
    reason="full labeled dataset not available (data/ is gitignored)",
)
def test_real_model_accuracy_on_labeled_crops(plate_reader):
    """Broader local check across the whole labeled dataset, if present."""
    import cv2

    total = correct = 0
    for image_path in sorted(_DATASET_DIR.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        truth = _ground_truth_number(label_path)
        if truth is None:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        total += 1
        if _ordered_join(plate_reader, img) == truth:
            correct += 1

    assert total > 0, "expected labeled images in the dataset"
    accuracy = correct / total
    assert accuracy >= _ACCURACY_FLOOR, f"plate accuracy {accuracy:.1%} ({correct}/{total}) below floor {_ACCURACY_FLOOR:.0%}"
