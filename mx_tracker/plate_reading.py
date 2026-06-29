"""Plate-number recognition from a vehicle crop.

Turns the raw output of the plate YOLO model into ordered digit groups and
then into `PlateRead` values. The grouping logic is intentionally model-free
and pure so it can be unit-tested without loading any weights.

`PlateReader` wraps a loaded plate model and exposes the two ways the pipeline
consumes it: the single best reading and every distinct reading in a crop
(two bikes can share one tracker box).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import TrackerSettings
from .geometry import expand_bbox

if TYPE_CHECKING:
    from ultralytics import YOLO


@dataclass(slots=True)
class PlateRead:
    text: str | None
    confidence: float
    plate_bbox: tuple[int, int, int, int] | None
    x_center: float = 0.0


def _xyxy_array(result_boxes: object, attr: str) -> np.ndarray:
    tensor = getattr(result_boxes, attr)
    return tensor.detach().cpu().numpy()


def _build_digit_groups(
    plate_model: "YOLO | None",
    crop: np.ndarray,
    settings: TrackerSettings,
) -> tuple[list[list[dict]], tuple[int, int, int, int] | None]:
    if plate_model is None or crop.size == 0:
        return [], None
    result = plate_model(crop, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], None

    xyxy = _xyxy_array(boxes, "xyxy")
    classes = _xyxy_array(boxes, "cls").astype(int)
    confidences = _xyxy_array(boxes, "conf")

    plate_candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    digit_candidates: list[dict[str, float | str]] = []
    for raw_bbox, class_id, confidence in zip(xyxy, classes, confidences):
        conf = float(confidence)
        if conf < settings.models.plate_conf:
            continue
        x1, y1, x2, y2 = map(int, raw_bbox.tolist())
        if settings.models.plate_has_class and class_id == settings.models.plate_class_id:
            plate_candidates.append((conf, (x1, y1, x2, y2)))
            continue
        digit = class_id - 1 if settings.models.plate_has_class else class_id
        if not (0 <= digit <= 9):
            continue
        digit_candidates.append(
            {
                "x_center": float(x1 + x2) / 2.0,
                "width": float(max(1, x2 - x1)),
                "digit": str(digit),
                "conf": conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )

    plate_bbox = None
    if plate_candidates:
        _, plate_bbox = max(plate_candidates, key=lambda item: item[0])

    if plate_bbox is not None:
        px1, py1, px2, py2 = expand_bbox(
            plate_bbox[0],
            plate_bbox[1],
            plate_bbox[2],
            plate_bbox[3],
            crop.shape[1],
            crop.shape[0],
            scale=1.0 + settings.models.plate_box_expand,
        )
        filtered_digits = []
        for candidate in digit_candidates:
            cx = float(candidate["x_center"])
            cy = float(candidate["y1"] + candidate["y2"]) / 2.0
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                filtered_digits.append(candidate)
        if filtered_digits:
            digit_candidates = filtered_digits

    if not digit_candidates:
        return [], plate_bbox

    digit_candidates.sort(key=lambda item: item["x_center"])
    deduped: list[dict[str, float | str]] = []
    for candidate in digit_candidates:
        if not deduped:
            deduped.append(candidate)
            continue
        prev = deduped[-1]
        threshold = 0.35 * max(float(prev["width"]), float(candidate["width"]))
        if abs(float(candidate["x_center"]) - float(prev["x_center"])) <= threshold:
            if float(candidate["conf"]) > float(prev["conf"]):
                deduped[-1] = candidate
            continue
        deduped.append(candidate)

    # Split into groups when gap between digits exceeds one digit width —
    # prevents merging digits from two bikes in the same crop.
    groups: list[list[dict]] = []
    for candidate in deduped:
        if not groups:
            groups.append([candidate])
            continue
        prev = groups[-1][-1]
        gap = float(candidate["x1"]) - float(prev["x2"])
        avg_width = (float(prev["width"]) + float(candidate["width"])) / 2.0
        if gap > avg_width:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)

    return groups, plate_bbox


def _group_score(g: list[dict]) -> tuple[int, float]:
    avg_conf = sum(float(d["conf"]) for d in g) / len(g)
    return (len(g), avg_conf)


def _group_to_plate_read(
    group: list[dict],
    plate_bbox: tuple[int, int, int, int] | None,
    min_digits: int,
) -> PlateRead | None:
    number = "".join(str(item["digit"]) for item in group)
    if len(number) < min_digits:
        return None
    confidence = sum(float(item["conf"]) for item in group) / len(group)
    x_center = sum(float(d["x_center"]) for d in group) / len(group)
    return PlateRead(number, confidence, plate_bbox, x_center=x_center)


def detect_plate_number(
    plate_model: "YOLO | None",
    crop: np.ndarray,
    settings: TrackerSettings,
) -> PlateRead:
    groups, plate_bbox = _build_digit_groups(plate_model, crop, settings)
    if not groups:
        return PlateRead(None, 0.0, plate_bbox)
    best_group = max(groups, key=_group_score)
    plate_read = _group_to_plate_read(best_group, plate_bbox, settings.reads.min_digits)
    return plate_read if plate_read is not None else PlateRead(None, 0.0, plate_bbox)


def detect_all_plate_numbers(
    plate_model: "YOLO | None",
    crop: np.ndarray,
    settings: TrackerSettings,
) -> list[PlateRead]:
    groups, plate_bbox = _build_digit_groups(plate_model, crop, settings)
    result = []
    for group in groups:
        plate_read = _group_to_plate_read(group, plate_bbox, settings.reads.min_digits)
        if plate_read is not None:
            result.append(plate_read)
    return result


class PlateReader:
    """Reads plate numbers from crops using a loaded plate model.

    Holds the model and settings so callers don't have to thread them through
    every call. A `None` model yields empty reads (used in collect-only runs).
    """

    def __init__(self, model: "YOLO | None", settings: TrackerSettings) -> None:
        self._model = model
        self._settings = settings

    @property
    def available(self) -> bool:
        return self._model is not None

    def read_best(self, crop: np.ndarray) -> PlateRead:
        """Single most likely plate in the crop."""
        return detect_plate_number(self._model, crop, self._settings)

    def read_all(self, crop: np.ndarray) -> list[PlateRead]:
        """Every distinct plate reading in the crop (one per bike)."""
        return detect_all_plate_numbers(self._model, crop, self._settings)
