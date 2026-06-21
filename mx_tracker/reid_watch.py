"""Post-processing ReID watcher.

Run in a second terminal while mx-tracker detect is running:

    mx-tracker reid-watch --run-dir data/artifacts/detect_file_20240119_120000

Polls plate_crops/unresolved/ for new crops and resolves them in priority order:

  1. Manual annotation  — human edits the JSON sidecar, adds "manual_plate": "133"
  2. Plate re-read      -- plate model with lower confidence threshold (--plate-model)
  3. ReID               -- visual similarity against the resolved gallery

Unmatched crops are retried every poll. When the gallery grows (new resolved crops
appear), all previously unmatched crops are retried automatically.

Appends resolved events to events.jsonl and events.csv with lap counting.
"""
from __future__ import annotations

import csv
import fcntl
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .reid import ReIdentifier

Logger = Callable[[str], None]

_CSV_FIELDS = [
    "timestamp",
    "frame_index",
    "tracker_id",
    "rider_id",
    "identity_source",
    "plate_text",
    "plate_conf",
    "lap",
    "lap_time",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "crop_file",
]


def _parse_crop_stem(stem: str) -> tuple[int, int] | None:
    try:
        a, b = stem.split("_", 1)
        return int(a.removeprefix("frame")), int(b.removeprefix("tid"))
    except Exception:
        return None


def _locked_append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(line)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _locked_append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writerow(row)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _resolved_latest_mtime(resolved_dir: Path) -> float:
    try:
        return max((p.stat().st_mtime for p in resolved_dir.rglob("*.jpg")), default=0.0)
    except Exception:
        return 0.0


def _try_plate_number(plate_model: object, img: np.ndarray, conf_threshold: float) -> str | None:
    """Run plate model on img with the given confidence floor, return digit string or None."""
    result = plate_model(img, verbose=False)[0]  # type: ignore[operator]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()

    digits: list[dict] = []
    for (x1, y1, x2, y2), class_id, conf in zip(xyxy, classes, confs):
        if float(conf) < conf_threshold:
            continue
        if class_id == 0:  # plate bbox class — skip
            continue
        digit = class_id - 1
        if not (0 <= digit <= 9):
            continue
        digits.append({
            "x": float(x1 + x2) / 2.0,
            "w": float(max(1, x2 - x1)),
            "x1": float(x1),
            "x2": float(x2),
            "d": str(digit),
            "conf": float(conf),
        })

    if not digits:
        return None

    digits.sort(key=lambda d: d["x"])

    # Dedup overlapping boxes
    deduped: list[dict] = []
    for dig in digits:
        if not deduped:
            deduped.append(dig)
            continue
        prev = deduped[-1]
        threshold = 0.35 * max(prev["w"], dig["w"])
        if abs(dig["x"] - prev["x"]) <= threshold:
            if dig["conf"] > prev["conf"]:
                deduped[-1] = dig
            continue
        deduped.append(dig)

    # Split into groups separated by gap > avg digit width
    groups: list[list[dict]] = []
    for dig in deduped:
        if not groups:
            groups.append([dig])
            continue
        prev = groups[-1][-1]
        gap = dig["x1"] - prev["x2"]
        avg_w = (prev["w"] + dig["w"]) / 2.0
        if gap > avg_w:
            groups.append([dig])
        else:
            groups[-1].append(dig)

    # Pick best group: longest, then highest avg conf
    def _score(g: list[dict]) -> tuple[int, float]:
        return (len(g), sum(d["conf"] for d in g) / len(g))

    best = max(groups, key=_score)
    number = "".join(d["d"] for d in best)
    return number if number else None


class ReidWatcher:
    def __init__(
        self,
        run_dir: Path,
        device: str = "cpu",
        threshold: float = 0.60,
        poll_interval: float = 2.0,
        plate_model_path: str | None = None,
        plate_conf_low: float = 0.15,
        logger: Logger | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.unresolved_dir = run_dir / "plate_crops" / "unresolved"
        self.resolved_dir = run_dir / "plate_crops" / "resolved"
        self.jsonl_path = run_dir / "events.jsonl"
        self.csv_path = run_dir / "events.csv"
        self.poll_interval = poll_interval
        self.plate_conf_low = plate_conf_low
        self.log = logger or print
        self._reid = ReIdentifier(gallery_path="", device=device, thresh=threshold)
        self._gallery_mtime: float = 0.0
        self._matched: set[str] = set()    # crop filenames resolved by any method
        self._attempted: set[str] = set()  # crop filenames tried but unmatched

        self._plate_model: object | None = None
        if plate_model_path:
            from ultralytics import YOLO  # noqa: PLC0415
            self._plate_model = YOLO(plate_model_path)
            self.log(f"[reid-watch] plate model: {plate_model_path}, conf_low={plate_conf_low}")

    def _rebuild_gallery(self) -> bool:
        """Rebuild gallery from resolved crops. Returns True if gallery changed."""
        if not self.resolved_dir.exists():
            return False
        latest = _resolved_latest_mtime(self.resolved_dir)
        if latest <= self._gallery_mtime:
            return False
        self._gallery_mtime = latest
        gallery: dict[str, object] = {}
        for plate_dir in sorted(self.resolved_dir.iterdir()):
            if not plate_dir.is_dir():
                continue
            features: list[np.ndarray] = []
            for img_path in sorted(plate_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    continue
                if self._reid.mode == "deep":
                    features.append(self._reid._embed(img))
                else:
                    features.append(self._reid._hist(img))
            if features:
                gallery[plate_dir.name] = (
                    np.vstack(features) if self._reid.mode == "deep" else features
                )
        self._reid.gallery = gallery
        total = sum(
            (len(v) if isinstance(v, list) else v.shape[0]) for v in gallery.values()
        )
        self.log(f"[reid-watch] gallery rebuilt: {len(gallery)} plate(s), {total} sample(s)")
        return True

    def _load_unresolved_meta(self) -> dict[tuple[int, int], dict]:
        """Return unresolved events from events.jsonl keyed by (frame_index, tracker_id)."""
        meta: dict[tuple[int, int], dict] = {}
        if not self.jsonl_path.exists():
            return meta
        with self.jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("identity_source") == "unresolved":
                    key = (int(record["frame_index"]), int(record["tracker_id"]))
                    meta[key] = record
        return meta

    def _resolve(
        self,
        crop_path: Path,
        frame_index: int,
        tracker_id: int,
        rider_id: str,
        plate_text: str,
        plate_conf: float,
        identity_source: str,
        meta: dict,
    ) -> None:
        bbox = meta.get("bbox", [])
        center = meta.get("center", [])
        row: dict = {
            "timestamp": meta.get("timestamp", ""),
            "frame_index": frame_index,
            "tracker_id": tracker_id,
            "rider_id": rider_id,
            "identity_source": identity_source,
            "plate_text": plate_text,
            "plate_conf": round(plate_conf, 3) if plate_conf else "",
            "lap": "",
            "lap_time": "",
            "bbox_x1": bbox[0] if len(bbox) > 0 else "",
            "bbox_y1": bbox[1] if len(bbox) > 1 else "",
            "bbox_x2": bbox[2] if len(bbox) > 2 else "",
            "bbox_y2": bbox[3] if len(bbox) > 3 else "",
            "center_x": center[0] if len(center) > 0 else "",
            "center_y": center[1] if len(center) > 1 else "",
            "crop_file": str(crop_path),
        }
        if self.jsonl_path.exists():
            _locked_append_jsonl(self.jsonl_path, row)
        if self.csv_path.exists():
            _locked_append_csv(self.csv_path, row)
        self.log(f"[reid-watch] {crop_path.stem} → {rider_id} [{identity_source}]")
        self._matched.add(crop_path.name)

    def _process(self, crop_path: Path) -> None:
        stem = crop_path.stem
        parsed = _parse_crop_stem(stem)
        if parsed is None:
            self._matched.add(crop_path.name)
            return
        frame_index, tracker_id = parsed

        # Priority 1: manual annotation in JSON sidecar (manual_plate filled + save == 1)
        sidecar = crop_path.with_suffix(".json")
        if sidecar.exists():
            try:
                sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
                manual_plate = str(sidecar_data.get("manual_plate") or "").strip()
                if manual_plate and sidecar_data.get("save") == 1:
                    meta = self._load_unresolved_meta().get((frame_index, tracker_id), {})
                    self._resolve(
                        crop_path, frame_index, tracker_id,
                        f"plate_{manual_plate}", manual_plate, 0.0,
                        "manual", meta,
                    )
                    return
            except Exception:
                pass

        # Auto methods are only tried once per crop (retried when gallery grows)
        if crop_path.name in self._attempted:
            return

        img = cv2.imread(str(crop_path))
        if img is None or img.size == 0:
            self._attempted.add(crop_path.name)
            return

        meta = self._load_unresolved_meta().get((frame_index, tracker_id), {})

        # Priority 2: plate model re-detection with lower confidence
        if self._plate_model is not None:
            number = _try_plate_number(self._plate_model, img, self.plate_conf_low)
            if number:
                self._resolve(
                    crop_path, frame_index, tracker_id,
                    f"plate_{number}", number, self.plate_conf_low,
                    "plate_reread", meta,
                )
                return

        # Priority 3: visual ReID
        matched = self._reid.identify(img)
        if matched is not None:
            plate_text = matched.removeprefix("plate_")
            self._resolve(
                crop_path, frame_index, tracker_id,
                matched, plate_text, 0.0,
                "reid_post", meta,
            )
            return

        self._attempted.add(crop_path.name)
        self.log(
            f"[reid-watch] {stem}: no match"
            f" (gallery={len(self._reid.gallery)} plates)"
        )

    def run(self, stop_after_idle_sec: float | None = None) -> None:
        self.log(f"[reid-watch] run_dir={self.run_dir}")
        self.log(f"[reid-watch] mode={self._reid.mode}, threshold={self._reid.thresh}")
        self.log(f"[reid-watch] polling every {self.poll_interval}s")
        idle_since: float | None = None

        while True:
            gallery_changed = self._rebuild_gallery()
            if gallery_changed:
                # New resolved crops → retry all unmatched with updated gallery
                self._attempted.clear()

            pending: list[Path] = []
            if self.unresolved_dir.exists():
                for p in sorted(self.unresolved_dir.glob("*.jpg")):
                    if p.name not in self._matched:
                        pending.append(p)

            any_resolved = False
            for p in pending:
                was_matched = p.name in self._matched
                self._process(p)
                if p.name in self._matched and not was_matched:
                    any_resolved = True

            if any_resolved:
                idle_since = None
            else:
                now = time.monotonic()
                if idle_since is None:
                    idle_since = now
                elif stop_after_idle_sec and (now - idle_since) >= stop_after_idle_sec:
                    self.log("[reid-watch] idle timeout, stopping")
                    break

            time.sleep(self.poll_interval)


def run_reid_watch(
    run_dir: str | Path,
    device: str = "cpu",
    threshold: float = 0.60,
    poll_interval: float = 2.0,
    plate_model_path: str | None = None,
    plate_conf_low: float = 0.15,
    stop_after_idle_sec: float | None = None,
    logger: Logger | None = None,
) -> None:
    watcher = ReidWatcher(
        run_dir=Path(run_dir),
        device=device,
        threshold=threshold,
        poll_interval=poll_interval,
        plate_model_path=plate_model_path,
        plate_conf_low=plate_conf_low,
        logger=logger,
    )
    watcher.run(stop_after_idle_sec=stop_after_idle_sec)
