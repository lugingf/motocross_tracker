"""Tests for reid_watch.py — crop stem parsing, manual annotation flow."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mx_tracker.reid_watch import ReidWatcher, _parse_crop_stem, _try_plate_number


# ---------------------------------------------------------------------------
# _parse_crop_stem — extracts (frame_index, tracker_id) from filename stem
# ---------------------------------------------------------------------------

class TestParseCropStem:
    def test_standard_stem_parsed_correctly(self):
        assert _parse_crop_stem("frame51624_tid6225") == (51624, 6225)

    def test_small_indices(self):
        assert _parse_crop_stem("frame0_tid0") == (0, 0)

    def test_large_indices(self):
        assert _parse_crop_stem("frame999999_tid12345") == (999999, 12345)

    def test_invalid_stem_returns_none(self):
        assert _parse_crop_stem("somerandombname") is None

    def test_missing_tid_prefix_returns_none(self):
        assert _parse_crop_stem("frame100_id200") is None

    def test_non_numeric_indices_return_none(self):
        assert _parse_crop_stem("frameABC_tidXYZ") is None


# ---------------------------------------------------------------------------
# Manual annotation: save=1 + manual_plate triggers resolution
# ---------------------------------------------------------------------------

class TestManualAnnotation:
    def _make_run_dir(self, tmp_path: Path) -> tuple[Path, Path]:
        run_dir = tmp_path / "run"
        unresolved = run_dir / "plate_crops" / "unresolved"
        unresolved.mkdir(parents=True)
        (run_dir / "events.jsonl").write_text(
            json.dumps({
                "timestamp": 12.5,
                "frame_index": 100,
                "tracker_id": 5,
                "identity_source": "unresolved",
                "rider_id": "",
                "bbox": [10, 20, 50, 80],
                "center": [30, 50],
            }) + "\n",
            encoding="utf-8",
        )
        (run_dir / "events.csv").write_text(
            "timestamp,frame_index,tracker_id,rider_id,identity_source,"
            "plate_text,plate_conf,lap,lap_time,"
            "bbox_x1,bbox_y1,bbox_x2,bbox_y2,center_x,center_y,crop_file\n",
            encoding="utf-8",
        )
        return run_dir, unresolved

    def _make_crop(self, unresolved: Path, stem: str, manual_plate: str, save: int) -> Path:
        import numpy as np
        import cv2
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        crop_path = unresolved / f"{stem}.jpg"
        cv2.imwrite(str(crop_path), img)
        sidecar = unresolved / f"{stem}.json"
        sidecar.write_text(
            json.dumps({"manual_plate": manual_plate, "save": save}),
            encoding="utf-8",
        )
        return crop_path

    def test_save_1_with_plate_number_resolves_event(self, tmp_path):
        run_dir, unresolved = self._make_run_dir(tmp_path)
        self._make_crop(unresolved, "frame100_tid5", "133", save=1)

        watcher = ReidWatcher(run_dir=run_dir)
        crop_path = unresolved / "frame100_tid5.jpg"
        watcher._process(crop_path)

        assert "frame100_tid5.jpg" in watcher._matched
        # Check event was written to jsonl
        lines = (run_dir / "events.jsonl").read_text().splitlines()
        resolved = [json.loads(l) for l in lines if json.loads(l).get("identity_source") == "manual"]
        assert len(resolved) == 1
        assert resolved[0]["plate_text"] == "133"
        assert resolved[0]["rider_id"] == "plate_133"

    def test_save_0_even_with_plate_number_is_ignored(self, tmp_path):
        run_dir, unresolved = self._make_run_dir(tmp_path)
        self._make_crop(unresolved, "frame100_tid5", "133", save=0)

        watcher = ReidWatcher(run_dir=run_dir)
        crop_path = unresolved / "frame100_tid5.jpg"
        watcher._process(crop_path)

        assert "frame100_tid5.jpg" not in watcher._matched

    def test_empty_manual_plate_is_ignored(self, tmp_path):
        run_dir, unresolved = self._make_run_dir(tmp_path)
        self._make_crop(unresolved, "frame100_tid5", "", save=1)

        watcher = ReidWatcher(run_dir=run_dir)
        crop_path = unresolved / "frame100_tid5.jpg"
        watcher._process(crop_path)

        assert "frame100_tid5.jpg" not in watcher._matched

    def test_manual_annotation_checked_on_every_poll_even_after_auto_failed(self, tmp_path):
        """If auto methods failed, the watcher must still check for manual_plate later."""
        run_dir, unresolved = self._make_run_dir(tmp_path)
        # Start with save=0 (human hasn't confirmed yet)
        crop_path = self._make_crop(unresolved, "frame100_tid5", "27", save=0)

        watcher = ReidWatcher(run_dir=run_dir)
        watcher._process(crop_path)
        assert "frame100_tid5.jpg" not in watcher._matched

        # Human confirms — update sidecar to save=1
        sidecar = unresolved / "frame100_tid5.json"
        sidecar.write_text(json.dumps({"manual_plate": "27", "save": 1}), encoding="utf-8")

        # Second poll must pick it up
        watcher._process(crop_path)
        assert "frame100_tid5.jpg" in watcher._matched


# ---------------------------------------------------------------------------
# _try_plate_number — digit extraction from a mocked YOLO result
# ---------------------------------------------------------------------------

class TestTryPlateNumber:
    def _make_mock_model(self, detections: list[tuple[float, float, float, float, int, float]]):
        """detections: list of (x1, y1, x2, y2, class_id, conf)"""
        import numpy as np
        import torch

        boxes = MagicMock()
        if detections:
            xyxy_arr = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=float)
            cls_arr = np.array([d[4] for d in detections], dtype=float)
            conf_arr = np.array([d[5] for d in detections], dtype=float)
        else:
            xyxy_arr = np.zeros((0, 4), dtype=float)
            cls_arr = np.zeros(0, dtype=float)
            conf_arr = np.zeros(0, dtype=float)

        boxes.xyxy = torch.tensor(xyxy_arr)
        boxes.cls = torch.tensor(cls_arr)
        boxes.conf = torch.tensor(conf_arr)
        boxes.__len__ = lambda self: len(detections)

        result = MagicMock()
        result.boxes = boxes
        model = MagicMock()
        model.return_value = [result]
        return model

    def test_no_detections_returns_none(self):
        import numpy as np
        model = self._make_mock_model([])
        model.return_value[0].boxes.__len__ = lambda self: 0
        # Override: boxes is None path
        model.return_value[0].boxes = None
        assert _try_plate_number(model, np.zeros((50, 100, 3), dtype="uint8"), 0.25) is None

    def test_digits_joined_left_to_right(self):
        import numpy as np
        # class_id=2 → digit "1", class_id=4 → digit "3", class_id=4 → digit "3"
        detections = [
            (10, 5, 20, 25, 2, 0.9),   # digit "1"
            (30, 5, 40, 25, 4, 0.85),  # digit "3"
            (50, 5, 60, 25, 4, 0.8),   # digit "3"
        ]
        model = self._make_mock_model(detections)
        img = np.zeros((50, 100, 3), dtype="uint8")
        result = _try_plate_number(model, img, conf_threshold=0.5)
        assert result == "133"

    def test_low_confidence_digits_filtered_out(self):
        import numpy as np
        detections = [
            (10, 5, 20, 25, 2, 0.9),   # digit "1" — kept
            (30, 5, 40, 25, 4, 0.1),   # digit "3" — filtered (conf < threshold)
        ]
        model = self._make_mock_model(detections)
        img = np.zeros((50, 100, 3), dtype="uint8")
        result = _try_plate_number(model, img, conf_threshold=0.5)
        assert result == "1"

    def test_plate_class_zero_skipped(self):
        import numpy as np
        detections = [
            (0, 0, 80, 30, 0, 0.95),   # class 0 = plate bbox — must be skipped
            (10, 5, 20, 25, 2, 0.9),   # digit "1"
        ]
        model = self._make_mock_model(detections)
        img = np.zeros((50, 100, 3), dtype="uint8")
        result = _try_plate_number(model, img, conf_threshold=0.5)
        assert result == "1"

    def test_two_separated_groups_picks_longer_one(self):
        import numpy as np
        # "44" (left group) and "9" (right group, gap > avg width)
        # gap between x2=40 and x1=200 is 160, avg_width ≈ 10 → different group
        detections = [
            (10, 5, 20, 25, 5, 0.9),   # digit "4"
            (25, 5, 35, 25, 5, 0.9),   # digit "4"
            (200, 5, 210, 25, 10, 0.9), # digit "9"
        ]
        model = self._make_mock_model(detections)
        img = np.zeros((50, 300, 3), dtype="uint8")
        result = _try_plate_number(model, img, conf_threshold=0.5)
        assert result == "44"  # longer group wins
