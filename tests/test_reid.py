"""Tests for reid.py — ReIdentifier in HSV histogram mode.

_FeatureExtractor is patched to None so all tests run in HSV mode
regardless of whether torchreid is installed, and no model download occurs.
"""
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
import pytest

import mx_tracker.reid as _reid_module
from mx_tracker.reid import ReIdentifier


@pytest.fixture(autouse=True)
def force_hsv_mode():
    """Hide _FeatureExtractor so ReIdentifier always uses HSV histogram mode."""
    with patch.object(_reid_module, "_FeatureExtractor", None):
        yield


def _make_reid(gallery_path: Path, thresh: float = 0.5) -> ReIdentifier:
    return ReIdentifier(gallery_path=str(gallery_path), device="cpu", thresh=thresh)


def _solid_bgr(h: int, w: int, channel: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, channel] = 255
    return img


# ---------------------------------------------------------------------------
# Early-exit / empty-gallery cases (mode-agnostic paths)
# ---------------------------------------------------------------------------

class TestReIdentifierCommon:
    def test_none_crop_returns_none(self, tmp_path):
        assert _make_reid(tmp_path).identify(None) is None

    def test_empty_array_crop_returns_none(self, tmp_path):
        assert _make_reid(tmp_path).identify(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    def test_empty_gallery_returns_none(self, tmp_path):
        crop = np.zeros((100, 60, 3), dtype=np.uint8)
        assert _make_reid(tmp_path).identify(crop) is None

    def test_nonexistent_gallery_path_gives_empty_gallery(self, tmp_path):
        reid = _make_reid(tmp_path / "nonexistent")
        assert reid.gallery == {}


# ---------------------------------------------------------------------------
# HSV mode behaviour
# ---------------------------------------------------------------------------

class TestReIdentifierHSV:
    def test_mode_is_hsv(self, tmp_path):
        assert _make_reid(tmp_path).mode == "hsv"

    def test_low_threshold_bumped_to_0_70(self, tmp_path):
        reid = _make_reid(tmp_path, thresh=0.3)
        assert reid.thresh == pytest.approx(0.70)

    def test_threshold_above_half_not_bumped(self, tmp_path):
        reid = _make_reid(tmp_path, thresh=0.6)
        assert reid.thresh == pytest.approx(0.6)

    def test_identifies_rider_by_matching_histogram(self, tmp_path):
        rider_dir = tmp_path / "rider_27"
        rider_dir.mkdir()
        img = np.random.randint(50, 200, (100, 60, 3), dtype=np.uint8)
        crop_path = rider_dir / "crop.png"  # PNG is lossless — no compression artifacts
        cv2.imwrite(str(crop_path), img)

        reid = _make_reid(tmp_path, thresh=0.5)
        # Re-read from disk so the query matches exactly what the gallery loaded
        query = cv2.imread(str(crop_path))
        assert reid.identify(query) == "rider_27"

    def test_very_different_image_returns_none_at_high_threshold(self, tmp_path):
        rider_dir = tmp_path / "rider_A"
        rider_dir.mkdir()
        # Gallery: solid blue image
        cv2.imwrite(str(rider_dir / "crop.jpg"), _solid_bgr(100, 60, 0))

        reid = _make_reid(tmp_path, thresh=0.999)
        # Query: solid red — orthogonal HSV histogram
        assert reid.identify(_solid_bgr(100, 60, 2)) is None

    def test_picks_best_matching_rider_from_multiple(self, tmp_path):
        for name, ch in [("rider_blue", 0), ("rider_red", 2)]:
            d = tmp_path / name
            d.mkdir()
            cv2.imwrite(str(d / "crop.jpg"), _solid_bgr(100, 60, ch))

        reid = _make_reid(tmp_path, thresh=0.0)
        # Blue query → rider_blue must win
        assert reid.identify(_solid_bgr(100, 60, 0)) == "rider_blue"

    def test_gallery_ignores_non_image_files(self, tmp_path):
        rider_dir = tmp_path / "rider_X"
        rider_dir.mkdir()
        (rider_dir / "notes.txt").write_text("not an image")
        reid = _make_reid(tmp_path)
        # cv2.imread returns None for text files → rider not added to gallery
        assert "rider_X" not in reid.gallery

    def test_gallery_entry_added_for_valid_image(self, tmp_path):
        rider_dir = tmp_path / "rider_Y"
        rider_dir.mkdir()
        cv2.imwrite(str(rider_dir / "crop.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
        reid = _make_reid(tmp_path)
        assert "rider_Y" in reid.gallery
