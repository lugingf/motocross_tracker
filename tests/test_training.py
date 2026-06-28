"""Tests for training.py — dataset building and validation.

Does NOT test train_model (which requires a real YOLO run).
All tests use temporary directories and synthetic image/label files.
"""
from pathlib import Path

import yaml
import pytest

try:
    from mx_tracker.training import (
        _collect_class_counts,
        _iter_images,
        _label_path_for,
        build_dataset,
        validate_dataset,
    )
except ImportError:
    pytest.skip("ultralytics not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_dir(base: Path, labeled: list[str] = (), unlabeled: list[str] = ()) -> Path:
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in labeled:
        (raw / name).touch()
        (raw / (Path(name).stem + ".txt")).write_text("0 0.5 0.5 0.2 0.3")
    for name in unlabeled:
        (raw / name).touch()
    return raw


def _dataset_dir(
    base: Path,
    train_imgs: list[str] = (),
    val_imgs: list[str] = (),
    train_labels: list[str] = (),
    val_labels: list[str] = (),
) -> Path:
    ds = base / "dataset"
    for split, imgs, labels in [
        ("train", train_imgs, train_labels),
        ("val",   val_imgs,   val_labels),
    ]:
        img_dir = ds / split / "images"
        lbl_dir = ds / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for name in imgs:
            (img_dir / name).touch()
        for name in labels:
            (lbl_dir / (Path(name).stem + ".txt")).write_text("0 0.5 0.5 0.2 0.3")
    return ds


# ---------------------------------------------------------------------------
# _iter_images
# ---------------------------------------------------------------------------

class TestIterImages:
    def test_finds_common_image_suffixes(self, tmp_path):
        for name in ("a.jpg", "b.jpeg", "c.png", "d.bmp"):
            (tmp_path / name).touch()
        names = {p.name for p in _iter_images(tmp_path)}
        assert names == {"a.jpg", "b.jpeg", "c.png", "d.bmp"}

    def test_case_insensitive_suffix_matching(self, tmp_path):
        (tmp_path / "A.JPG").touch()
        (tmp_path / "B.PNG").touch()
        assert len(_iter_images(tmp_path)) == 2

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "img.jpg").touch()
        (tmp_path / "label.txt").touch()
        result = _iter_images(tmp_path)
        assert len(result) == 1 and result[0].name == "img.jpg"

    def test_returns_sorted_list(self, tmp_path):
        for name in ("c.jpg", "a.jpg", "b.jpg"):
            (tmp_path / name).touch()
        assert [p.name for p in _iter_images(tmp_path)] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert _iter_images(tmp_path) == []


# ---------------------------------------------------------------------------
# _label_path_for
# ---------------------------------------------------------------------------

class TestLabelPathFor:
    def test_replaces_suffix_with_txt(self):
        assert _label_path_for(Path("/dir/img.jpg")) == Path("/dir/img.txt")

    def test_works_for_png(self):
        assert _label_path_for(Path("/dir/img.png")).suffix == ".txt"

    def test_preserves_parent_directory(self):
        assert _label_path_for(Path("/deep/path/image.jpg")).parent == Path("/deep/path")


# ---------------------------------------------------------------------------
# _collect_class_counts
# ---------------------------------------------------------------------------

class TestCollectClassCounts:
    def test_counts_class_ids_from_yolo_label_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.3 0.1 0.1\n")
        (tmp_path / "b.txt").write_text("0 0.6 0.6 0.2 0.3\n")
        result = _collect_class_counts(tmp_path)
        assert result[0] == 2
        assert result[1] == 1

    def test_skips_empty_lines(self, tmp_path):
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n\n1 0.3 0.3 0.1 0.1\n")
        assert _collect_class_counts(tmp_path) == {0: 1, 1: 1}

    def test_skips_malformed_lines_not_five_fields(self, tmp_path):
        (tmp_path / "a.txt").write_text("0 0.5 0.5\n0 0.5 0.5 0.2 0.3\n")
        assert _collect_class_counts(tmp_path) == {0: 1}

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        assert _collect_class_counts(tmp_path) == {}

    def test_result_keys_are_sorted(self, tmp_path):
        (tmp_path / "a.txt").write_text("5 0.5 0.5 0.2 0.3\n1 0.3 0.3 0.1 0.1\n3 0.4 0.4 0.2 0.2\n")
        keys = list(_collect_class_counts(tmp_path).keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

class TestValidateDataset:
    def test_counts_images_and_labels(self, tmp_path):
        ds = _dataset_dir(tmp_path,
            train_imgs=["a.jpg", "b.jpg"], val_imgs=["c.jpg"],
            train_labels=["a"],            val_labels=["c"],
        )
        r = validate_dataset(ds)
        assert r.train_images == 2
        assert r.val_images == 1
        assert r.train_labels == 1
        assert r.val_labels == 1

    def test_detects_missing_labels(self, tmp_path):
        ds = _dataset_dir(tmp_path, train_imgs=["a.jpg", "b.jpg"], val_imgs=["c.jpg"])
        r = validate_dataset(ds)
        assert r.missing_train_labels == 2
        assert r.missing_val_labels == 1

    def test_detects_orphan_labels(self, tmp_path):
        ds = _dataset_dir(tmp_path, train_labels=["ghost"])
        r = validate_dataset(ds)
        assert r.orphan_train_labels == 1

    def test_detects_overlap_between_splits(self, tmp_path):
        ds = _dataset_dir(tmp_path, train_imgs=["a.jpg"], val_imgs=["a.jpg"])
        assert validate_dataset(ds).overlap_images == 1

    def test_no_overlap_for_disjoint_splits(self, tmp_path):
        ds = _dataset_dir(tmp_path, train_imgs=["a.jpg"], val_imgs=["b.jpg"])
        assert validate_dataset(ds).overlap_images == 0

    def test_raises_when_dataset_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_dataset(tmp_path / "nonexistent")

    def test_counts_class_ids_in_labels(self, tmp_path):
        ds = _dataset_dir(tmp_path, train_imgs=["a.jpg"], train_labels=["a"])
        r = validate_dataset(ds)
        assert r.class_counts_train == {0: 1}


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def test_creates_train_val_split(self, tmp_path):
        raw = _raw_dir(tmp_path, labeled=["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"])
        r = build_dataset(raw, tmp_path / "ds", train_ratio=0.8, seed=0)
        assert r.train_images + r.val_images == 5

    def test_creates_data_yaml(self, tmp_path):
        raw = _raw_dir(tmp_path, labeled=["a.jpg"])
        out = tmp_path / "ds"
        build_dataset(raw, out)
        data = yaml.safe_load((out / "data.yaml").read_text())
        assert "names" in data
        assert data["nc"] == len(data["names"])

    def test_custom_classes_written_to_data_yaml(self, tmp_path):
        raw = _raw_dir(tmp_path, labeled=["a.jpg"])
        out = tmp_path / "ds"
        build_dataset(raw, out, classes=["plate", "digit"])
        data = yaml.safe_load((out / "data.yaml").read_text())
        assert data["names"] == ["plate", "digit"]
        assert data["nc"] == 2

    def test_unlabeled_images_excluded_by_default(self, tmp_path):
        raw = _raw_dir(tmp_path, labeled=["a.jpg"], unlabeled=["b.jpg"])
        r = build_dataset(raw, tmp_path / "ds")
        assert r.unlabeled_images == 1
        assert r.train_images + r.val_images == 1  # only labeled

    def test_unlabeled_included_when_flag_set(self, tmp_path):
        raw = _raw_dir(tmp_path, labeled=["a.jpg"], unlabeled=["b.jpg"])
        r = build_dataset(raw, tmp_path / "ds", include_unlabeled=True)
        assert r.train_images + r.val_images == 2

    def test_raises_when_raw_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_dataset(tmp_path / "missing", tmp_path / "out")

    def test_clean_removes_existing_output(self, tmp_path):
        raw = _raw_dir(tmp_path, labeled=["a.jpg"])
        out = tmp_path / "ds"
        build_dataset(raw, out)
        marker = out / "stale.txt"
        marker.write_text("should be gone")
        build_dataset(raw, out, clean=True)
        assert not marker.exists()

    def test_no_overlap_between_train_and_val(self, tmp_path):
        raw = _raw_dir(tmp_path, labeled=[f"{i}.jpg" for i in range(10)])
        out = tmp_path / "ds"
        build_dataset(raw, out, train_ratio=0.8, seed=42)
        assert validate_dataset(out).overlap_images == 0

    def test_seed_produces_deterministic_split(self, tmp_path):
        labeled = [f"{i}.jpg" for i in range(8)]
        raw1 = _raw_dir(tmp_path / "r1", labeled=labeled)
        raw2 = _raw_dir(tmp_path / "r2", labeled=labeled)
        r1 = build_dataset(raw1, tmp_path / "ds1", seed=7)
        r2 = build_dataset(raw2, tmp_path / "ds2", seed=7)
        assert r1.train_images == r2.train_images
        assert r1.val_images == r2.val_images
