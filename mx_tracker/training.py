from __future__ import annotations

import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .runtime import prepare_runtime_environment

prepare_runtime_environment()

from ultralytics import YOLO

from .runtime import REPO_ROOT


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class DatasetBuildReport:
    raw_images: int
    labeled_images: int
    unlabeled_images: int
    train_images: int
    val_images: int
    train_labels: int
    val_labels: int
    overlap_images: int
    overlap_labels: int
    classes: list[str]
    output_dir: str
    data_yaml: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DatasetValidationReport:
    train_images: int
    val_images: int
    train_labels: int
    val_labels: int
    missing_train_labels: int
    missing_val_labels: int
    orphan_train_labels: int
    orphan_val_labels: int
    overlap_images: int
    overlap_labels: int
    class_counts_train: dict[int, int]
    class_counts_val: dict[int, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _iter_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _label_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def build_dataset(
    raw_dir: str | Path,
    dataset_dir: str | Path,
    train_ratio: float = 0.8,
    seed: int = 0,
    clean: bool = False,
    include_unlabeled: bool = False,
    classes: list[str] | None = None,
) -> DatasetBuildReport:
    raw_path = Path(raw_dir).expanduser().resolve()
    out_path = Path(dataset_dir).expanduser().resolve()
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")
    if clean and out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    images = _iter_images(raw_path)
    labeled_images: list[Path] = []
    unlabeled_images: list[Path] = []
    for image_path in images:
        if _label_path_for(image_path).exists():
            labeled_images.append(image_path)
        else:
            unlabeled_images.append(image_path)

    rng = random.Random(seed)
    rng.shuffle(labeled_images)
    labeled_split_index = int(round(len(labeled_images) * train_ratio))
    train_images = labeled_images[:labeled_split_index]
    val_images = labeled_images[labeled_split_index:]

    if include_unlabeled and unlabeled_images:
        rng.shuffle(unlabeled_images)
        unlabeled_split_index = int(round(len(unlabeled_images) * train_ratio))
        train_images.extend(unlabeled_images[:unlabeled_split_index])
        val_images.extend(unlabeled_images[unlabeled_split_index:])

    for split_name, split_images in {"train": train_images, "val": val_images}.items():
        image_dir = out_path / split_name / "images"
        label_dir = out_path / split_name / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            target_image = image_dir / image_path.name
            shutil.copy2(image_path, target_image)
            label_path = _label_path_for(image_path)
            if label_path.exists():
                shutil.copy2(label_path, label_dir / label_path.name)
            elif include_unlabeled:
                (label_dir / label_path.name).write_text("")

    class_names = classes or ["plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    data_yaml_path = out_path / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "train": "train/images",
                "val": "val/images",
                "nc": len(class_names),
                "names": class_names,
            },
            sort_keys=False,
        )
    )

    validation = validate_dataset(out_path)
    return DatasetBuildReport(
        raw_images=len(images),
        labeled_images=len(labeled_images),
        unlabeled_images=len(unlabeled_images),
        train_images=validation.train_images,
        val_images=validation.val_images,
        train_labels=validation.train_labels,
        val_labels=validation.val_labels,
        overlap_images=validation.overlap_images,
        overlap_labels=validation.overlap_labels,
        classes=class_names,
        output_dir=str(out_path),
        data_yaml=str(data_yaml_path),
    )


def _collect_class_counts(label_dir: Path) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for label_file in sorted(label_dir.glob("*.txt")):
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            counts[int(parts[0])] += 1
    return dict(sorted(counts.items()))


def validate_dataset(dataset_dir: str | Path) -> DatasetValidationReport:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    train_images = {path.stem for path in (dataset_path / "train" / "images").glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    val_images = {path.stem for path in (dataset_path / "val" / "images").glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    train_labels = {path.stem for path in (dataset_path / "train" / "labels").glob("*.txt")}
    val_labels = {path.stem for path in (dataset_path / "val" / "labels").glob("*.txt")}
    return DatasetValidationReport(
        train_images=len(train_images),
        val_images=len(val_images),
        train_labels=len(train_labels),
        val_labels=len(val_labels),
        missing_train_labels=len(train_images - train_labels),
        missing_val_labels=len(val_images - val_labels),
        orphan_train_labels=len(train_labels - train_images),
        orphan_val_labels=len(val_labels - val_images),
        overlap_images=len(train_images & val_images),
        overlap_labels=len(train_labels & val_labels),
        class_counts_train=_collect_class_counts(dataset_path / "train" / "labels"),
        class_counts_val=_collect_class_counts(dataset_path / "val" / "labels"),
    )


def train_model(
    data_yaml: str | Path,
    model_path: str | Path,
    project_dir: str | Path,
    run_name: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    workers: int = 8,
) -> dict[str, object]:
    dataset_yaml = Path(data_yaml).expanduser().resolve()
    model_source = Path(model_path).expanduser()
    if not model_source.is_absolute():
        cwd_candidate = (Path.cwd() / model_source).resolve()
        repo_candidate = (REPO_ROOT / model_source).resolve()
        model_source = cwd_candidate if cwd_candidate.exists() else repo_candidate
    project_path = Path(project_dir).expanduser().resolve()
    project_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_source))
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(project_path),
        name=run_name,
    )
    save_dir = getattr(results, "save_dir", None)
    return {
        "data_yaml": str(dataset_yaml),
        "model": str(model_source),
        "project_dir": str(project_path),
        "run_name": run_name,
        "save_dir": str(save_dir) if save_dir is not None else None,
    }
