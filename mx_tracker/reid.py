from __future__ import annotations

import os

import cv2
import numpy as np

_FeatureExtractor = None
try:
    from torchreid.utils import FeatureExtractor as _FE

    _FeatureExtractor = _FE
except Exception:
    try:
        from torchreid.utils.feature_extractor import FeatureExtractor as _FE

        _FeatureExtractor = _FE
    except Exception:
        _FeatureExtractor = None


class ReIdentifier:
    def __init__(self, gallery_path: str, device: str = "cpu", thresh: float = 0.60) -> None:
        self.gallery: dict[str, np.ndarray | list[np.ndarray]] = {}
        self.mode = "deep" if _FeatureExtractor is not None else "hsv"
        self.thresh = float(thresh)
        if self.mode == "deep":
            from PIL import Image
            import torch

            actual_device = device
            if actual_device in {"auto", None}:
                if torch.cuda.is_available():
                    actual_device = "cuda:0"
                elif torch.backends.mps.is_available():
                    actual_device = "mps"
                else:
                    actual_device = "cpu"
            self.extractor = _FeatureExtractor(model_name="osnet_x0_25", device=actual_device)

            def embed(bgr: np.ndarray) -> np.ndarray:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                vector = self.extractor([Image.fromarray(rgb)])[0]
                vector = vector / (np.linalg.norm(vector) + 1e-12)
                return vector[np.newaxis, :]

            self._embed = embed
        else:
            if self.thresh < 0.5:
                self.thresh = 0.70

            def hist_hs(bgr: np.ndarray) -> np.ndarray:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                return hist

            self._hist = hist_hs

        if not os.path.isdir(gallery_path):
            return
        for rider_name in os.listdir(gallery_path):
            rider_dir = os.path.join(gallery_path, rider_name)
            if not os.path.isdir(rider_dir):
                continue
            features: list[np.ndarray] = []
            for filename in os.listdir(rider_dir):
                image = cv2.imread(os.path.join(rider_dir, filename))
                if image is None:
                    continue
                if self.mode == "deep":
                    features.append(self._embed(image))
                else:
                    features.append(self._hist(image))
            if not features:
                continue
            self.gallery[rider_name] = np.vstack(features) if self.mode == "deep" else features

    def identify(self, crop: np.ndarray | None) -> str | None:
        if crop is None or crop.size == 0 or not self.gallery:
            return None
        best_name: str | None = None
        best_score = -1.0
        if self.mode == "deep":
            query = self._embed(crop)
            for rider_name, gallery_vectors in self.gallery.items():
                score = float(np.max(gallery_vectors @ query.T))
                if score > best_score:
                    best_score = score
                    best_name = rider_name
        else:
            query_hist = self._hist(crop)
            for rider_name, histograms in self.gallery.items():
                for histogram in histograms:
                    score = float(cv2.compareHist(histogram, query_hist, cv2.HISTCMP_CORREL))
                    if score > best_score:
                        best_score = score
                        best_name = rider_name
        if best_name is None or best_score < self.thresh:
            return None
        return best_name
