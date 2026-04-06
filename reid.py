# reid.py
import os, cv2, numpy as np

_FeatureExtractor = None
try:
    from torchreid.utils import FeatureExtractor as _FE
    _FeatureExtractor = _FE
except Exception:
    try:
        from torchreid.utils.feature_extractor import FeatureExtractor as _FE
        _FeatureExtractor = _FE
    except Exception:
        pass


class ReIdentifier:
    """
    Универсальный ReID:
      - deep (OSNet) если torchreid доступен
      - иначе HSV-гистограммы.
    """
    def __init__(self, gallery_path, device="mps", thresh=0.60):
        self.gallery = {}
        self.mode = "deep" if _FeatureExtractor is not None else "hsv"
        self.thresh = float(thresh)

        if self.mode == "deep":
            from PIL import Image
            import torch
            dev = device
            if dev is None or dev == "auto":
                dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self.ext = _FeatureExtractor(model_name="osnet_x0_25", device=dev)

            def embed(bgr):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                v = self.ext([Image.fromarray(rgb)])[0]
                v = v / (np.linalg.norm(v) + 1e-12)
                return v[np.newaxis, :]
            self._embed = embed
            print("[ReID] mode: deep (OSNet)")
        else:
            if self.thresh < 0.5:
                self.thresh = 0.70

            def hist_hs(bgr):
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                return hist
            self._hist = hist_hs
            print("[ReID] mode: HSV baseline")

        for rider in os.listdir(gallery_path):
            p = os.path.join(gallery_path, rider)
            if not os.path.isdir(p):
                continue
            feats = []
            for f in os.listdir(p):
                im = cv2.imread(os.path.join(p, f))
                if im is None:
                    continue
                if self.mode == "deep":
                    feats.append(self._embed(im))
                else:
                    feats.append(self._hist(im))
            if feats:
                if self.mode == "deep":
                    self.gallery[rider] = np.vstack(feats)
                else:
                    self.gallery[rider] = feats

        if not self.gallery:
            print("[ReID] warning: gallery is empty")

    def identify(self, crop):
        if crop is None or crop.size == 0 or not self.gallery:
            return "unknown"

        best_name, best = "unknown", -1.0
        if self.mode == "deep":
            q = self._embed(crop)
            for name, G in self.gallery.items():
                s = float(np.max(G @ q.T))  # косинус
                if s > best:
                    best, best_name = s, name
        else:
            q = self._hist(crop)
            for name, Hlist in self.gallery.items():
                for h in Hlist:
                    s = cv2.compareHist(h, q, cv2.HISTCMP_CORREL)
                    if s > best:
                        best, best_name = s, name

        return best_name if best >= self.thresh else "unknown"
