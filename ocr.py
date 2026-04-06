# ocr.py
import cv2
import numpy as np

# буквы, которые часто путаются с цифрами
SIMILAR_MAP = {
    "B": "8",
    "S": "5",
    "O": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "G": "6",
    "Q": "0",
}

def normalize_plate_text(txt: str) -> str:
    txt = txt.upper()
    mapped = "".join(SIMILAR_MAP.get(ch, ch) for ch in txt)
    digits = "".join(ch for ch in mapped if ch.isdigit())
    return digits


class OcrReader:
    def __init__(self, min_conf=0.4, lang="en"):
        self.min_conf = min_conf
        self.enabled = False
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=False,  # поворот строки нам не нужен
            )
            print("[OCR] PaddleOCR initialized")
            self.enabled = True
        except Exception as e:
            print("[OCR] disabled:", e)
            self.ocr = None

    def _preprocess(self, img):
        """
        Агрессивный препроцесс:
        - апскейл;
        - перевод в серый;
        - размытие;
        - порог (Otsu) -> ч/б;
        - авто-инверсия, если фон темнее цифр.
        """
        h, w = img.shape[:2]

        # 1) апскейл, чтобы цифры были крупнее
        target = 400  # побольше, чем было
        scale = max(3.0, target / max(h, w))
        img = cv2.resize(
            img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # 2) серый
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3) лёгкое размытие, чтобы убрать шум
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4) Otsu-порог -> бинарная картинка
        _, th = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 5) проверим, что у нас "цифры светлые на тёмном"
        # если наоборот — инвертируем
        mean_val = th.mean()
        # если белого слишком много, возможно цифры чёрные, фон белый → инвертируем
        if mean_val > 127:
            th = cv2.bitwise_not(th)

        # можно чуть утолщить цифры, чтобы OCR их лучше цеплял
        # kernel = np.ones((3, 3), np.uint8)
        # th = cv2.dilate(th, kernel, iterations=1)

        # PaddleOCR ожидает BGR
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


    def read_plate(self, crop):
        if (not self.enabled) or crop is None or crop.size == 0:
            return None, 0.0

        img = self._preprocess(crop)

        # ВАЖНО: вызываем метод .ocr, а не сам объект
        res = self.ocr.ocr(img)  # <- вот так, не self.ocr(img)

        if not res:
            return None, 0.0

        block = res[0]
        texts = block.get("rec_texts", [])
        scores = block.get("rec_scores", [])

        best_txt, best_conf = None, 0.0
        for txt, conf in zip(texts, scores):
            digits = normalize_plate_text(str(txt))
            if not digits:
                continue
            conf = float(conf)
            if conf > best_conf:
                best_txt, best_conf = digits, conf

        if not best_txt or best_conf < self.min_conf:
            return None, 0.0

        return best_txt, best_conf
