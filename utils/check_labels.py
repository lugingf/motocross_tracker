import cv2
import os
import sys

CLASSES = ["plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
COLORS = [(0,255,0)] + [(255,100,0),(0,100,255),(255,0,100),(0,255,255),
          (255,255,0),(100,0,255),(255,150,0),(0,150,255),(150,255,0),(0,255,150)]

img_dir = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))

i = 0
while 0 <= i < len(files):
    jpg = files[i]
    txt = jpg.replace(".jpg", ".txt")
    img = cv2.imread(os.path.join(img_dir, jpg))
    h, w = img.shape[:2]

    txt_path = os.path.join(img_dir, txt)
    if os.path.exists(txt_path):
        for line in open(txt_path):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, CLASSES[cls], (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    label = f"[{i+1}/{len(files)}] {jpg}"
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("check", img)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == 81 or key == ord('a'):  # left arrow or a
        i -= 1
    else:
        i += 1

cv2.destroyAllWindows()
