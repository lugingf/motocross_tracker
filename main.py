import cv2
from ultralytics import YOLO
import supervision as sv
import pandas as pd
import time
import numpy as np
import torch, os, pathlib, importlib.util

from reid import ReIdentifier


def expand_bbox(x1, y1, x2, y2, W, H, scale=1.2):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    x1n = max(0, int(cx - w / 2))
    y1n = max(0, int(cy - h / 2))
    x2n = min(W, int(cx + w / 2))
    y2n = min(H, int(cy + h / 2))
    return x1n, y1n, x2n, y2n


def parse_line_arg(arg, W, H):
    def pt(tok, dim):
        tok = tok.strip()
        if tok.endswith("%"):
            return int(round(float(tok[:-1]) * dim / 100.0))
        return int(round(float(tok)))

    x1, y1, x2, y2 = [t.strip() for t in arg.split(",")]
    return pt(x1, W), pt(y1, H), pt(x2, W), pt(y2, H)


def to_percent_str(x1, y1, x2, y2, W, H):
    def pct(v, d): return f"{(v / d) * 100:.2f}%"

    return f'{pct(x1, W)},{pct(y1, H)},{pct(x2, W)},{pct(y2, H)}'


def pick_line_on_frame(frame):
    win = "Pick finish line: 2 clicks, Enter to confirm, R to reset, Esc to cancel"
    pts = []
    vis = frame.copy()

    def cb(event, x, y, flags, param):
        nonlocal vis, pts
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < 2:
                pts.append((x, y))
                cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
            if len(pts) == 2:
                cv2.line(vis, pts[0], pts[1], (0, 255, 255), 2)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while True:
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):  # Enter
            break
        elif k == 27:  # Esc
            pts = []
            break
        elif k in (ord('r'), ord('R')):
            pts = []
            vis = frame.copy()
    cv2.destroyWindow(win)
    return pts if len(pts) == 2 else None


def point_line_side_and_dist(P, A, B):
    A = np.array(A, float)
    B = np.array(B, float)
    P = np.array(P, float)
    AB = B - A
    AP = P - A
    t = max(0.0, min(1.0, (AP @ AB) / (AB @ AB + 1e-9)))
    H = A + t * AB
    side = np.sign(AB[0] * AP[1] - AB[1] * AP[0])
    dist = float(np.linalg.norm(P - H))
    return side, dist


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a, b, c):
    return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))


def segments_intersect(p1, p2, q1, q2):
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    if (o1 == 0 and _on_seg(p1, p2, q1)) or \
            (o2 == 0 and _on_seg(p1, p2, q2)) or \
            (o3 == 0 and _on_seg(q1, q2, p1)) or \
            (o4 == 0 and _on_seg(q1, q2, p2)):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


# ---------- детекция цифр на кропе мотоцикла ----------

def detect_plate_number(digit_model, crop, has_plate_class=False, conf_th=0.25):
    """
    digit_model: YOLO-модель цифр
    crop: np.ndarray (BGR), кроп всего мотоцикла
    has_plate_class: True, если в датасете есть класс 'plate' с id=0.
                     False, если классы только цифр 0..9.
    """
    res = digit_model(crop, verbose=False)[0]
    boxes = res.boxes

    if boxes is None or len(boxes) == 0:
        return "", 0.0

    digits = []
    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        c = float(conf)
        if c < conf_th:
            continue

        cls_id = int(cls.item())
        if has_plate_class:
            # 0 = plate, 1..10 = цифры 0..9
            if cls_id == 0:
                continue
            digit = cls_id - 1
        else:
            # 0..9 = цифры 0..9
            digit = cls_id

        if digit < 0 or digit > 9:
            continue

        x1, y1, x2, y2 = box.tolist()
        x_center = (x1 + x2) / 2.0
        digits.append((x_center, str(digit), c))

    if not digits:
        return "", 0.0

    digits.sort(key=lambda x: x[0])  # слева направо
    number = "".join(d[1] for d in digits)
    avg_conf = sum(d[2] for d in digits) / len(digits)
    return number, avg_conf


# -------------------------------------------------------------

def main(video_source,
         gallery_path,
         output_video,
         output_csv,
         line_str,
         line_width,
         calibrate_line,
         get_data=False,
         digits_model_path="runs/detect/train3/weights/best.pt",
         has_plate_class=False):

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    capture = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
    if not capture.isOpened():
        print("Error: cannot open video source", video_source)
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if calibrate_line:
        ok, frame0 = capture.read()
        if not ok:
            print("Cannot read first frame for calibration")
            return
        pts = pick_line_on_frame(frame0)
        if not pts:
            print("Calibration cancelled")
            return
        (x1, y1), (x2, y2) = pts
        line_str = to_percent_str(x1, y1, x2, y2, W, H)
        print("Use this line next time:")
        print(f'--line "{line_str}"')
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    Lx1, Ly1, Lx2, Ly2 = parse_line_arg(line_str, W, H)
    print(f"Finish line: ({Lx1},{Ly1})-({Lx2},{Ly2}), width={line_width}px")

    # YOLO для мотоциклов/трекера
    model = YOLO("yolov8n.pt")
    model.to(DEVICE)

    # YOLO для цифр
    digit_model = YOLO(digits_model_path)
    digit_model.to(DEVICE)

    spec = importlib.util.find_spec("ultralytics")
    ultra_dir = pathlib.Path(spec.submodule_search_locations[0])
    tracker_yaml = ultra_dir / "cfg/trackers/botsort.yaml"

    box_annotator = sv.BoxAnnotator()

    # ReID оставляем как дополнительный источник ID (если захочешь)
    reid = ReIdentifier(gallery_path) if not get_data else None

    log = []
    last_cross_t_by_tid = {}
    prev_center_by_tid = {}
    id_to_rider = {}
    lap_counts = {}
    last_rider_time = {}

    crops_dir = "plates_dataset"
    saved_crops = 0
    if get_data:
        os.makedirs(crops_dir, exist_ok=True)

    frame_idx = 0
    is_live = str(video_source).isdigit()
    t0_live = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_for_crop = frame.copy()
        frame_idx += 1

        if is_live:
            video_ts = time.perf_counter() - t0_live
        else:
            ts_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            video_ts = (ts_ms / 1000.0) if ts_ms else frame_idx / fps

        res = model.track(
            frame, tracker=str(tracker_yaml), persist=True,
            device=DEVICE, conf=0.35, iou=0.5, imgsz=1280
        )[0]

        detects = sv.Detections.from_ultralytics(res)
        if len(detects) == 0:
            cv2.line(frame, (Lx1, Ly1), (Lx2, Ly2), (255, 255, 255), 2)
            writer.write(frame)
            continue

        keep = detects.class_id == 3  # motorcycle
        detects = detects[keep]
        tids = detects.tracker_id
        if tids is None:
            cv2.line(frame, (Lx1, Ly1), (Lx2, Ly2), (255, 255, 255), 2)
            writer.write(frame)
            continue

        for bbox, tid_val in zip(detects.xyxy, tids):
            if tid_val is None:
                continue
            tid = int(tid_val)
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cur_c = (cx, cy)

            prev_c = prev_center_by_tid.get(tid)
            prev_center_by_tid[tid] = cur_c

            if prev_c is None:
                continue

            side_prev, d_prev = point_line_side_and_dist(prev_c, (Lx1, Ly1), (Lx2, Ly2))
            side_cur, d_cur = point_line_side_and_dist(cur_c, (Lx1, Ly1), (Lx2, Ly2))

            inside_prev = d_prev <= line_width / 2.0
            inside_cur = d_cur <= line_width / 2.0

            crossed = False

            if (side_prev * side_cur < 0) and (min(d_prev, d_cur) <= line_width):
                crossed = True
            elif inside_prev != inside_cur:
                crossed = True
            elif segments_intersect(prev_c, cur_c, (Lx1, Ly1), (Lx2, Ly2)):
                crossed = True

            if not crossed:
                continue

            last_t = last_cross_t_by_tid.get(tid, -1e9)
            if (video_ts - last_t) < 2.0:
                continue
            last_cross_t_by_tid[tid] = video_ts

            # кроп всего мотоцикла
            x1e, y1e, x2e, y2e = expand_bbox(x1, y1, x2, y2, W, H, scale=1.10)
            crop = frame_for_crop[y1e:y2e, x1e:x2e]
            h, w = crop.shape[:2]

            if get_data:
                # в режиме сбора данных по-прежнему сохраняем весь кроп мотоцикла
                filename = os.path.join(
                    crops_dir,
                    f"frame{frame_idx}_tid{tid}.jpg"
                )
                cv2.imwrite(filename, crop)
                saved_crops += 1
                print("saved bike crop:", filename)
                continue

            # кроп всего мотоцикла
            x1e, y1e, x2e, y2e = expand_bbox(x1, y1, x2, y2, W, H, scale=1.10)
            crop = frame_for_crop[y1e:y2e, x1e:x2e]
            h, w = crop.shape[:2]

            if get_data:
                filename = os.path.join(
                    crops_dir,
                    f"frame{frame_idx}_tid{tid}.jpg"
                )
                cv2.imwrite(filename, crop)
                saved_crops += 1
                print("saved bike crop:", filename)
                continue

            # ---- КРОП ТАБЛИЧКИ, КАК ПРИ СБОРЕ ДАТАСЕТА ----
            px0 = int(0.15 * w)
            px1 = int(0.50 * w)
            py0 = int(0.30 * h)
            py1 = int(0.70 * h)

            plate_crop = crop[py0:py1, px0:px1]

            # для дебага рисуем прямоугольник на полном кадре
            dbg_x1 = x1e + px0
            dbg_x2 = x1e + px1
            dbg_y1 = y1e + py0
            dbg_y2 = y1e + py1
            cv2.rectangle(frame, (dbg_x1, dbg_y1), (dbg_x2, dbg_y2), (0, 255, 0), 2)

            # детекция цифр на plate_crop
            plate_text, plate_conf = detect_plate_number(
                digit_model,
                plate_crop,
                has_plate_class=has_plate_class,
                conf_th=0.25,
            )

            if plate_text:
                rider_id = f"plate_{plate_text}"
            else:
                rider_id = None
                if reid is not None:
                    rider_id = reid.identify(crop)

                if not rider_id:
                    print(f"No plate digits for tid={tid} at t={video_ts:.2f}")
                    continue

            os.makedirs("debug_plates", exist_ok=True)
            debug_name = f"debug_plates/frame{frame_idx}_tid{tid}.jpg"
            cv2.imwrite(debug_name, plate_crop)
            print("PLATE:", debug_name, plate_text, plate_conf)

            os.makedirs("debug_plates", exist_ok=True)
            debug_name = f"debug_plates/frame{frame_idx}_tid{tid}.jpg"
            cv2.imwrite(debug_name, plate_crop)
            print("PLATE:", debug_name, plate_text, plate_conf)

            id_to_rider[tid] = rider_id

            lap_counts[rider_id] = lap_counts.get(rider_id, 0) + 1
            prev_rt = last_rider_time.get(rider_id)
            lap_time = (video_ts - prev_rt) if prev_rt is not None else video_ts
            last_rider_time[rider_id] = video_ts

            log.append({
                "timestamp": round(video_ts, 3),
                "tracker_id": tid,
                "rider": rider_id,
                "plate": plate_text or "",
                "plate_conf": round(plate_conf, 3),
                "lap": lap_counts[rider_id],
                "lap_time": round(lap_time, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int(cx), int(cy)]
            })

        frame = box_annotator.annotate(scene=frame, detections=detects)
        cv2.line(frame, (Lx1, Ly1), (Lx2, Ly2), (255, 255, 255), int(line_width))

        if not get_data:
            y0 = 30
            for idx, (r, lap) in enumerate(sorted(lap_counts.items(), key=lambda x: -x[1])[:10]):
                cv2.putText(frame, f"{r} lap {lap}", (10, y0 + idx * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        writer.write(frame)

    capture.release()
    writer.release()

    if not get_data:
        pd.DataFrame(log).to_csv(output_csv, index=False)
        print(f"Done. {len(log)} events -> {output_csv}")
    else:
        print(f"Done. Saved {saved_crops} bike crops to '{crops_dir}'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/Users/evgenylugin/py/motocross_tracker/merged.mp4",
                        help="Video source (0 for webcam or file path)")
    parser.add_argument("--gallery", default="./gallery", help="Path to gallery folder")
    parser.add_argument("--out_video", default="output.mp4", help="Output video file")
    parser.add_argument("--out_csv", default="laps.csv", help="Output CSV file")

    parser.add_argument(
        "--line",
        default="12%,78%,88%,35%",
        help='Финишная линия: "x1,y1,x2,y2" в пикселях или процентах (например "12%,78%,88%,35%").'
    )
    parser.add_argument(
        "--line_width", type=int, default=20,
        help="Ширина полосы вокруг линии (px), влияет и на визуализацию и на детект пересечения."
    )
    parser.add_argument(
        "--calibrate_line", action="store_true",
        help="Открыть первый кадр и выбрать линию мышкой (две точки, Enter чтобы подтвердить)."
    )
    parser.add_argument(
        "--get-data", "-get-data",
        dest="get_data",
        action="store_true",
        help="Режим сбора данных: сохранять кропы мотоциклов и не считать круги."
    )
    parser.add_argument(
        "--digits_model",
        default="runs/detect/train3/weights/best.pt",
        help="Путь к YOLO модели цифр."
    )
    parser.add_argument(
        "--digits_has_plate",
        action="store_true",
        help="Указать, если в модели цифр есть класс plate с id=0."
    )

    args = parser.parse_args()
    main(
        args.video,
        args.gallery,
        args.out_video,
        args.out_csv,
        args.line,
        args.line_width,
        args.calibrate_line,
        get_data=args.get_data,
        digits_model_path=args.digits_model,
        has_plate_class=args.digits_has_plate,
    )
