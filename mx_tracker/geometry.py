from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# rows × cols for each supported zone count; zones numbered left-to-right, top-to-bottom
_ZONE_GRID: dict[int, tuple[int, int]] = {
    1: (1, 1),
    2: (2, 1),   # top / bottom
    4: (2, 2),
    9: (3, 3),
    32: (4, 8),
}


def zone_crop(image: np.ndarray, n_zones: int, selected: list[int]) -> np.ndarray:
    """Return the sub-image covering the union of the selected zones.

    Zones are numbered 1-based, left-to-right then top-to-bottom.
    Falls back to the full image when n_zones=1 or selected is empty.
    """
    if n_zones <= 1 or not selected or n_zones not in _ZONE_GRID:
        return image
    rows, cols = _ZONE_GRID[n_zones]
    h, w = image.shape[:2]
    zone_h = h / rows
    zone_w = w / cols

    min_r = min_c = 10**9
    max_r = max_c = 0
    for z in selected:
        z0 = z - 1
        r, c = z0 // cols, z0 % cols
        if r < min_r:
            min_r = r
        if r > max_r:
            max_r = r
        if c < min_c:
            min_c = c
        if c > max_c:
            max_c = c

    y1 = int(min_r * zone_h)
    y2 = min(h, int((max_r + 1) * zone_h))
    x1 = int(min_c * zone_w)
    x2 = min(w, int((max_c + 1) * zone_w))
    cropped = image[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else image


def expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
    scale: float = 1.2,
) -> tuple[int, int, int, int]:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = (x2 - x1) * scale
    height = (y2 - y1) * scale
    nx1 = max(0, int(cx - width / 2.0))
    ny1 = max(0, int(cy - height / 2.0))
    nx2 = min(frame_width, int(cx + width / 2.0))
    ny2 = min(frame_height, int(cy + height / 2.0))
    return nx1, ny1, nx2, ny2


def parse_line_arg(arg: str, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    def _point(token: str, size: int) -> int:
        token = token.strip()
        if token.endswith("%"):
            return int(round(float(token[:-1]) * size / 100.0))
        return int(round(float(token)))

    x1, y1, x2, y2 = [chunk.strip() for chunk in arg.split(",")]
    return (
        _point(x1, frame_width),
        _point(y1, frame_height),
        _point(x2, frame_width),
        _point(y2, frame_height),
    )


def to_percent_str(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
) -> str:
    def pct(value: int, size: int) -> str:
        return f"{(value / size) * 100:.2f}%"

    return ",".join(
        (
            pct(x1, frame_width),
            pct(y1, frame_height),
            pct(x2, frame_width),
            pct(y2, frame_height),
        )
    )


def point_line_side_and_dist(
    point: tuple[int, int],
    line_a: tuple[int, int],
    line_b: tuple[int, int],
) -> tuple[float, float]:
    a = np.array(line_a, dtype=float)
    b = np.array(line_b, dtype=float)
    p = np.array(point, dtype=float)
    ab = b - a
    ap = p - a
    denom = float(ab @ ab) + 1e-9
    projection = max(0.0, min(1.0, float(ap @ ab) / denom))
    nearest = a + projection * ab
    side = float(np.sign(ab[0] * ap[1] - ab[1] * ap[0]))
    distance = float(np.linalg.norm(p - nearest))
    return side, distance


def _orient(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    )


def segments_intersect(
    p1: tuple[int, int],
    p2: tuple[int, int],
    q1: tuple[int, int],
    q2: tuple[int, int],
) -> bool:
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    if (o1 == 0 and _on_segment(p1, p2, q1)) or (o2 == 0 and _on_segment(p1, p2, q2)):
        return True
    if (o3 == 0 and _on_segment(q1, q2, p1)) or (o4 == 0 and _on_segment(q1, q2, p2)):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


@dataclass(slots=True)
class CrossingDecision:
    crossed: bool
    direction: str | None
    distance: float


def detect_crossing(
    prev_point: tuple[int, int] | None,
    cur_point: tuple[int, int],
    line_a: tuple[int, int],
    line_b: tuple[int, int],
    line_width: int,
    direction_mode: str,
) -> CrossingDecision:
    if prev_point is None:
        _, cur_dist = point_line_side_and_dist(cur_point, line_a, line_b)
        return CrossingDecision(False, None, cur_dist)

    side_prev, dist_prev = point_line_side_and_dist(prev_point, line_a, line_b)
    side_cur, dist_cur = point_line_side_and_dist(cur_point, line_a, line_b)
    inside_prev = dist_prev <= line_width / 2.0
    inside_cur = dist_cur <= line_width / 2.0
    crossed = False
    if (side_prev * side_cur < 0) and (min(dist_prev, dist_cur) <= line_width):
        crossed = True
    elif inside_prev != inside_cur:
        crossed = True
    elif segments_intersect(prev_point, cur_point, line_a, line_b):
        crossed = True
    direction = None
    if crossed:
        if side_cur < side_prev:
            direction = "left_to_right"
        elif side_cur > side_prev:
            direction = "right_to_left"
        else:
            direction = "left_to_right"
        if direction != direction_mode:
            crossed = False
    return CrossingDecision(crossed, direction, min(dist_prev, dist_cur))


def pick_line_on_frame(frame: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    window_name = "Pick line: 2 clicks, Enter to confirm, R to reset, Esc to cancel"
    points: list[tuple[int, int]] = []
    preview = frame.copy()

    def callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal preview
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 2:
            return
        points.append((x, y))
        cv2.circle(preview, (x, y), 5, (0, 255, 255), -1)
        if len(points) == 2:
            cv2.line(preview, points[0], points[1], (0, 255, 255), 2)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback)
    try:
        while True:
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and len(points) == 2:
                return points[0], points[1]
            if key == 27:
                return None
            if key in (ord("r"), ord("R")):
                points.clear()
                preview = frame.copy()
    finally:
        cv2.destroyWindow(window_name)
