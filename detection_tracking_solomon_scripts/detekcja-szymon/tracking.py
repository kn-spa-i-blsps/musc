import itertools
import time
from typing import List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from blink_protocol import PACKET_DUR_MS
from normal_light import Light

# parametry detekcji
MIN_RADIUS = 3
MAX_RADIUS = 50
INTENSITY = 250
THRESHOLD = 245
RADIUS = 20

# parametry śledzenia
MIN_DIST = 200          # minimalny promień bramkowania [px]
GATE_VEL_COEF = 2.5     # bramka rośnie z prędkością trackera
# tracker musi przeżyć najdłuższą możliwą ciemność w ramce
# (ID=0 -> 20 zer pod rząd = ~667 ms zgaszonej diody)
TRACK_TIMEOUT_MS = int(1.5 * PACKET_DUR_MS)

MIN_CONFIDENCE = 0.4


def detect_lights(gray: np.ndarray) -> Tuple[list, list]:
    """Zwraca listę jasnych punktów [(x, y, r), ...] oraz kontury do podglądu."""
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    points = []
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)
        ix, iy = int(x), int(y)
        if (MIN_RADIUS < r < MAX_RADIUS
                and 0 <= ix < width and 0 <= iy < height
                and gray[iy, ix] > INTENSITY):
            points.append((ix, iy, r))
    return points, contours


def match_detections(sources: List[Light], detections: list, ts: int):
    """
    Predykcja Kalmana + algorytm węgierski. Aktualizuje trackery na miejscu
    i zwraca indeksy detekcji, które nie pasują do żadnego trackera.
    """
    n_trk, n_det = len(sources), len(detections)

    predictions = np.empty((n_trk, 2), dtype=np.float32)
    gates = np.empty(n_trk, dtype=np.float32)
    for i, s in enumerate(sources):
        predictions[i] = s.kalman.predict()
        _, _, vx, vy = s.kalman.get_state()
        gates[i] = max(MIN_DIST, GATE_VEL_COEF * float(np.hypot(vx, vy)))

    matched_trk = set()
    matched_det = set()
    if n_trk and n_det:
        det_xy = np.array([(p[0], p[1]) for p in detections], dtype=np.float32)
        cost = cdist(predictions, det_xy)               # macierz odległości bez pętli
        rows, cols = linear_sum_assignment(cost)
        for t, d in zip(rows, cols):
            if cost[t, d] <= gates[t]:                  # bramkowanie dopasowania
                s = sources[t]
                x, y, _ = detections[d]
                s.kalman.update(x, y)
                s.x, s.y, s.dx, s.dy = s.kalman.get_state()
                s.add_record({'timestamp': ts, 'state': True})
                matched_trk.add(t)
                matched_det.add(d)

    # niedopasowane trackery: światło zgaszone (bit 0) albo chwilowo zgubione
    for t in range(n_trk):
        if t not in matched_trk:
            s = sources[t]
            s.x, s.y, s.dx, s.dy = s.kalman.get_state()
            s.add_record({'timestamp': ts, 'state': False})

    return [d for d in range(n_det) if d not in matched_det]


def draw_overlay(frame: np.ndarray, sources: List[Light], contours: list) -> None:
    cv2.drawContours(frame, contours, -1, (200, 200, 200), 1)

    for s in sources:
        identified = s.drone_id is not None and s.confidence >= MIN_CONFIDENCE
        color = (0, 255, 0) if identified else (0, 0, 255)
        cx, cy = int(s.x), int(s.y)
        bx, by = int(s.x + s.dx), int(s.y + s.dy)

        velocity = float(np.hypot(s.dx, s.dy))
        gate = int(max(MIN_DIST, GATE_VEL_COEF * velocity))

        cv2.line(frame, (cx, cy), (bx, by), (0, 255, 255), 1)
        cv2.circle(frame, (cx, cy), gate, (0, 150, 150), 1)
        cv2.circle(frame, (cx, cy), RADIUS, color, 2)

        if identified:
            label = f"DRON:{s.drone_id}"
        elif s.drone_id is not None:
            label = f"DRON?:{s.drone_id}"
        else:
            label = f"TRK:{s.ID}"
        bits = "".join(map(str, s.get_quantized_bits()[-10:]))
        cv2.putText(frame, f"{label} C:{s.confidence:.2f}", (cx - 40, cy - 60),
                    1, 0.8, color, 1)
        cv2.putText(frame, f"BITS:{bits} V:{velocity:.1f}", (cx - 40, cy - 40),
                    1, 0.7, (0, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no stream")
        return

    # stała, krótka ekspozycja: dioda ma być jedynym prześwietlonym punktem
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 50)
    cap.set(cv2.CAP_PROP_FPS, 60)

    sources: List[Light] = []
    track_id_gen = itertools.count(1)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = int((time.time() - start_time) * 1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections, contours = detect_lights(gray)
        unmatched = match_detections(sources, detections, ts)

        for d in unmatched:
            x, y, _ = detections[d]
            sources.append(Light(next(track_id_gen), ts, x, y))

        sources = [s for s in sources if (ts - s.last_seen_ts) < TRACK_TIMEOUT_MS]

        draw_overlay(frame, sources, contours)
        cv2.imshow("Optical Auth Receiver", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
