import cv2
import numpy as np
import time
import random
from typing import List
from scipy.optimize import linear_sum_assignment
from normal_light import Light

# Parametry
MIN_RADIUS = 6
INTENSITY = 250
THRESHOLD = 245
RADIUS = 20
MIN_DIST = 200
DETECTION_TIMEOUT = 1500
MIN_CONFIDENCE = 0.4
TRACKER_MAX_AGE_MS = 300
DEBUG = False


def init_capture() -> cv2.VideoCapture:
    #Otwiere kamere i ustawia jej parametry
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no stream")
        exit()
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 50)
    return cap


def detect_lights(gray: np.ndarray) -> tuple[list, list]:
    #Wykrywa znalezione punkty i zwraca liste {x,y,r}
    height, width = gray.shape
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_points = []
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)
        ix, iy = int(x), int(y)
        if MIN_RADIUS < r < RADIUS * 2.5:
            if 0 <= ix < width and 0 <= iy < height:
                if gray[iy, ix] > INTENSITY:
                    detected_points.append({'x': ix, 'y': iy, 'r': r})

    return contours, detected_points


def build_cost_matrix(sources: list, detected_points: list) -> tuple[np.ndarray, list, list]:
    # najpierw wykonujemy predykcję dla wysztskich świateł zidentyfikowanych wczesniej
    # i liczymy ich dopuszczalne promienie
    predictions = []
    thresholds = []

    for s in sources:
        pred_x, pred_y = s.kalman.predict()
        predictions.append((pred_x, pred_y))

        _, _, vel_x, vel_y = s.kalman.get_state()
        velocity = (vel_x ** 2 + vel_y ** 2) ** 0.5
        # Dynamiczny próg (bramkowanie) dla każdego trackera osobno
        thresholds.append(max(MIN_DIST, velocity * 2.5))

    # budujemy macierz kosztów
    # (dla każdego juz znalezionego punktu obliczamy jego odleglosc źródeł światła które teraz poznalismy)
    num_trackers = len(sources)
    num_detections = len(detected_points)
    cost_matrix = np.zeros((num_trackers, num_detections))

    for t in range(num_trackers):
        for d in range(num_detections):
            pred_x, pred_y = predictions[t]
            det_x, det_y = detected_points[d]['x'], detected_points[d]['y']
            dist = ((det_x - pred_x) ** 2 + (det_y - pred_y) ** 2) ** 0.5
            cost_matrix[t, d] = dist

    return cost_matrix, thresholds, predictions


def match_trackers(
    sources: list,
    detected_points: list,
    cost_matrix: np.ndarray,
    thresholds: list,
    ts: int,
) -> tuple[set, set]:

    num_trackers = len(sources)
    num_detections = len(detected_points)

    # odpalamy Algorytm Węgierski
    # (znajduje globalnie optymalne dopasowanie)
    if num_trackers > 0 and num_detections > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        row_ind, col_ind = [], []

    # linear_sum zwraca krotki dopasowan
    # zbiory pomocnicze, żeby wiedzieć, kto został bez pary
    unmatched_trackers = set(range(num_trackers))
    unmatched_detections = set(range(num_detections))

    # sprawdzamy wyniki dopasowan
    for t, d in zip(row_ind, col_ind):
        # sprawdzamy czy dobrze to przypisało
        if cost_matrix[t, d] <= thresholds[t]:
            # jak tu weszlismy to znaczy że zidentyfikowalismy gdzie znajduje sie obecnie nasz wczesniejszy punkt
            s = sources[t]
            p = detected_points[d]

            # Aktualizacja Filtru (widzimy obiekt)
            s.kalman.update(p['x'], p['y'])
            s.last_seen_ts = ts

            kx, ky, kdx, kdy = s.kalman.get_state()
            s.x, s.y, s.dx, s.dy = kx, ky, kdx, kdy
            s.add_record({'timestamp': ts, 'state': True, 'x': p['x'], 'y': p['y']})
            # Udało nam się go zidentyfikowaac
            unmatched_trackers.discard(t)
            unmatched_detections.discard(d)

    #jesli nie namiezylismy go to uatwaimy mu wspolrzedne gdzie "może" sie znajdować
    for t in unmatched_trackers:
        s = sources[t]
        kx, ky, kdx, kdy = s.kalman.get_state()
        s.x, s.y, s.dx, s.dy = kx, ky, kdx, kdy
        s.add_record({'timestamp': ts, 'state': False, 'x': int(s.x), 'y': int(s.y)})

    return unmatched_trackers, unmatched_detections


def update_sources(
    sources: list,
    unmatched_detections: set,
    detected_points: list,
    ts: int,
    ids: set,
) -> tuple[list, set]:
    # dodawanie nowych punktów
    for d in unmatched_detections:
        p = detected_points[d]

        while True:
            new_ID = random.randint(0, 100)
            if new_ID not in ids:
                ids.add(new_ID)
                break

        new_s = Light(new_ID, ts, p['x'], p['y'])
        sources.append(new_s)

    sources = [s for s in sources if (ts - s.last_seen_ts) < TRACKER_MAX_AGE_MS]
    ids = {s.ID for s in sources}

    return sources, ids


def draw_overlay(frame: np.ndarray, sources: list, contours) -> None:
    cv2.drawContours(frame, contours, -1, (200, 200, 200), 1)

    for s in sources:
        if s.confidence < 0.1:
            continue

        is_auth = s.match_score > 0.75 and s.confidence > MIN_CONFIDENCE
        color = (0, 255, 0) if is_auth else (0, 0, 255)

        cx, cy = int(s.x), int(s.y)
        bx, by = int(s.x + s.dx), int(s.y + s.dy)

        velocity = (s.dx ** 2 + s.dy ** 2) ** 0.5
        d_dist = int(max(MIN_DIST, velocity * 2.5))

        cv2.line(frame, (cx, cy), (bx, by), (0, 255, 255), 1)
        cv2.circle(frame, (cx, cy), d_dist, (0, 150, 150), 1)
        cv2.circle(frame, (bx, by), d_dist, (0, 255, 255), 1)
        cv2.circle(frame, (cx, cy), RADIUS, color, 2)

        q_bits = s.get_quantized_bits()
        q_str = "".join(map(str, q_bits[-8:]))
        cv2.putText(frame, f"ID:{s.ID} BITS:{q_str}", (cx - 40, cy - 60), 1, 0.7, (0, 255, 255), 1)
        cv2.putText(frame, f"M:{s.match_score:.2f} V:{velocity:.1f}", (cx - 40, cy - 40), 1, 0.7, color, 1)


def main():
    cap = init_capture()
    sources: list = []
    ids: set = set()
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = int((time.time() - start_time) * 1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours, detected_points = detect_lights(gray)
        cost_matrix, thresholds, _ = build_cost_matrix(sources, detected_points)
        _, unmatched_d = match_trackers(sources, detected_points, cost_matrix, thresholds, ts)
        sources, ids = update_sources(sources, unmatched_d, detected_points, ts, ids)
        draw_overlay(frame, sources, contours)

        cv2.imshow("Optical Auth Receiver", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()