import math
import random
import time
from typing import List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from normal_light import Light

#Skrypt do oceny odległości (odl.py) połączony z wykrywaniem świateł

# =========================
# PARAMETRY DETEKCJI ŚWIATEŁ
# =========================
MIN_RADIUS = 3
INTENSITY = 250
THRESHOLD = 245
RADIUS = 20
MIN_DIST = 200
MIN_CONFIDENCE = 0.4
TRACK_MAX_AGE_MS = 300

# =========================
# PARAMETRY WYBORU PUNKTÓW DO ODLEGŁOŚCI
# =========================
BOK_KWADRATU_M = 0.4
MIN_AUTH_MATCH_SCORE = 0.75
MAX_GROUP_SPAN_PX = 1200  # maks. rozmiar grupy 2-4 punktów

# ==========================================================
# KONFIGURACJA KAMERY (PAMIETAJCIE ZEBY SOBIE ZMIENIC!!!)
# ==========================================================
SZEROKOSC = 1920
WYSOKOSC = 1080

CX = SZEROKOSC / 2.0
CY = WYSOKOSC / 2.0

FX = 458
FY = 458

MACIERZ_K = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
], dtype=np.float32)

DYSTORSJA = np.array([0.0, 0.0, 0, 0, 0], dtype=np.float32)


def oblicz_pozycje_drona(punkty_2d, bok_a):
    """
    punkty_2d: lista krotek (x, y) w pikselach
    bok_a: rzeczywisty bok kwadratu w metrach (np. 0.21)
    Zakładana kolejność punktów: LD, LG, PG, PD
    Zwraca odległość od obiektywu do ŚRODKA kwadratu.
    """
    n = len(punkty_2d)
    punkty_2d = np.array(punkty_2d, dtype=np.float32)
    polowa = bok_a / 2.0

    if n == 4:
        punkty_3d = np.array([
            [-polowa,  polowa, 0],
            [-polowa, -polowa, 0],
            [ polowa, -polowa, 0],
            [ polowa,  polowa, 0]
        ], dtype=np.float32)

        sukces, rvec, tvec = cv2.solvePnP(punkty_3d, punkty_2d, MACIERZ_K, DYSTORSJA)
        if sukces:
            return float(np.linalg.norm(tvec)), "PnP (4 punkty)"

    elif n == 3:
        punkty_3d = np.array([
            [-polowa,  polowa, 0],
            [-polowa, -polowa, 0],
            [ polowa, -polowa, 0]
        ], dtype=np.float32)

        sukces, rvecs, tvecs = cv2.solveP3P(
            punkty_3d,
            punkty_2d,
            MACIERZ_K,
            DYSTORSJA,
            flags=cv2.SOLVEPNP_P3P
        )

        if sukces:
            poprawne_odleglosci = []
            for tvec in tvecs:
                if tvec[2][0] > 0:
                    poprawne_odleglosci.append(float(np.linalg.norm(tvec)))

            if poprawne_odleglosci:
                opcje_tekst = " lub ".join([f"{d:.2f}m" for d in poprawne_odleglosci])
                return poprawne_odleglosci[0], f"P3P (Możliwe opcje: {opcje_tekst})"

    elif n == 2:
        p1, p2 = punkty_2d[0], punkty_2d[1]
        p = np.linalg.norm(p1 - p2)
        if p == 0:
            return None, "Błąd: punkty nakładają się"

        f_avg = (FX + FY) / 2.0
        Z = (bok_a * f_avg) / p

        u_c = (p1[0] + p2[0]) / 2.0
        v_c = (p1[1] + p2[1]) / 2.0

        X = (u_c - CX) * Z / FX
        Y = (v_c - CY) * Z / FY

        odleglosc = float(np.sqrt(X**2 + Y**2 + Z**2))
        return odleglosc, "Proporcja (2 punkty - Środek krawędzi)"

    return None, "Błąd: Za mało punktów lub nieudane obliczenia"


def order_square_points_ld_lg_pg_pd(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Porządkuje 4 punkty do kolejności: LD, LG, PG, PD.
    Zakładamy obraz OpenCV: Y rośnie w dół.
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) != 4:
        return list(points)

    idx_y = np.argsort(pts[:, 1])
    top = pts[idx_y[:2]]
    bottom = pts[idx_y[2:]]

    top = top[np.argsort(top[:, 0])]      # lewy-góra, prawy-góra
    bottom = bottom[np.argsort(bottom[:, 0])]  # lewy-dół, prawy-dół

    lg = tuple(map(int, top[0]))
    pg = tuple(map(int, top[1]))
    ld = tuple(map(int, bottom[0]))
    pd = tuple(map(int, bottom[1]))
    return [ld, lg, pg, pd]


def wybierz_punkty_do_odleglosci(auth_sources: List[Light]) -> List[Tuple[int, int]]:
    """
    Wybiera 2-4 punkty spełniające kod. Bierze te najbardziej wiarygodne i zwarte przestrzennie.
    """
    if len(auth_sources) < 2:
        return []

    posortowane = sorted(
        auth_sources,
        key=lambda s: (s.match_score, s.confidence),
        reverse=True
    )

    wybrane: List[Light] = []
    for s in posortowane:
        if not wybrane:
            wybrane.append(s)
            continue

        kandydaci = wybrane + [s]
        xs = [p.x for p in kandydaci]
        ys = [p.y for p in kandydaci]
        span = max(max(xs) - min(xs), max(ys) - min(ys))

        if span <= MAX_GROUP_SPAN_PX:
            wybrane.append(s)

        if len(wybrane) == 4:
            break

    if len(wybrane) < 2:
        return []

    punkty = [(int(s.x), int(s.y)) for s in wybrane[:4]]

    if len(punkty) == 4:
        return order_square_points_ld_lg_pg_pd(punkty)

    if len(punkty) == 3:
        # z grubsza: lewy-dół, lewy-góra, prawy-góra
        pts = np.array(punkty, dtype=np.float32)
        idx_y = np.argsort(pts[:, 1])
        top_two = pts[idx_y[:2]]
        bottom_one = pts[idx_y[2:]]
        top_two = top_two[np.argsort(top_two[:, 0])]
        lg = tuple(map(int, top_two[0]))
        pg = tuple(map(int, top_two[1]))
        ld = tuple(map(int, bottom_one[0]))
        return [ld, lg, pg]

    if len(punkty) == 2:
        return punkty

    return []


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no stream")
        return

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 50)

    sources: List[Light] = []
    start_time = time.time()
    ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = int((time.time() - start_time) * 1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_points = []
        for c in contours:
            (x, y), r = cv2.minEnclosingCircle(c)
            ix, iy = int(x), int(y)
            if MIN_RADIUS < r < RADIUS * 2.5 and 0 <= ix < width and 0 <= iy < height:
                if gray[iy, ix] > INTENSITY:
                    detected_points.append({'x': ix, 'y': iy, 'r': r})

        predictions = []
        thresholds = []
        for s in sources:
            pred_x, pred_y = s.kalman.predict()
            predictions.append((pred_x, pred_y))

            _, _, vel_x, vel_y = s.kalman.get_state()
            velocity = math.sqrt(vel_x ** 2 + vel_y ** 2)
            thresholds.append(max(MIN_DIST, velocity * 2.5))

        num_trackers = len(sources)
        num_detections = len(detected_points)
        cost_matrix = np.zeros((num_trackers, num_detections))

        for t in range(num_trackers):
            for d in range(num_detections):
                pred_x, pred_y = predictions[t]
                det_x, det_y = detected_points[d]['x'], detected_points[d]['y']
                dist = math.sqrt((det_x - pred_x) ** 2 + (det_y - pred_y) ** 2)
                cost_matrix[t, d] = dist

        if num_trackers > 0 and num_detections > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []

        unmatched_trackers = set(range(num_trackers))
        unmatched_detections = set(range(num_detections))

        for t, d in zip(row_ind, col_ind):
            if cost_matrix[t, d] <= thresholds[t]:
                s = sources[t]
                p = detected_points[d]

                s.kalman.update(p['x'], p['y'])
                s.last_seen_ts = ts

                kx, ky, kdx, kdy = s.kalman.get_state()
                s.x, s.y, s.dx, s.dy = kx, ky, kdx, kdy
                s.add_record({'timestamp': ts, 'state': True, 'x': p['x'], 'y': p['y']})

                unmatched_trackers.remove(t)
                unmatched_detections.remove(d)

        for t in unmatched_trackers:
            s = sources[t]
            kx, ky, kdx, kdy = s.kalman.get_state()
            s.x, s.y, s.dx, s.dy = kx, ky, kdx, kdy
            s.add_record({'timestamp': ts, 'state': False, 'x': int(s.x), 'y': int(s.y)})

        for d in unmatched_detections:
            p = detected_points[d]
            while True:
                new_id = random.randint(0, 1000)
                if new_id not in ids:
                    ids.add(new_id)
                    break

            new_s = Light(new_id, ts, p['x'], p['y'])
            sources.append(new_s)

        sources = [s for s in sources if (ts - s.last_seen_ts) < TRACK_MAX_AGE_MS]
        ids = {s.ID for s in sources}

        cv2.drawContours(frame, contours, -1, (200, 200, 200), 1)

        auth_sources = []
        for s in sources:
            if s.confidence < 0.1:
                continue

            is_auth = s.match_score > MIN_AUTH_MATCH_SCORE and s.confidence > MIN_CONFIDENCE
            if is_auth:
                auth_sources.append(s)

            color = (0, 255, 0) if is_auth else (0, 0, 255)
            cx, cy = int(s.x), int(s.y)
            bx, by = int(s.x + s.dx), int(s.y + s.dy)

            velocity = math.sqrt(s.dx ** 2 + s.dy ** 2)
            d_dist = int(max(MIN_DIST, velocity * 2.5))

            cv2.line(frame, (cx, cy), (bx, by), (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), d_dist, (0, 150, 150), 1)
            cv2.circle(frame, (bx, by), d_dist, (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), RADIUS, color, 2)

            q_bits = s.get_quantized_bits()
            q_str = "".join(map(str, q_bits[-8:]))
            cv2.putText(frame, f"ID:{s.ID} BITS:{q_str}", (cx - 40, cy - 60), 1, 0.7, (0, 255, 255), 1)
            cv2.putText(frame, f"M:{s.match_score:.2f} C:{s.confidence:.2f}", (cx - 40, cy - 40), 1, 0.7, color, 1)

        punkty_do_pnp = wybierz_punkty_do_odleglosci(auth_sources)
        distance_text = "Brak wystarczającej liczby poprawnych punktów"

        if len(punkty_do_pnp) >= 2:
            odleglosc, info = oblicz_pozycje_drona(punkty_do_pnp, BOK_KWADRATU_M)
            if odleglosc is not None:
                distance_text = f"Odległość: {odleglosc:.2f} m | {info} | pkt={len(punkty_do_pnp)}"

                for i, (px, py) in enumerate(punkty_do_pnp):
                    cv2.circle(frame, (px, py), 12, (255, 255, 0), 2)
                    cv2.putText(frame, f"P{i+1}", (px + 8, py - 8), 1, 0.8, (255, 255, 0), 2)
            else:
                distance_text = f"Nie udało się policzyć odległości | {info}"

        cv2.putText(frame, distance_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Optical Auth Receiver + Distance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
