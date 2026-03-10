import cv2
import numpy as np
import time
from light_source import LightSource
from typing import List
from kalman_tracker import KalmanTracker

# --- PARAMETRY ---
MIN_RADIUS = 3
RADIUS = 12
MIN_DIST = 50
INTENSITY = 230
THRESHOLD = 180
MIN_CONFIDENCE = 0.4
AUTH = [True, False, True, False]
FREQ = 24
SIZE = 60
DRIFT_COEF = 1
DETECTION_TIMEOUT = 1500


def video():
    cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("no stream")
    #     exit()

    sources: List[LightSource] = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = int((time.time() - start_time) * 1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_points = []
        for c in contours:
            (x, y), r = cv2.minEnclosingCircle(c)
            if MIN_RADIUS < r < RADIUS * 2.5:
                if gray[int(y), int(x)] > INTENSITY:
                    detected_points.append({'x': int(x), 'y': int(y)})

        # tracking kalman filter
        used_points = set()
        for s in sources:
            # predykcja
            pred_x, pred_y = s.kf.predict()

            # Pobieramy prędkość wyliczoną przez Kalmana dla dynamicznego dystansu
            _, _, vel_x, vel_y = s.kf.get_state()
            velocity = (vel_x ** 2 + vel_y ** 2) ** 0.5
            dynamic_min_dist = max(MIN_DIST, velocity * 2.5)

            # Do logiki "kapsuły" używamy obecnej pozycji (ax, ay) i przewidzianej (bx, by)
            ax, ay = s.x, s.y
            bx, by = pred_x, pred_y

            found_match = None
            current_min_dist = dynamic_min_dist
            match_idx = -1

            # 2. SZUKANIE DOPASOWANIA
            for i, p in enumerate(detected_points):
                if i in used_points: continue
                px, py = p['x'], p['y']

                dx, dy = bx - ax, by - ay
                mag_sq = dx * dx + dy * dy

                if mag_sq == 0:
                    dist = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
                else:
                    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / mag_sq))
                    closest_x = ax + t * dx
                    closest_y = ay + t * dy
                    dist = ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

                if dist < current_min_dist:
                    current_min_dist = dist
                    found_match = p
                    match_idx = i

            # 3. AKTUALIZACJA STANU
            if found_match:
                # Jeśli znaleźliśmy plamkę światła, korygujemy Kalmana pomiarem
                s.kf.update(found_match['x'], found_match['y'])
                s.last_seen_ts = ts
                used_points.add(match_idx)

            # Niezależnie od tego, czy znaleźliśmy punkt w tej klatce, czy nie:
            # Aktualizujemy obiekt LightSource wygładzonymi danymi z Kalmana!
            # Dzięki temu w przypadku braku detekcji przez 1-2 klatki, obiekt "pojedzie" siłą inercji.
            kx, ky, kdx, kdy = s.kf.get_state()
            s.x, s.y = kx, ky
            s.dx, s.dy = kdx, kdy  # Nadpisujemy prędkość oryginalną tą stabilniejszą z Kalmana

            if found_match:
                s.add_record({'timestamp': ts, 'state': True, 'x': int(kx), 'y': int(ky)})
            else:
                s.add_record({'timestamp': ts, 'state': False, 'x': int(kx), 'y': int(ky)})

        # --- DODAWANIE NOWYCH ---
        for i, p in enumerate(detected_points):
            if i not in used_points:
                new_s = LightSource(SIZE, FREQ, AUTH, DRIFT_COEF, p['x'], p['y'])
                new_s.last_seen_ts = ts
                sources.append(new_s)

        # --- USUWANIE STARYCH ---
        sources = [s for s in sources if (ts - s.last_seen_ts) < (DETECTION_TIMEOUT if s.match_score > 0.4 else 300)]

        # --- RYSOWANIE I DEBUG ---
        cv2.drawContours(frame, contours, -1, (200, 200, 200), 1)

        for s in sources:
            if s.confidence < 0.1:
                continue

            is_auth = s.match_score > 0.75 and s.confidence > MIN_CONFIDENCE
            color = (0, 255, 0) if is_auth else (0, 0, 255)

            # Korzystamy z uaktualnionych, wygładzonych danych
            cx, cy = int(s.x), int(s.y)
            bx, by = int(s.x + s.dx), int(s.y + s.dy)

            velocity = (s.dx ** 2 + s.dy ** 2) ** 0.5
            d_dist = int(max(MIN_DIST, velocity * 2.5))

            # Rysujemy "korytarz" szukania (predykcję Kalmana)
            cv2.line(frame, (cx, cy), (bx, by), (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), d_dist, (0, 150, 150), 1)
            cv2.circle(frame, (bx, by), d_dist, (0, 255, 255), 1)

            # Główne kółko (Wygładzona pozycja ze wskaźnika)
            cv2.circle(frame, (cx, cy), RADIUS, color, 2)

            q_bits = s.get_quantized_bits()
            q_str = "".join(map(str, q_bits[-8:])) if q_bits else "0"
            cv2.putText(frame, f"ID:{id(s) % 100} BITS:{q_str}", (cx - 40, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            cv2.putText(frame, f"M:{s.match_score:.2f} V:{velocity:.1f}", (cx - 40, cy - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

        cv2.imshow("Optical Auth Receiver", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video()