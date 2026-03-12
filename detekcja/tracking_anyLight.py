import cv2
import numpy as np
import time
from typing import List
from zwykle_swiatlo import Light
import random
from scipy.optimize import linear_sum_assignment


# paramtery
MIN_RADIUS = 3
INTENSITY = 250
THRESHOLD = 245
RADIUS = 20
MIN_DIST = 100
SIZE = 60
DRIFT_COEF = 1
DETECTION_TIMEOUT = 1500
#auto.exposire

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no stream")
        exit()

    sources: List[Light] = []
    start_time = time.time()
    ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = int((time.time() - start_time) * 1000)  # timestamp w ms
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(thresh)
        # detekcja punktów świetlnych
        detected_points = []
        for c in contours:
            (x, y), r = cv2.minEnclosingCircle(c)
            ix, iy = int(x), int(y)
            if MIN_RADIUS < r < RADIUS * 2.5:
                if 0 <= ix < width and 0 <= iy < height:
                    if gray[iy, ix] > INTENSITY:
                        detected_points.append({'x': ix, 'y': iy, 'r': r})

        # najpierw wykonujemy predykcję dla wysztskich świateł zidentyfikowanych wczesniej
        # i liczymy ich dopuszczalne promienie
        predictions = []
        thresholds = []
        for s in sources:
            print(s)
            pred_x, pred_y = s.kalman.predict()
            print(f"predykcja{pred_x}, {pred_y}")
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

                # Udało nam się go zidentyfikowaac
                unmatched_trackers.remove(t)
                unmatched_detections.remove(d)

        #jesli nie namiezylismy go to uatwaimy mu wspolrzedne gdzie "może" sie znajdować
        for t in unmatched_trackers:
            s = sources[t]
            kx, ky, kdx, kdy = s.kalman.get_state()
            s.x, s.y, s.dx, s.dy = kx, ky, kdx, kdy

        # dodawanie nowych punktów
        for d in unmatched_detections:
            p = detected_points[d]
            while True:
                new_ID = random.randint(0, 100)
                if new_ID not in ids:
                    ids.add(new_ID)
                    break

            new_s = Light(new_ID, ts, p['x'] , p['y'])
            sources.append(new_s)

        sources = [s for s in sources if (ts - s.last_seen_ts) < 300]
        helper_ids = {s.ID for s in sources}
        print(helper_ids)
        ids = helper_ids

        cv2.drawContours(frame, contours, -1, (200, 200, 200), 1)

        for s in sources:
            color = (0, 255, 0)
            cx, cy = int(s.x), int(s.y)
            bx, by = int(s.x + s.dx), int(s.y + s.dy)

            velocity = (s.dx ** 2 + s.dy ** 2) ** 0.5
            d_dist = int(max(MIN_DIST, velocity * 2.5))

            cv2.line(frame, (cx, cy), (bx, by), (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), d_dist, (0, 150, 150), 1)
            cv2.circle(frame, (bx, by), d_dist, (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), RADIUS, color, 2)
            cv2.putText(frame, f"ID:{s.ID}", (cx - 40, cy - 60), 1, 0.7, (0, 255, 255), 1)

        cv2.imshow("Optical Auth Receiver", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()