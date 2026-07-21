"""
Wariant tracking.py bez wymuszania ekspozycji kamery (automatyczna ekspozycja).
Przydatny do testów z dowolnym źródłem światła. Cała logika detekcji,
śledzenia i dekodowania ID jest współdzielona z tracking.py.
"""

import itertools
import time
from typing import List

import cv2

from normal_light import Light
from tracking import TRACK_TIMEOUT_MS, detect_lights, draw_overlay, match_detections


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no stream")
        return

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
        cv2.imshow("Optical Auth Receiver (any light)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
