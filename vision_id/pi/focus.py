"""
Pomoc przy ustawianiu ostrości obiektywu M12.

Wypisuje ostrość środka kadru (wariancja laplasjanu) kilka razy na sekundę.
Kręć obiektywem powoli: liczba ma dojść do maksimum. Ctrl+C kończy.
Celuj w coś z detalami w odległości, na jaką chcesz ostrzyć (najlepiej daleko, za oknem).
"""
import time

import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
cam.configure(cam.create_video_configuration(main={"size": (1332, 990), "format": "RGB888"}))
cam.start()
time.sleep(1)
best = 0.0
try:
    while True:
        g = cam.capture_array("main").mean(axis=2)
        h, w = g.shape
        c = g[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        lap = 4 * c[1:-1, 1:-1] - c[:-2, 1:-1] - c[2:, 1:-1] - c[1:-1, :-2] - c[1:-1, 2:]
        s = float(lap.var())
        best = max(best, s)
        bar = "#" * int(40 * s / best)
        print(f"\rostrosc {s:8.1f}   max {best:8.1f}   {bar:<40}", end="", flush=True)
        time.sleep(0.2)
except KeyboardInterrupt:
    print()
finally:
    cam.stop()
