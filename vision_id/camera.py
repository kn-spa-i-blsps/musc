"""
Kamera w osobnym wątku.

Wątek tylko czyta klatki (z czasem zrobienia) do krótkiej kolejki, a pętla
główna je przetwarza. Dzięki temu chwilowo wolniejsza klatka (wykres, GC,
okno) nie powoduje zgubienia następnej - bez tego pętla trwająca 34 ms przy
kamerze 30 fps czyta co drugą klatkę i spada równo do 15 fps.
"""

import threading
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

MAX_QUEUE = 8          # tyle klatek może czekać; starsze wypadają (liczone w `dropped`)


def open_camera(index=None, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """
    Otwiera kamerę w 1280x720 (i tak liczymy na ~960 px, a odczyt 1080p jest
    wyraźnie droższy). Bez indeksu bierze pierwszą, która nie daje czarnego
    obrazu (na Macu pod 0 bywa iPhone przez Continuity Camera, często zasłonięty).
    """
    for i in ([index] if index is not None else range(4)):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 60)       # macOS i tak zwykle da 30
        frame = None
        for _ in range(15):                 # kilka klatek na rozruch auto-ekspozycji
            ok, frame = cap.read()
        if index is not None or (frame is not None and frame.max() > 40):
            print(f"Kamera {i}: {frame.shape[1]}x{frame.shape[0]}" if frame is not None else f"Kamera {i}")
            return cap
        cap.release()
    if index is not None:
        raise SystemExit(f"Kamera {index} nie istnieje albo nie działa. Lista kamer widocznych dla systemu:\n"
                         "  system_profiler SPCameraDataType | grep Model\n"
                         "(iPhone przez Continuity Camera pojawia się dopiero, gdy jest zablokowany, poziomo i blisko Maca)")
    raise SystemExit("Nie znalazłem działającej kamery (macOS: daj Terminalowi dostęp do kamery)")


class CameraThread:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.dropped = 0
        self._q: deque = deque()
        self._cv = threading.Condition()
        self._running = True
        self._t0 = time.perf_counter()
        self._recent: deque = deque(maxlen=60)       # czasy ostatnich klatek -> FPS kamery
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            ts = (time.perf_counter() - self._t0) * 1000.0
            with self._cv:
                if not ok:
                    self._running = False
                    self._cv.notify_all()
                    return
                if len(self._q) >= MAX_QUEUE:
                    self._q.popleft()
                    self.dropped += 1
                self._q.append((frame, ts))
                self._recent.append(ts)
                self._cv.notify()

    @property
    def t0(self) -> float:
        """Początek zegara klatek (time.perf_counter) - ta sama oś czasu dla telemetrii FC."""
        return self._t0

    def read(self, timeout: float = 2.0) -> Optional[Tuple[np.ndarray, float]]:
        """Następna klatka (frame, czas_ms) w kolejności, albo None, gdy kamera przestała działać."""
        with self._cv:
            if not self._q and self._running:
                self._cv.wait(timeout)
            return self._q.popleft() if self._q else None

    @property
    def backlog(self) -> int:
        return len(self._q)

    @property
    def fps(self) -> float:
        r = list(self._recent)
        return (len(r) - 1) * 1000.0 / (r[-1] - r[0]) if len(r) > 1 and r[-1] > r[0] else 0.0

    def measure_fps(self, seconds: float = 1.5) -> float:
        """Ile klatek na sekundę faktycznie daje kamera (klatki z pomiaru są zużywane)."""
        end = time.perf_counter() + seconds
        stamps = []
        while time.perf_counter() < end:
            item = self.read()
            if item is None:
                break
            stamps.append(item[1])
        return (len(stamps) - 1) * 1000.0 / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
