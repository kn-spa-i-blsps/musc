"""
Test integracyjny bez kamery: symulujemy 2 poruszające się drony mrugające
ramką protokołu (60 fps, 30 bit/s) i sprawdzamy, czy potok
detekcja -> węgierski -> Kalman -> dekoder poprawnie odczytuje ich ID.

Uruchomienie: python3 test_simulation.py
"""

import itertools
import random
from typing import List

import numpy as np

from blink_protocol import BIT_DUR_MS, PACKET_BITS, PACKET_DUR_MS, encode_packet, is_safe_id
from normal_light import Light
from tracking import TRACK_TIMEOUT_MS, match_detections

FPS = 60.0
FRAME_DUR = 1000.0 / FPS


class SimDrone:
    def __init__(self, drone_id: int, x: float, y: float, vx: float, vy: float,
                 phase_ms: float):
        assert is_safe_id(drone_id)
        self.drone_id = drone_id
        self.packet = encode_packet(drone_id)
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.phase_ms = phase_ms  # przesunięcie startu nadawania

    def step(self):
        self.x += self.vx
        self.y += self.vy

    def is_on(self, ts: float) -> bool:
        bit_idx = int(((ts + self.phase_ms) % PACKET_DUR_MS) // BIT_DUR_MS)
        return bool(self.packet[min(bit_idx, PACKET_BITS - 1)])


def run():
    random.seed(7)
    drones = [
        SimDrone(1234, x=100, y=100, vx=1.5, vy=0.7, phase_ms=0.0),
        SimDrone(31000, x=500, y=300, vx=-1.0, vy=0.5, phase_ms=417.0),
    ]

    sources: List[Light] = []
    track_id_gen = itertools.count(1)

    n_frames = int(4 * PACKET_DUR_MS / FRAME_DUR)  # 4 sekundy
    for f in range(n_frames):
        ts = int(f * FRAME_DUR + random.uniform(-1.0, 1.0))
        detections = []
        for d in drones:
            d.step()
            if d.is_on(ts):
                # szum pozycji detekcji +/- 1 px
                detections.append((int(d.x + random.uniform(-1, 1)),
                                   int(d.y + random.uniform(-1, 1)), 5.0))

        unmatched = match_detections(sources, detections, ts)
        for di in unmatched:
            x, y, _ = detections[di]
            sources.append(Light(next(track_id_gen), ts, x, y))
        sources = [s for s in sources if (ts - s.last_seen_ts) < TRACK_TIMEOUT_MS]

    decoded = sorted(s.drone_id for s in sources if s.drone_id is not None)
    expected = sorted(d.drone_id for d in drones)

    print(f"Trackery na koniec: {len(sources)} (oczekiwane: {len(drones)})")
    for s in sources:
        print(f"  TRK:{s.ID} pos=({s.x:.0f},{s.y:.0f}) "
              f"drone_id={s.drone_id} confidence={s.confidence:.2f}")

    assert len(sources) == len(drones), "zla liczba trackerow (track zginal lub sie zdublowal)"
    assert decoded == expected, f"zle ID: {decoded} != {expected}"
    print("\nTEST OK - oba drony zidentyfikowane po ID z mrugania")


if __name__ == "__main__":
    run()
