"""
Symulacja bez sprzętu: renderuje syntetyczne wideo z kilkoma "dronami"
(po 4 kolorowe diody) i puszcza je przez cały odbiornik.

Uwzględnia rzeczy, które w realu psują odczyt:
  * nadajniki niezsynchronizowane z kamerą (losowa faza) i z rozjechanym
    zegarem (Arduino +-1%),
  * jitter czasu klatek, czas naświetlania (klatka na zboczu = pół-jasna dioda),
  * ruch dronów, szum sensora, statyczna lampa w kadrze,
  * chwilowe zasłonięcie diody, czerwone diody (słabe w skali szarości).

    uv run simulate.py                 # test, wypisuje wynik
    uv run simulate.py --fps 60 --chip-hz 20
    uv run simulate.py --out sim.mp4   # zapisz wideo -> uv run receiver.py --video sim.mp4
"""

import argparse
import math
import random
from typing import List, Optional

import cv2
import numpy as np

from detector import LedIdReceiver, ReceiverConfig
from protocol import FRAME_CHIPS, encode_frame

W, H = 960, 540
COLORS = [(40, 40, 255), (40, 255, 40), (40, 230, 255), (255, 255, 255)]  # BGR: czerw., ziel., żółta, biała


class SimDrone:
    def __init__(self, drone_id: int, chip_hz: float, x: float, y: float, rng: random.Random):
        self.drone_id = drone_id
        self.chips = encode_frame(drone_id)
        self.chip_ms = 1000.0 / chip_hz * rng.uniform(0.99, 1.01)    # błąd zegara
        self.phase = rng.uniform(0, FRAME_CHIPS * self.chip_ms)
        self.x0, self.y0 = x, y
        self.ax, self.ay = rng.uniform(40, 120), rng.uniform(20, 60)   # amplituda ruchu
        self.w = rng.uniform(0.15, 0.4)                                # rad/s
        self.size = rng.uniform(30, 60)                                # rozstaw diod [px]
        self.yaw = rng.uniform(0, math.pi)
        self.color = rng.choice(COLORS)
        self.occluded_led = rng.randrange(4)
        self.occl_from = rng.uniform(2000, 6000)

    def on_fraction(self, t0: float, t1: float, steps: int = 8) -> float:
        """Jaką część czasu naświetlania [t0, t1] dioda świeciła."""
        n = 0
        for k in range(steps):
            t = t0 + (k + 0.5) * (t1 - t0) / steps + self.phase
            n += self.chips[int(t // self.chip_ms) % FRAME_CHIPS]
        return n / steps

    def leds(self, t: float):
        cx = self.x0 + self.ax * math.sin(self.w * t / 1000)
        cy = self.y0 + self.ay * math.sin(1.3 * self.w * t / 1000)
        yaw = self.yaw + 0.3 * t / 1000
        pts = []
        for i in range(4):
            a = yaw + i * math.pi / 2
            # widok lekko z boku: kwadrat spłaszczony w pionie
            pts.append((i, cx + self.size * math.cos(a), cy + 0.35 * self.size * math.sin(a)))
        return pts


def render(drones: List[SimDrone], t: float, exposure_ms: float, rng: np.random.Generator) -> np.ndarray:
    img = np.full((H, W, 3), 35, np.float32)
    img += rng.normal(0, 4, img.shape).astype(np.float32)              # szum sensora
    cv2.circle(img, (W - 120, 90), 14, (250, 250, 250), -1)            # statyczna lampa
    glow = np.zeros_like(img)
    for d in drones:
        frac = d.on_fraction(t, t + exposure_ms)
        if frac <= 0:
            continue
        for i, x, y in d.leds(t):
            if i == d.occluded_led and d.occl_from < t < d.occl_from + 700:
                continue
            col = np.array(d.color, np.float32) * frac
            cv2.circle(glow, (int(x), int(y)), 4, col.tolist(), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 1.5) * 1.6
    return np.clip(img + glow, 0, 255).astype(np.uint8)


def run(fps: float, chip_hz: float, seconds: float, ids: List[int], seed: int,
        out: Optional[str] = None, verbose: bool = True) -> dict:
    rnd = random.Random(seed)
    nrng = np.random.default_rng(seed)
    drones = [SimDrone(i, chip_hz, x=rnd.uniform(200, W - 200), y=rnd.uniform(150, H - 150), rng=rnd)
              for i in ids]
    # symulowane diody mają losowe kolory -> tryb samego mrugania
    rx = LedIdReceiver(ReceiverConfig(chip_hz=chip_hz, led_color=None))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if out else None

    first_seen = {}
    wrong = set()
    frame_ms = 1000.0 / fps
    for f in range(int(seconds * fps)):
        t = f * frame_ms + rnd.uniform(-1.5, 1.5)
        frame = render(drones, t, exposure_ms=0.5 * frame_ms, rng=nrng)
        if writer:
            writer.write(frame)
        for d in rx.process(frame, t):
            if d.drone_id in ids:
                first_seen.setdefault(d.drone_id, t)
            else:
                wrong.add(d.drone_id)
    if writer:
        writer.release()

    if verbose:
        frame_s = FRAME_CHIPS / chip_hz
        print(f"fps={fps:g} chip={chip_hz:g}Hz ({fps / chip_hz:.1f} klatki/chip), ramka={frame_s:.1f}s")
        for i in ids:
            s = f"{first_seen[i] / 1000:.2f}s" if i in first_seen else "NIE ZDEKODOWANO"
            print(f"  ID {i:2d}: {s}")
        if wrong:
            print(f"  FAŁSZYWE ID: {sorted(wrong)}")
    return {"found": set(first_seen), "wrong": wrong}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--chip-hz", type=float, default=10)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 5, 42, 63])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = run(args.fps, args.chip_hz, args.seconds, args.ids, args.seed, args.out)
    ok = res["found"] == set(args.ids) and not res["wrong"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
