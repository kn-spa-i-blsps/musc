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
from typing import List, Optional, Tuple

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


HFOV_DEG = 65.0
FX = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)


def cam_angles(t: float, shake_deg: float) -> Tuple[float, float]:
    """Obrót kamery (yaw w prawo, pitch w górę) [rad] w chwili t: drganie drona w zawisie + korekty kursu."""
    a = math.radians(shake_deg)
    return (a * (math.sin(2 * math.pi * 0.7 * t / 1000) + 0.4 * math.sin(2 * math.pi * 2.3 * t / 1000 + 1.0)),
            0.6 * a * (math.sin(2 * math.pi * 0.9 * t / 1000 + 2.0) + 0.4 * math.sin(2 * math.pi * 3.1 * t / 1000)))


def render(drones: List[SimDrone], t: float, exposure_ms: float, rng: np.random.Generator,
           offset: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """offset = przesunięcie całej sceny w obrazie [px] (obrót kamery)."""
    ox, oy = offset
    img = np.full((H, W, 3), 35, np.float32)
    img += rng.normal(0, 4, img.shape).astype(np.float32)              # szum sensora
    cv2.circle(img, (int(W - 120 + ox), int(90 + oy)), 14, (250, 250, 250), -1)   # statyczna lampa
    glow = np.zeros_like(img)
    for d in drones:
        frac = d.on_fraction(t, t + exposure_ms)
        if frac <= 0:
            continue
        for i, x, y in d.leds(t):
            if i == d.occluded_led and d.occl_from < t < d.occl_from + 700:
                continue
            col = np.array(d.color, np.float32) * frac
            cv2.circle(glow, (int(x + ox), int(y + oy)), 4, col.tolist(), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 1.5) * 1.6
    return np.clip(img + glow, 0, 255).astype(np.uint8)


def run(fps: float, chip_hz: float, seconds: float, ids: List[int], seed: int,
        out: Optional[str] = None, verbose: bool = True, cam_shake_deg: float = 0.0,
        gyro: bool = False, motion_comp: bool = True) -> dict:
    """
    cam_shake_deg > 0: kamera (dron, na którym siedzi) obraca się o tyle stopni w obie
    strony; gyro=True podaje odbiornikowi ten obrót tak, jak podałby go żyroskop FC.
    """
    rnd = random.Random(seed)
    nrng = np.random.default_rng(seed)
    drones = [SimDrone(i, chip_hz, x=rnd.uniform(200, W - 200), y=rnd.uniform(150, H - 150), rng=rnd)
              for i in ids]
    # symulowane diody mają losowe kolory -> tryb samego mrugania
    rx = LedIdReceiver(ReceiverConfig(chip_hz=chip_hz, led_color=None, hfov_deg=HFOV_DEG, motion_comp=motion_comp))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if out else None

    first_seen = {}
    first_group = {}                       # dron -> kiedy jego diody pierwszy raz były w jednej konstelacji
    wrong = set()
    seen_frames = 0
    frame_ms = 1000.0 / fps
    prev_ang = None
    n_frames = int(seconds * fps)
    for f in range(n_frames):
        t = f * frame_ms + rnd.uniform(-1.5, 1.5)
        yaw, pitch = cam_angles(t, cam_shake_deg) if cam_shake_deg else (0.0, 0.0)
        offset = (-FX * math.tan(yaw), FX * math.tan(pitch))          # kamera w prawo/górę -> scena w lewo/dół
        frame = render(drones, t, exposure_ms=0.5 * frame_ms, rng=nrng, offset=offset)
        if writer:
            writer.write(frame)
        cam_rot = None
        if gyro and prev_ang is not None:
            # wektor obrotu w osiach kamery (x w prawo, y w dół, z do przodu): pitch w górę = +x, yaw w prawo = +y
            cam_rot = np.array([pitch - prev_ang[1], yaw - prev_ang[0], 0.0])
        prev_ang = (yaw, pitch)
        for d in rx.process(frame, t, cam_rot=cam_rot):
            if d.drone_id in ids:
                first_seen.setdefault(d.drone_id, t)
                seen_frames += 1
            else:
                wrong.add(d.drone_id)
        # czyje diody są w której konstelacji (tor -> najbliższa prawdziwa dioda)
        owner = {}
        for tr in rx.tracks:
            best = min(((math.hypot(x + offset[0] - tr.x, y + offset[1] - tr.y), d.drone_id)
                        for d in drones for _, x, y in d.leds(t)), default=(1e9, None))
            owner[tr.tid] = best[1] if best[0] < 12 else None
        for c in rx.constellations:
            own = [owner.get(k) for k in c.track_ids if owner.get(k) is not None]
            if len(own) >= 2 and len(set(own)) == 1:
                first_group.setdefault(own[0], t)
    if writer:
        writer.release()

    if verbose:
        frame_s = FRAME_CHIPS / chip_hz
        print(f"fps={fps:g} chip={chip_hz:g}Hz ({fps / chip_hz:.1f} klatki/chip), ramka={frame_s:.1f}s")
        for i in ids:
            s = f"{first_seen[i] / 1000:.2f}s" if i in first_seen else "NIE ZDEKODOWANO"
            g = f"{first_group[i] / 1000:.2f}s" if i in first_group else "brak"
            print(f"  ID {i:2d}: {s}   (konstelacja po {g})")
        if wrong:
            print(f"  FAŁSZYWE ID: {sorted(wrong)}")
    return {"found": set(first_seen), "wrong": wrong, "first_seen": first_seen, "first_group": first_group,
            "coverage": seen_frames / max(n_frames * len(ids), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--chip-hz", type=float, default=10)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 5, 42, 63])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--cam-shake", type=float, default=0.0, help="obrót kamery +- tyle stopni (dron z kamerą się rusza)")
    ap.add_argument("--gyro", action="store_true", help="podaj odbiornikowi obrót kamery jak z żyroskopu FC")
    args = ap.parse_args()
    res = run(args.fps, args.chip_hz, args.seconds, args.ids, args.seed, args.out,
              cam_shake_deg=args.cam_shake, gyro=args.gyro)
    ok = res["found"] == set(args.ids) and not res["wrong"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
