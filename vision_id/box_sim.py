"""
Eksperyment: ile da detektor drona (YOLO na AI HAT) - diod szukamy tylko w ramce
drona, a kod czytamy wspólnie ze wszystkich diod w ramce.

Ramka "prawdziwa" powstaje z etykiet annotate.py: pozycje diod interpolowane
między oznaczonymi klatkami, ramka = diody + zapas na płytkę/kadłub. Potem psuje
ją model typowych błędów detektora (YoloSim):
    * drganie środka ramki (skorelowane w czasie) i błąd rozmiaru
    * pominięcia wykrycia SERIAMI (rozmycie w ruchu, nietypowy kąt)
    * detektor liczony co N klatek - w ruchu ramka jest spóźniona
    * po zgubieniu ramka jest jeszcze chwilę podtrzymywana (jak ByteTrack)

Tryby:
    caly    - obecny detektor, cały kadr (punkt odniesienia)
    ramka   - nowi kandydaci na diody tylko w ramce
    ramka_e - ramka + dekodery torów z progiem względem lokalnego "zapalona"
    ramka+  - ramka + WSPÓLNY odczyt: jeden dekoder na ramkę (= drona), karmiony
              średnią z torów w ramce. Kilka diod i kolejne kawałki toru (dioda
              zgubiona i złapana od nowa) składają się w jeden odczyt kodu.
    ramka++ - jak ramka+, ale tylko tory mrugające zgodnie ze sobą, próg wzgl. "zapalona"
    caly+h  - cały kadr + trzymanie ID: raz potwierdzone ID zostaje przy punkcie
              śledzonym filtrem Kalmana (bez ramki), gdy dekoder milczy
    ramka++h- ramka++ + trzymanie ID, dopóki tracker nie zgubi ramki

    uv run box_sim.py ~/Downloads/IMG_5655.MOV ~/Downloads/IMG_5660.MOV
    uv run box_sim.py ~/Downloads/IMG_5660.MOV --render typowa ramka+ out.mp4

Uwaga: ramka pochodzi z tych samych etykiet, którymi oceniamy - wariant
"idealna" to górna granica; "typowa" i "slaba" są bliżej rzeczywistości.
"""

import argparse
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

Box = Tuple[float, float, float, float]          # x0, y0, x1, y1 [px pełnej klatki]

SCENARIOS = {
    #            co ile klatek, drganie (część rozmiaru), błąd rozmiaru, start pominięcia, śr. długość serii, podtrzymanie [kl.]
    "idealna": dict(every=1, jitter=0.0, scale_sd=0.0, miss_start=0.0, miss_len=1, coast=0),
    "typowa": dict(every=2, jitter=0.08, scale_sd=0.15, miss_start=0.03, miss_len=6, coast=10),
    "slaba": dict(every=3, jitter=0.15, scale_sd=0.25, miss_start=0.08, miss_len=12, coast=10),
}
MODES = {
    "caly": dict(envelope=False, pool=False, smart=False),     # cały kadr, bez ramki
    "caly+h": dict(envelope=False, pool=False, smart=False, hold=True),
    "ramka": dict(envelope=False, pool=False, smart=False),    # kandydaci tylko w ramce
    "ramka_e": dict(envelope=True, pool=False, smart=False),   # + próg względem "zapalona"
    "ramka+": dict(envelope=False, pool=True, smart=False),    # + wspólny odczyt wszystkich torów
    "ramka++": dict(envelope=True, pool=True, smart=True),     # + wspólny odczyt torów zgodnych, próg wzgl. "zapalona"
    "ramka++h": dict(envelope=True, pool=True, smart=True, hold=True),
}
MARGIN = 60.0          # zapas wokół diod (płytka/kadłub) [px pełnej klatki]
MIN_HALF = 100.0       # najmniejsza połowa boku ramki


def gt_boxes(labels_path: Path, max_gap: int = 40) -> Dict[int, Box]:
    """Ramka drona dla każdej klatki między dwiema oznaczonymi (z diodą), interpolowana liniowo."""
    labs = sorted((l for l in json.loads(labels_path.read_text())["labels"] if l["state"] in ("led", "absent")),
                  key=lambda l: l["frame"])

    def box_of(pts):
        p = np.array(pts, float)
        cx, cy = p.mean(axis=0)
        hw = max((p[:, 0].max() - p[:, 0].min()) / 2 + MARGIN, MIN_HALF)
        hh = max((p[:, 1].max() - p[:, 1].min()) / 2 + MARGIN, MIN_HALF)
        return np.array([cx - hw, cy - hh, cx + hw, cy + hh])

    out: Dict[int, Box] = {}
    for a, b in zip(labs, labs[1:]):
        if a["state"] != "led" or b["state"] != "led" or b["frame"] - a["frame"] > max_gap:
            continue
        ba, bb = box_of(a["pts"]), box_of(b["pts"])
        for f in range(a["frame"], b["frame"] + 1):
            k = (f - a["frame"]) / max(b["frame"] - a["frame"], 1)
            out[f] = tuple((1 - k) * ba + k * bb)
    return out


class YoloSim:
    """Zamienia prawdziwą ramkę w taką, jaką dałby niedoskonały detektor + prosty tracker."""

    def __init__(self, every, jitter, scale_sd, miss_start, miss_len, coast, seed=0):
        self.every, self.jitter, self.scale_sd = every, jitter, scale_sd
        self.miss_start, self.miss_len, self.coast = miss_start, miss_len, coast
        self.rng = np.random.default_rng(seed)
        self.n = np.zeros(2)
        self.missing = False
        self.held: Optional[Box] = None
        self.age = 0

    def step(self, i: int, gt: Optional[Box]) -> Optional[Box]:
        if i % self.every == 0:
            det = None
            if gt is not None:
                if self.missing:
                    self.missing = self.rng.random() > 1.0 / self.miss_len
                else:
                    self.missing = self.rng.random() < self.miss_start
                if not self.missing:
                    rho = 0.8
                    self.n = rho * self.n + math.sqrt(1 - rho * rho) * self.rng.normal(0, self.jitter, 2)
                    x0, y0, x1, y1 = gt
                    w, h = x1 - x0, y1 - y0
                    s = math.exp(self.rng.normal(0, self.scale_sd)) if self.scale_sd else 1.0
                    cx, cy = (x0 + x1) / 2 + self.n[0] * w, (y0 + y1) / 2 + self.n[1] * h
                    det = (cx - s * w / 2, cy - s * h / 2, cx + s * w / 2, cy + s * h / 2)
            if det is not None:
                self.held, self.age = det, 0
            else:
                self.age += self.every
        elif self.held is not None:
            self.age += 1
        return self.held if self.held is not None and self.age <= self.coast else None


@dataclass
class Obs:
    drone_id: int
    x: float               # [px obrazu pomniejszonego], jak DroneObs
    y: float
    confirmed: bool
    held: bool = False     # pozycja z trzymania ID (Kalman), nie z dekodera


class BoxPool:
    """
    Jeden dekoder na ramkę drona. Co klatkę każdy mrugający tor w ramce daje swój
    poziom znormalizowany do [-1, 1], średnia idzie do dekodera.

    smart=False: normalizacja z percentyli toru, średnia ze WSZYSTKICH torów w ramce.
    smart=True:
      * normalizacja względem lokalnego poziomu "zapalona" (maks. z ostatnich
        2 chipów - dioda nie gaśnie dłużej niż 1 chip), odporna na zmianę tła
      * do średniej tylko tory mrugające ZGODNIE ze sobą (diody drona i ich
        odblaski - ten sam kod); krawędź ręki czy kabla mruga inaczej i odpada
    Pozycja = tor z grupy (albo najzgodniejszy ze wspólnym sygnałem) z najjaśniejszym rdzeniem.
    """

    def __init__(self, chip_hz: float, min_contrast: float, smart: bool):
        from protocol import TemplateDecoder
        self.dec = TemplateDecoder(chip_hz, min_contrast=0.5)
        self.chip_ms = 1000.0 / chip_hz
        self.min_contrast = min_contrast
        self.smart = smart
        self.hist: deque = deque(maxlen=40)                      # (ts, wspólny poziom)
        self.norm: Dict[int, deque] = {}                         # tid -> (ts, poziom toru [-1, 1])
        self.gaps: Dict[int, deque] = {}                         # tid -> spadki względem "zapalona"
        self.members: List = []

    def _level(self, t, frame_dt: float) -> Optional[float]:
        if not self.smart:
            if t.lohi is None or t.contrast < self.min_contrast:
                return None
            lo, hi = t.lohi
            return float(np.clip((t.levels[-1] - 0.5 * (lo + hi)) / max(0.5 * (hi - lo), 1e-6), -1, 1))
        k = max(3, int(round(2 * self.chip_ms / max(frame_dt, 1.0))))
        if len(t.levels) < k:
            return None
        lv = list(t.levels)[-k:]
        gap = max(lv) - lv[-1]
        g = self.gaps.setdefault(t.tid, deque(maxlen=90))
        g.append(gap)
        depth = float(np.percentile(g, 90)) if len(g) >= 6 else 0.0
        if depth < self.min_contrast:
            return None
        return float(np.clip(1.0 - 2.0 * gap / depth, -1, 1))

    def _pair_corr(self, a, b) -> Optional[float]:
        ha = dict(self.norm.get(a.tid, ()))
        pairs = [(v, ha[s]) for s, v in self.norm.get(b.tid, ()) if s in ha]
        if len(pairs) < 10:
            return None
        x, y = np.array(pairs).T
        if x.std() < 1e-3 or y.std() < 1e-3:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def step(self, ts: float, rx, roi) -> List[Obs]:
        if roi is None:
            self.members = []
            return self._out(ts)
        x0, y0, x1, y1 = roi
        cur = {}
        for t in rx.tracks:
            if x0 <= t.x <= x1 and y0 <= t.y <= y1 and t.levels:
                v = self._level(t, rx._frame_dt)
                if v is not None:
                    self.norm.setdefault(t.tid, deque(maxlen=40)).append((ts, v))
                    cur[t.tid] = (t, v)
        members = list(cur.values())
        if self.smart and len(members) > 1:
            ts_ = [t for t, _ in members]
            corr = {(a.tid, b.tid): self._pair_corr(a, b) for a in ts_ for b in ts_ if a.tid != b.tid}
            known = [t for t in ts_ if any(corr[(t.tid, o.tid)] is not None for o in ts_ if o.tid != t.tid)]
            if known:
                # ziarno = tor najbardziej zgodny z resztą; grupa = tory zgodne z ziarnem
                seed = max(known, key=lambda t: sum(max(corr[(t.tid, o.tid)] or 0.0, 0.0)
                                                    for o in ts_ if o.tid != t.tid))
                members = [(t, v) for t, v in members
                           if t is seed or (corr[(seed.tid, t.tid)] or 0.0) >= 0.5]
        self.members = [t for t, _ in members]
        if members:
            pooled = float(np.mean([v for _, v in members]))
            self.hist.append((ts, pooled))
            self.dec.push(ts, pooled)
        return self._out(ts)

    def _corr(self, t) -> float:
        h = dict(self.hist)
        pairs = [(v, h[s]) for s, v in self.norm.get(t.tid, ()) if s in h]
        if len(pairs) < 10:
            return 0.0
        a, b = np.array(pairs).T
        if a.std() < 1e-3 or b.std() < 1e-3:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _out(self, ts: float) -> List[Obs]:
        d = self.dec
        if d.last_id is None or ts - d.last_ok_ts > 1.5 * d.frame_ms or not self.members:
            return []
        good = [t for t in self.members if self._corr(t) >= 0.5]
        if not good:
            return []
        best = max(good, key=lambda t: t.core) if self.smart else \
            max(good, key=lambda t: (round(self._corr(t), 1), t.core))
        return [Obs(d.last_id, best.x, best.y, d.confirmed_id == d.last_id)]


class IdHold:
    """
    Trzymanie raz potwierdzonego ID, gdy dekoder chwilowo milczy.

    Pozycja źródła jest śledzona filtrem Kalmana (stała prędkość). Gdy dekoder
    milczy: przewidujemy, gdzie będzie dioda, i bierzemy tor z najjaśniejszym
    rdzeniem w bramce wokół przewidywania (źródło jest białe i jaśniejsze niż
    odblaski). Bez pomiaru dłużej niż max_coast_s nie podajemy pozycji.

    use_box=False: ID ginie po max_lost_s bez pomiaru.
    use_box=True: ID należy do toru ramki - ginie dopiero, gdy tracker zgubi ramkę.
    """

    def __init__(self, use_box: bool, gate: float = 15.0, max_coast_s: float = 0.25, max_lost_s: float = 1.0):
        self.use_box, self.gate = use_box, gate
        self.max_coast_ms, self.max_lost_ms = 1000 * max_coast_s, 1000 * max_lost_s
        self.reset()

    def reset(self) -> None:
        self.id: Optional[int] = None
        self.x = np.zeros(4)                        # x, y, vx, vy [px/ms]
        self.P = np.eye(4)
        self.ts = 0.0
        self.meas_ts = float("-inf")

    def _predict(self, ts: float) -> None:
        dt = max(ts - self.ts, 1.0)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = 0.02                                    # niepewność przyspieszenia (ruch ręką / dronem)
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q * q * G @ G.T
        self.ts = ts

    def _update(self, x: float, y: float, ts: float) -> None:
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = H @ self.P @ H.T + np.eye(2) * 2.0 ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([x, y]) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.meas_ts = ts

    def step(self, ts: float, rx, ds: List, roi) -> List:
        if self.use_box and roi is None:
            self.reset()                             # tracker zgubił drona -> tożsamość też
            return ds
        conf = [d for d in ds if d.confirmed]
        if conf:
            d = next((d for d in conf if d.drone_id == self.id), conf[0])
            if d.drone_id != self.id:                # pierwszy odczyt albo ponowna identyfikacja
                self.reset()
                self.id, self.ts = d.drone_id, ts
                self.x[:2] = d.x, d.y
                self.P = np.diag([4.0, 4.0, 0.01, 0.01])
                self.meas_ts = ts
                return ds
        if self.id is None:
            return ds
        self._predict(ts)
        own = [d for d in ds if d.drone_id == self.id]
        if own:
            self._update(own[0].x, own[0].y, ts)
            return ds
        px, py = self.x[:2]
        gate = self.gate + 2 * float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))
        cand = [t for t in rx.tracks if math.hypot(t.x - px, t.y - py) <= gate and
                (roi is None or roi[0] <= t.x <= roi[2] and roi[1] <= t.y <= roi[3])]
        if cand:
            best = max(cand, key=lambda t: t.core)
            self._update(best.x, best.y, ts)
        if not self.use_box and ts - self.meas_ts > self.max_lost_ms:
            self.reset()
            return ds
        if ts - self.meas_ts > self.max_coast_ms:
            return ds                                # ID trzymamy, ale nie wiemy dokładnie gdzie dioda
        return ds + [Obs(self.id, float(self.x[0]), float(self.x[1]), True, True)]


def _labels_path(video: str) -> Path:
    return Path(__file__).parent / "labels" / f"{Path(video).stem}.json"


def _run(job) -> dict:
    video, scen, mode, true_id, hit_px, seed, render = job
    import cv2
    cv2.setNumThreads(1)
    from bench import _labels, _score_label
    from detector import LedIdReceiver, ReceiverConfig

    labels = _labels(_labels_path(video)) or {}
    gts = gt_boxes(_labels_path(video))
    use_box = not mode.startswith("caly")
    yolo = YoloSim(**SCENARIOS[scen], seed=seed) if use_box else None
    # w ramce drona dekodery torów mogą używać progu względem "zapalona" (MODES)
    cfg = ReceiverConfig(chip_hz=20.0, envelope=MODES[mode]["envelope"])
    rx = LedIdReceiver(cfg)
    pool = BoxPool(cfg.chip_hz, cfg.min_contrast, MODES[mode]["smart"]) if MODES[mode]["pool"] else None
    hold = IdHold(use_box) if MODES[mode].get("hold") else None
    cap = cv2.VideoCapture(video)
    writer = None
    lab = {"hitC": 0, "hitQ": 0, "off": 0, "none": 0, "false": 0, "neg_ok": 0}
    errs, n, boxed, wrong = [], 0, 0, 0
    id_known = 0                                  # klatki z diodą, w których dron ma poprawne ID (gdziekolwiek)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        scale = frame.shape[1] / cfg.proc_width
        box = yolo.step(n, gts.get(n)) if yolo else None
        roi = None if not use_box else (tuple(v / scale for v in box) if box else (0, 0, 0, 0))
        boxed += box is not None
        ds = rx.process(frame, ts, roi)
        if pool is not None:
            ds = pool.step(ts, rx, roi if box else None)
        elif use_box:
            ds = [d for d in ds if roi[0] <= d.x <= roi[2] and roi[1] <= d.y <= roi[3]]
        if hold is not None:
            ds = hold.step(ts, rx, ds, roi if box or not use_box else None)
        mine = [d for d in ds if d.drone_id == true_id]
        wrong += any(d.drone_id != true_id for d in ds)
        if n in labels:
            cat, err = _score_label(labels[n], mine, scale, hit_px)
            lab[cat] += 1
            if labels[n]:
                id_known += bool(mine) or (hold is not None and hold.id == true_id)
            if cat.startswith("hit"):
                errs.append(err)
        if render:
            img = rx.small.copy()
            if box:
                cv2.rectangle(img, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (255, 200, 0), 2)
            for t in rx.tracks:
                cv2.circle(img, (int(t.x), int(t.y)), 4, (200, 200, 200), 1)
            for d in ds:
                col = (0, 255, 0) if d.drone_id == true_id and d.confirmed else \
                      (0, 200, 255) if d.drone_id == true_id else (0, 0, 255)
                cv2.circle(img, (int(d.x), int(d.y)), 14, col, 2)
                cv2.putText(img, f"DRON {d.drone_id}{'' if d.confirmed else '?'}{' K' if getattr(d, 'held', False) else ''}", (int(d.x) + 16, int(d.y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            for x, y in labels.get(n, []):
                cv2.drawMarker(img, (int(x / scale), int(y / scale)), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(img, f"{mode} / ramka {scen}  {n / 60:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            if writer is None:
                writer = cv2.VideoWriter(render, cv2.VideoWriter_fourcc(*"mp4v"), 60, (img.shape[1], img.shape[0]))
            writer.write(img)
        n += 1
    if writer:
        writer.release()
    return {"video": Path(video).stem, "scen": scen if use_box else "-", "mode": mode, "lab": lab,
            "err": float(np.median(errs)) if errs else None, "boxed": boxed / max(n, 1), "wrong": wrong,
            "id_known": id_known}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--true-id", type=int, default=5)
    ap.add_argument("--hit-px", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=2, help="ile losowań błędów ramki (wyniki uśrednione)")
    ap.add_argument("--render", nargs=3, metavar=("SCENARIUSZ", "TRYB", "PLIK"), help="nagraj podgląd jednego wariantu")
    ap.add_argument("--scen", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--modes", nargs="+", default=[m for m in MODES if not m.startswith("caly")], choices=list(MODES))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    videos = [str(Path(v).expanduser()) for v in args.videos]

    if args.render:
        scen, mode, out = args.render
        for v in videos:
            path = out if len(videos) == 1 else f"{Path(out).stem}_{Path(v).stem}{Path(out).suffix}"
            r = _run((v, scen, mode, args.true_id, args.hit_px, 0, path))
            print(f"{path}: {r['lab']}")
        return

    jobs = [(v, "-", m, args.true_id, args.hit_px, 0, None) for v in videos for m in MODES
            if m.startswith("caly") and (m == "caly" or m in args.modes)]
    jobs += [(v, s, m, args.true_id, args.hit_px, seed, None) for v in videos for s in args.scen
             for m in args.modes if not m.startswith("caly") for seed in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        res = list(ex.map(_run, jobs))

    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in res:
        groups.setdefault((r["video"], r["mode"], r["scen"]), []).append(r)
    print(f"{'nagranie':10s} {'tryb':7s} {'ramka':8s} | na diodzie (C) | obok | brak | precyzja | ramka jest | obce ID | ID drona")
    for (v, m, s), rs in groups.items():
        L = {k: sum(r["lab"][k] for r in rs) for k in rs[0]["lab"]}
        npos = L["hitC"] + L["hitQ"] + L["off"] + L["none"]
        hits = L["hitC"] + L["hitQ"]
        prec = 100 * hits / max(hits + L["off"] + L["false"], 1)
        print(f"{v:10s} {m:7s} {s:8s} | {100 * hits / npos:4.0f}% ({100 * L['hitC'] / npos:3.0f}%)  "
              f"| {100 * L['off'] / npos:3.0f}% | {100 * L['none'] / npos:3.0f}% | {prec:5.0f}%   "
              f"| {100 * np.mean([r['boxed'] for r in rs]):4.0f}% kl. | {np.mean([r['wrong'] for r in rs]):5.0f} kl. "
              f"| {100 * sum(r['id_known'] for r in rs) / max(npos, 1):4.0f}%")
    print(f"({len(jobs)} zadań w {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
