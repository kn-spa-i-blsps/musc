"""
Porównanie wariantów ustawień detektora na wielu nagraniach - równolegle
(każda para nagranie x wariant w osobnym procesie, na wszystkich rdzeniach).

    uv run bench.py recordings/*                              # sesje z verify_live
    uv run bench.py film.mov --chip-hz 20 --true-id 5         # zwykłe nagranie
    uv run bench.py recordings/* film.mov:20 \\
        --config feature=max,energy_filter=blur --config feature=luma

Sesja z recordings/ sama podaje HZ i prawdziwe ID (info.json) i czasy klatek
(ts.npy). Zwykły plik: HZ po dwukropku albo --chip-hz.

Wynik dla każdego nagrania: sekundy z potwierdzonym poprawnym ID (C),
z wstępnym (?), klatki z obcym ID potwierdzonym (X) / wstępnym (x),
czas do pierwszego ID i średni czas przetwarzania klatki.

Jeśli nagranie ma etykiety z annotate.py (labels/<nazwa>.json albo
recordings/X/labels.json), liczy też, czy ID siedzi NA diodzie - na
oznaczonych klatkach z diodą:
    na diodzie C / ?   poprawne ID (potwierdzone / wstępne) w promieniu --hit-px
    obok               poprawne ID jest, ale daleko od diody (odblask, ekran...)
    brak               nie ma poprawnego ID
i na klatkach bez diody: fałszywe = mimo to jest poprawne ID (gdzieś indziej).
Precyzja = jaka część podanych pozycji z poprawnym ID jest naprawdę na diodzie
(zła pozycja z pewnym ID jest gorsza niż jej brak).
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _parse_config(text: str) -> dict:
    cfg = {}
    for part in filter(None, text.split(",")):
        k, v = part.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        if v in ("True", "False"):
            v = v == "True"
        cfg[k.strip()] = v
    return cfg


def _video_spec(arg: str, default_hz: float, default_id: int) -> dict:
    p = Path(arg.rsplit(":", 1)[0] if ":" in arg and arg.rsplit(":", 1)[1].replace(".", "").isdigit() else arg)
    hz = float(arg.rsplit(":", 1)[1]) if p.as_posix() != arg else default_hz
    if p.is_dir():                                   # sesja z verify_live
        info = json.loads((p / "info.json").read_text())
        return {"name": p.name, "video": str(p / "raw.avi"), "ts": str(p / "ts.npy"),
                "hz": float(info["arduino"]["hz"]), "true_id": int(info["arduino"]["id"]),
                "labels": _labels(p / "labels.json")}
    return {"name": p.name, "video": str(p), "ts": None, "hz": hz, "true_id": default_id,
            "labels": _labels(Path(__file__).parent / "labels" / f"{p.stem}.json")}


def _labels(path: Path):
    """klatka -> lista punktów diody [px pełnej klatki]; [] = diody nie ma w kadrze."""
    if not path.exists():
        return None
    out = {}
    for l in json.loads(path.read_text())["labels"]:
        if l["state"] == "led":
            out[l["frame"]] = l["pts"]
        elif l["state"] == "absent":
            out[l["frame"]] = []
    return out


def _score_label(pts, mine, scale: float, hit_px: float):
    """Ocena jednej oznaczonej klatki: (kategoria, błąd położenia w px pełnej klatki)."""
    if not pts:
        return ("false" if mine else "neg_ok"), None
    best = None
    for d in mine:
        dist = min(np.hypot(d.x * scale - x, d.y * scale - y) for x, y in pts)
        if best is None or (dist <= hit_px, d.confirmed, -dist) > (best[0] <= hit_px, best[1], -best[0]):
            best = (dist, d.confirmed)
    if best is None:
        return "none", None
    if best[0] > hit_px:
        return "off", best[0]
    return ("hitC" if best[1] else "hitQ"), best[0]


def _run(job) -> dict:
    spec, cfg, hit_px = job
    import cv2
    cv2.setNumThreads(1)                             # 1 rdzeń na zadanie - równolegle są procesy
    from detector import LedIdReceiver, ReceiverConfig
    cap = cv2.VideoCapture(spec["video"])
    tsl = np.load(spec["ts"]) if spec["ts"] else None
    rx = LedIdReceiver(ReceiverConfig(chip_hz=spec["hz"], **cfg))
    tid = spec["true_id"]
    n = C = Q = X = x = 0
    first = None
    proc = 0.0
    tss = []
    labels = spec["labels"] or {}
    lab = {"hitC": 0, "hitQ": 0, "off": 0, "none": 0, "false": 0, "neg_ok": 0}
    errs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = float(tsl[n]) if tsl is not None and n < len(tsl) else cap.get(cv2.CAP_PROP_POS_MSEC)
        tss.append(ts)
        t0 = time.perf_counter()
        ds = rx.process(frame, ts)
        proc += time.perf_counter() - t0
        mine = [d for d in ds if d.drone_id == tid]
        if n in labels:
            scale = frame.shape[1] / rx.small.shape[1]
            cat, err = _score_label(labels[n], mine, scale, hit_px)
            lab[cat] += 1
            if cat.startswith("hit"):
                errs.append(err)
        n += 1
        other = [d for d in ds if d.drone_id != tid]
        if any(d.confirmed for d in other):
            X += 1
        elif other:
            x += 1
        if mine:
            first = first if first is not None else ts
            if any(d.confirmed for d in mine):
                C += 1
            else:
                Q += 1
    fps = 1000.0 / np.median(np.diff(tss)) if len(tss) > 1 else 30.0
    return {"name": spec["name"], "cfg": cfg, "dur": n / fps, "C": C / fps, "Q": Q / fps, "X": X, "x": x,
            "first": None if first is None else (first - tss[0]) / 1000.0, "ms": 1000.0 * proc / max(n, 1),
            "lab": lab if labels else None, "err": float(np.median(errs)) if errs else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="sesje recordings/X albo pliki (plik.mov:HZ)")
    ap.add_argument("--config", action="append", default=[],
                    help="wariant ustawień ReceiverConfig, np. feature=max,energy_filter=blur (można wiele)")
    ap.add_argument("--chip-hz", type=float, default=20.0, help="HZ dla zwykłych plików bez :HZ")
    ap.add_argument("--true-id", type=int, default=5, help="nadawane ID dla zwykłych plików")
    ap.add_argument("--hit-px", type=float, default=20.0,
                    help="max odległość ID od oznaczonej diody [px pełnej klatki], żeby liczyć jako trafienie")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    specs = [_video_spec(v, args.chip_hz, args.true_id) for v in args.videos]
    configs = [_parse_config(c) for c in args.config] or [{}]
    jobs = [(s, c, args.hit_px) for c in configs for s in specs]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(_run, jobs))

    for c in configs:
        print(f"\n== {c or 'domyślne'}")
        tot_c = tot_d = 0.0
        tot_lab = {"hitC": 0, "hitQ": 0, "off": 0, "none": 0, "false": 0, "neg_ok": 0}
        for r in [r for r in results if r["cfg"] == c]:
            first = f"{r['first']:.1f}s" if r["first"] is not None else "brak"
            print(f"  {r['name']:28s} {r['dur']:5.1f}s | C {r['C']:5.1f}s  ? {r['Q']:4.1f}s | "
                  f"obce X {r['X']:3d} x {r['x']:3d} | pierwsze {first:>6s} | {r['ms']:4.1f} ms/kl")
            if r["lab"]:
                L = r["lab"]
                npos = L["hitC"] + L["hitQ"] + L["off"] + L["none"]
                pct = lambda v, tot: f"{100 * v / tot:3.0f}%" if tot else "  - "
                err = f"{r['err']:.1f}px" if r["err"] is not None else "-"
                print(f"  {'':28s} etykiety ({npos} z diodą): na diodzie C {pct(L['hitC'], npos)}"
                      f" ? {pct(L['hitQ'], npos)} | obok {pct(L['off'], npos)} | brak {pct(L['none'], npos)}"
                      f" | błąd {err} | bez diody: fałszywe {L['false']}/{L['false'] + L['neg_ok']}")
                for k in L:
                    tot_lab[k] += L[k]
            tot_c += r["C"] + r["Q"]
            tot_d += r["dur"]
        print(f"  razem z ID: {tot_c:.0f}s z {tot_d:.0f}s nagrań")
        npos = tot_lab["hitC"] + tot_lab["hitQ"] + tot_lab["off"] + tot_lab["none"]
        if npos:
            hits = tot_lab["hitC"] + tot_lab["hitQ"]
            given = hits + tot_lab["off"] + tot_lab["false"]
            print(f"  razem etykiety: NA DIODZIE {100 * hits / npos:.0f}%"
                  f" (C {100 * tot_lab['hitC'] / npos:.0f}%), obok {100 * tot_lab['off'] / npos:.0f}%,"
                  f" fałszywe bez diody {tot_lab['false']} | PRECYZJA {100 * hits / max(given, 1):.0f}%"
                  f" (podana pozycja jest na diodzie)")
    print(f"\n({len(jobs)} zadań w {time.time() - t0:.0f}s na {args.jobs} procesach)")


if __name__ == "__main__":
    main()
