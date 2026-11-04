"""
Ocena dekodera na nagraniu z kamery (raw.avi + ts.npy, np. z pi/station.py albo pi/record.py).

    uv run eval_rec.py NAGRANIE --id 5 --chip-hz 20 --preset pi_niebo
    uv run eval_rec.py NAGRANIE --id 5 --chip-hz 20 --color nonblue --set proc_width=1332 --set point_lock=True

Wypisuje: po ilu sekundach pierwszy raz prawdziwe ID, w ilu klatkach potem było
(i ile potwierdzone), fałszywe ID (inne niż nadajnik) i ile torów średnio/maks.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from detector import PRESETS, LedIdReceiver, ReceiverConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", help="katalog z raw.avi i ts.npy")
    ap.add_argument("--id", type=int, required=True, help="ID nadawane przez Arduino")
    ap.add_argument("--chip-hz", type=float, default=20.0)
    ap.add_argument("--color", default="none", choices=["red", "green", "blue", "yellow", "nonblue", "none"])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--set", action="append", default=[], metavar="KLUCZ=WARTOŚĆ")
    args = ap.parse_args()

    cfg = ReceiverConfig(chip_hz=args.chip_hz, led_color=None if args.color == "none" else args.color)
    for key, val in PRESETS.get(args.preset, {}).items():
        setattr(cfg, key, val)
    for item in args.set:
        key, val = item.split("=", 1)
        cur = getattr(cfg, key)
        setattr(cfg, key, (val.lower() in ("1", "true")) if isinstance(cur, bool) else type(cur)(val))
    rx = LedIdReceiver(cfg)

    rec = Path(args.rec)
    cap = cv2.VideoCapture(str(rec / "raw.avi"))
    ts = np.load(rec / "ts.npy")
    first, hit, hit_conf, after, wrong, ntr = None, 0, 0, 0, {}, []
    for t in ts:
        ok, frame = cap.read()
        if not ok:
            break
        ids = {d.drone_id: d for d in rx.process(frame, float(t))}
        if first is None and args.id in ids:
            first = t
        if first is not None:
            after += 1
            hit += args.id in ids
            hit_conf += args.id in ids and ids[args.id].confirmed
        for i in ids:
            if i != args.id:
                wrong[i] = wrong.get(i, 0) + 1
        ntr.append(len(rx.tracks))
    when = f"{first / 1000:.2f} s" if first is not None else "nigdy"
    print(f"{rec.name}: pierwsze ID {args.id} po {when}; potem w {hit}/{after} klatek (potwierdzone {hit_conf}); "
          f"fałszywe ID: {wrong or 'brak'}; torów średnio {np.mean(ntr):.1f}, maks. {max(ntr)}")


if __name__ == "__main__":
    main()
