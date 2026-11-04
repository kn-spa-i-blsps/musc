"""
Odczyt ID z nagrania na Raspberry Pi (wywołuje station.py po każdym nagraniu).

    python3 decode_rec.py ~/recordings/03_ruch [--preset pi_niebo]

Czyta raw.avi + ts.npy, puszcza LedIdReceiver (detector.py) na każdej klatce i zapisuje:
    overlay.mjpg + overlay_idx.npy   film z nakładką dekodera (co 2. klatka, do odtworzenia na stronie)
    progress.json                    postęp w trakcie
    result.json                      podsumowanie: od kiedy ID, w ilu klatkach, fałszywe ID, sekunda po sekundzie
Prawdziwe ID nadajnika bierze z info.json (odpowiedź Arduino "ID=5 ...").
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import PRESETS, LedIdReceiver, ReceiverConfig  # noqa: E402
from receiver import draw  # noqa: E402

# kamera stoi na biurku: bez kompensacji ruchu kamery i bez przepływu optycznego
# (na Pi razem ~26 ms/klatkę, a przy point_lock prawie nic nie dają)
STATION = {"flow": False, "motion_comp": False}
OUT_W = 666                     # szerokość filmu z nakładką
GAMMA = np.array([min(255, int(255 * (i / 255) ** 0.45)) for i in range(256)], np.uint8)   # rozjaśnienie podglądu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--preset", default="pi_niebo", choices=sorted(PRESETS))
    ap.add_argument("--chip-hz", type=float, default=20.0)
    args = ap.parse_args()
    d = Path(args.dir)
    info = json.loads((d / "info.json").read_text()) if (d / "info.json").exists() else {}
    m = re.search(r"ID=(\d+)", info.get("arduino") or "")
    true_id = int(m.group(1)) if m else None

    cfg = ReceiverConfig(chip_hz=args.chip_hz)
    for k, v in {**PRESETS[args.preset], **STATION}.items():
        setattr(cfg, k, v)
    rx = LedIdReceiver(cfg)
    cap = cv2.VideoCapture(str(d / "raw.avi"))
    ts = np.load(d / "ts.npy")
    n = len(ts)

    out = open(d / "overlay.mjpg", "wb")
    idx = []                                   # (bajt początku, czas klatki ms)
    first = None
    hit = after = 0
    wrong: dict = {}
    per_sec: dict = {}
    t0 = time.perf_counter()
    for i, t in enumerate(ts):
        ok, frame = cap.read()
        if not ok:
            break
        drones = rx.process(frame, float(t))
        ids = {dr.drone_id for dr in drones}
        sec = int(t // 1000)
        ps = per_sec.setdefault(sec, {"klatek": 0, "z_id": 0})
        ps["klatek"] += 1
        good = true_id in ids if true_id is not None else bool(ids)
        ps["z_id"] += good
        if first is None and good:
            first = float(t)
        if first is not None:
            after += 1
            hit += good
        for x in ids:
            if true_id is not None and x != true_id:
                wrong[x] = wrong.get(x, 0) + 1
        if i % 2 == 0:
            view = cv2.LUT(rx.small, GAMMA)
            draw(view, rx, drones, 60.0)
            cv2.putText(view, f"t = {t / 1000:5.2f} s", (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            small = cv2.resize(view, (OUT_W, int(view.shape[0] * OUT_W / view.shape[1])), interpolation=cv2.INTER_AREA)
            jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            idx.append((out.tell(), float(t)))
            out.write(jpg)
        if i % 30 == 0:
            (d / "progress.json").write_text(json.dumps({"klatka": i, "z": n}))
    out.close()
    np.save(d / "overlay_idx.npy", np.array(idx))
    dt = time.perf_counter() - t0
    res = {
        "nadajnik_id": true_id,
        "klatek": n,
        "dekoder_fps_na_pi": round(n / dt, 1),
        "pierwsze_id_s": None if first is None else round(first / 1000, 2),
        "klatki_z_id_po_pierwszym": [hit, after],
        "procent": None if not after else round(100 * hit / after, 1),
        "falszywe_id": wrong,
        "ustawienia": args.preset + " + " + ", ".join(f"{k}={v}" for k, v in STATION.items()),
        "sekundy": {str(k): v for k, v in sorted(per_sec.items())},
    }
    (d / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (d / "progress.json").write_text(json.dumps({"klatka": n, "z": n}))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
