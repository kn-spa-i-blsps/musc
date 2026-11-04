"""
Odbiornik na żywo (kamera) albo z nagrania.

    uv run receiver.py                       # pierwsza kamera, która nie daje czarnego obrazu
    uv run receiver.py --video nagranie.mov  # plik z telefonu
    uv run receiver.py --chip-hz 20          # gdy kamera ma 60 fps i Arduino ustawione na HZ 20
    uv run receiver.py --record raw.mp4      # zapis surowego obrazu do późniejszego strojenia
    uv run receiver.py --video in.mov --out wynik.mp4 --no-gui   # przetworzony film z nakładką

Klawisze: q - wyjście, spacja - pauza, +/- - czułość, d - podgląd "co widzi maszyna".
"""

import argparse

import cv2
import numpy as np

from camera import CameraThread, open_camera
from detector import PRESETS, LedIdReceiver, ReceiverConfig


BLINK_CONTRAST = 30.0      # od tego kontrastu tor uznajemy za mrugający (rysujemy go)
MAX_PLOTS = 4              # ile torów dostaje wykres sygnału w trybie podglądu


def draw_machine_view(frame, rx: LedIdReceiver) -> None:
    """
    Podgląd "co widzi maszyna": obraz przyciemniony, na nim
      zielone   - piksele, które właśnie się ZAPALIŁY (zdarzenie +1)
      czerwone  - piksele, które właśnie ZGASŁY (zdarzenie -1)
      niebieskie- piksele prawie nasycone (kandydat na świecącą diodę)
    Dioda, którą da się odczytać, miga na zmianę zielono/czerwono w jednym
    miejscu; jeśli tego nie widać, kamera po prostu nie widzi mrugania.
    """
    cv2.convertScaleAbs(frame, frame, alpha=0.35)   # przyciemnienie bez liczenia na float
    if rx.dbg_lit is not None and rx.dbg_lit.shape == frame.shape[:2]:
        frame[rx.dbg_lit] = (200, 100, 0)
    if rx.dbg_on is not None and rx.dbg_on.shape == frame.shape[:2]:
        frame[rx.dbg_on] = (0, 255, 0)
        frame[rx.dbg_off] = (0, 0, 255)


def draw_signal(frame, t, x: int, y: int, w: int = 150, h: int = 44) -> None:
    """Wykres sygnału toru: biała linia = jasność plamki, szara = próg, pasek = odczyt 0/1."""
    if t.lohi is None or len(t.levels) < 6:
        return
    vals = np.array(t.levels, dtype=np.float32)
    lo, hi = t.lohi
    span = max(hi - lo, 1.0)
    H, W = frame.shape[:2]
    x = int(min(max(x, 0), W - w - 1))
    y = int(min(max(y, 0), H - h - 14))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

    def ypix(val: float) -> int:
        return int(min(max(y + h - 4 - (val - (lo - 0.25 * span)) / (1.5 * span) * (h - 8), y), y + h))

    cv2.line(frame, (x, ypix(0.5 * (lo + hi))), (x + w, ypix(0.5 * (lo + hi))), (120, 120, 120), 1)
    n = len(vals)
    pts = [(x + int(i * (w - 1) / max(n - 1, 1)), ypix(v)) for i, v in enumerate(vals)]
    cv2.polylines(frame, [np.array(pts, np.int32)], False, (255, 255, 255), 1)
    for i, st in enumerate(list(t.states)[-n:]):
        px = x + int(i * (w - 1) / max(n - 1, 1))
        cv2.line(frame, (px, y + h + 2), (px, y + h + 8), (0, 255, 255) if st else (60, 60, 60), 2)
    d = t.decoder
    cv2.putText(frame, f"T{t.tid} kontr {t.contrast:.0f} ramka {100 * d.coverage:.0f}%",
                (x, y - 16), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
    cv2.putText(frame, f"dopas {d.last_score:.2f} przewaga {d.last_margin:.2f}",
                (x, y - 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)


def draw(frame, rx: LedIdReceiver, drones, fps: float, debug: bool = False) -> None:
    """Rysuje na obrazie POMNIEJSZONYM (rx.small) - w nim są współrzędne torów."""
    if debug:
        draw_machine_view(frame, rx)
    else:
        for ex, ey, _, sign in rx.last_events:
            cv2.circle(frame, (int(ex), int(ey)), 1, (90, 90, 90) if sign > 0 else (60, 60, 60), -1)

    shown = []
    for t in rx.tracks:
        if t.drone_id is None and t.contrast < BLINK_CONTRAST:
            continue                       # szum / coś, co nie mruga - nie zaśmiecamy podglądu
        shown.append(t)
        if t.drone_id is None:
            color = (0, 200, 255)
        else:
            color = (0, 220, 0) if t.id_confirmed else (0, 160, 160)
        p = (int(t.x), int(t.y))
        if debug:
            x0, y0, x1, y1 = t.roi
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.circle(frame, p, int(max(6, 2 * t.radius)), color, 1)
        if not debug:
            bits = "".join(map(str, list(t.states)[-24:]))
            cv2.putText(frame, bits, (p[0] + 10, p[1] + 4), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)

    if debug:
        # wykresy dla najbardziej "mrugających" torów (z ID najpierw)
        # wykresy w kolumnie przy prawej krawędzi, przy kropce tylko numer toru
        best = sorted(shown, key=lambda t: (t.drone_id is None, -t.contrast))[:MAX_PLOTS]
        W = frame.shape[1]
        for k, t in enumerate(best):
            cv2.putText(frame, f"T{t.tid}", (int(t.x) + 8, int(t.y) - 8),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
            draw_signal(frame, t, W - 160, 110 + k * 85)

    for d in drones:
        c = (int(d.x), int(d.y))
        r = int(max(20, d.spread_px / 2 + 15))
        col = (0, 255, 0) if d.confirmed else (0, 200, 200)
        cv2.circle(frame, c, r, col, 2)
        cv2.putText(frame, f"DRON {d.drone_id}" + ("" if d.confirmed else "?"), (c[0] - r, c[1] + r + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(frame, f"az {d.azimuth_deg:+.1f} el {d.elevation_deg:+.1f}",
                    (c[0] - r, c[1] + r + 40), cv2.FONT_HERSHEY_PLAIN, 1.0, col, 1)

    per_chip = fps / rx.cfg.chip_hz
    cv2.putText(frame, f"{fps:4.1f} fps  prog energii {rx.cfg.energy_threshold:g}  chip {rx.cfg.chip_hz:g} Hz"
                f"  ({per_chip:.1f} klatek/chip)   [d] podglad maszyny",
                (10, 22), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    if debug:
        cv2.putText(frame, "niebieski=energia mrugania  zielony=jasnieje  czerwony=ciemnieje  kropki=szczyty  ramka=okno pomiaru",
                    (10, 64), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)
    if 0 < per_chip < 2.2:
        cv2.putText(frame, "ZA MALO KLATEK NA CHIP - zmniejsz HZ w Arduino",
                    (10, 44), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="plik wideo zamiast kamery")
    ap.add_argument("--ts", help="czasy klatek .npy (z recordings/X/ts.npy) - jak na żywo")
    ap.add_argument("--camera", type=int, help="indeks kamery (domyślnie pierwsza nie-czarna)")
    ap.add_argument("--chip-hz", type=float, default=ReceiverConfig.chip_hz)
    ap.add_argument("--color", default="none", choices=["red", "green", "blue", "yellow", "nonblue", "none"],
                    help="śledzenie po kolorze diody (ułatwienie); none = tylko jasność i mruganie, jak docelowo z IR")
    ap.add_argument("--threshold", type=float, default=ReceiverConfig.energy_threshold,
                    help="min energia mrugania kandydata na diodę (mniej = czulej)")
    ap.add_argument("--hfov", type=float, default=ReceiverConfig.hfov_deg, help="poziomy FOV kamery [deg]")
    ap.add_argument("--record", help="zapisz surowe klatki do pliku .mp4")
    ap.add_argument("--out", help="zapisz przetworzony obraz (z nakładką) do .mp4")
    ap.add_argument("--no-gui", action="store_true", help="bez okna podglądu")
    ap.add_argument("--debug", action="store_true", help="start w podglądzie \"co widzi maszyna\" (klawisz d)")
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="gotowy zestaw ustawień z detector.PRESETS (np. pi_niebo); --set nadpisuje dalej")
    ap.add_argument("--set", action="append", default=[], metavar="KLUCZ=WARTOŚĆ",
                    help="nadpisz pole ReceiverConfig, np. --set seek_source=False (można wiele)")
    args = ap.parse_args()

    cfg = ReceiverConfig(chip_hz=args.chip_hz, energy_threshold=args.threshold, hfov_deg=args.hfov,
                         led_color=None if args.color == "none" else args.color)
    for key, val in PRESETS.get(args.preset, {}).items():
        setattr(cfg, key, val)
    for item in args.set:
        key, val = item.split("=", 1)
        cur = getattr(cfg, key)
        setattr(cfg, key, (val == "True") if isinstance(cur, bool) else type(cur)(val))
    rx = LedIdReceiver(cfg)
    rx.debug = args.debug

    cam = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit("Nie da się otworzyć źródła wideo")
    else:
        cap = open_camera(args.camera)
        cam = CameraThread(cap)              # odczyt w tle - wolniejsza klatka nie gubi następnej

    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ts_list = np.load(args.ts) if args.ts else None
    writer = None
    out_writer = None
    frame_idx = 0
    fps_est, last_ts = 0.0, None
    seen = set()
    paused = False

    while True:
        if not paused:
            if cam is not None:
                item = cam.read()
                if item is None:
                    break
                frame, ts = item
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            if args.video and ts_list is not None:
                if frame_idx >= len(ts_list):
                    break
                ts = float(ts_list[frame_idx])
            elif args.video:
                # czas z kontenera (telefony nagrywają ze zmiennym FPS); fallback na numer klatki
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ts <= 0 and frame_idx > 0:
                    ts = frame_idx * 1000.0 / file_fps
            frame_idx += 1

            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                writer.write(frame)

            drones = rx.process(frame, ts)
            if last_ts is not None and ts > last_ts:
                fps_est = 0.9 * fps_est + 0.1 * 1000.0 / (ts - last_ts) if fps_est else 1000.0 / (ts - last_ts)
            last_ts = ts

            for d in drones:
                if d.drone_id not in seen:
                    seen.add(d.drone_id)
                    print(f"[{ts / 1000:7.2f}s] nowy dron ID={d.drone_id} az={d.azimuth_deg:+.1f} "
                          f"el={d.elevation_deg:+.1f} diod={d.n_leds}")

        if args.no_gui and not args.out:
            continue
        view = rx.small.copy()
        draw(view, rx, drones, fps_est, debug=rx.debug)
        if args.out and not paused:
            cv2.putText(view, f"t = {ts / 1000:5.2f} s", (10, view.shape[0] - 12),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2)
            if out_writer is None:
                # avc1 (H.264) otwiera się w QuickTime; mp4v jako zapas
                for fourcc in ("avc1", "mp4v"):
                    out_writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*fourcc),
                                                 file_fps, (view.shape[1], view.shape[0]))
                    if out_writer.isOpened():
                        break
            out_writer.write(view)
        if args.no_gui:
            continue
        cv2.imshow("vision_id", view)
        key = cv2.waitKey(1 if not args.video else max(1, int(1000 / file_fps))) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("d"):
            rx.debug = not rx.debug
        if key in (ord("+"), ord("=")):
            cfg.energy_threshold = min(200, cfg.energy_threshold + 3)
        if key == ord("-"):
            cfg.energy_threshold = max(3, cfg.energy_threshold - 3)

    if cam is not None:
        cam.release()
    else:
        cap.release()
    if writer is not None:
        writer.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()
    print(f"Zidentyfikowane drony: {sorted(seen) or 'brak'}")


if __name__ == "__main__":
    main()
