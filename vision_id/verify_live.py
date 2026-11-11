"""
Test na żywo: kamera Maca + Arduino na USB.

Skrypt pyta Arduino przez port szeregowy, jakie ID nadaje (prawda), a z obrazu
kamery dekoduje ID z mrugania diod. Na ekranie:
    DETECTED: MATCHING ID 5      - kamera odczytała to samo, co nadaje Arduino
    MISMATCH: kamera 7 != Arduino 5
    SZUKAM...                    - jeszcze nic nie zdekodowane

    uv run verify_live.py                    # port wykrywany automatycznie
    uv run verify_live.py --port /dev/cu.usbserial-XXXX --camera 1
    uv run verify_live.py --no-record        # bez zapisu sesji

Bez Arduino na USB (np. nadajnik z firmware/led_blink_id5 na zasilaczu):
    uv run verify_live.py --no-arduino --id 5 --hz 10   # porównuje z podanym ID
    uv run verify_live.py --no-arduino                  # pokazuje każde odczytane ID
Jeśli Arduino nie zostanie znalezione albo nie odpowiada, skrypt sam przechodzi w ten tryb.

Każda sesja zapisuje się do recordings/<data_godzina>/ (surowe klatki, czasy,
log dekodera) - ścieżkę wypisuje na końcu. Ctrl+C też poprawnie zamyka zapis.

Klawisze: q - wyjście, +/- - czułość, d - przełącz podgląd "co widzi maszyna".
"""

import argparse
import glob
import re
import time

import cv2
import serial

from detector import LedIdReceiver, ReceiverConfig
from protocol import encode_frame
from camera import CameraThread, open_camera
from receiver import draw
from recorder import SessionRecorder, config_dict


def find_port():
    ports = sorted(glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*")
                   + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def query_arduino(port: str):
    """Otwiera port (to resetuje Nano), włącza nadawanie i odczytuje ID oraz HZ."""
    ser = serial.Serial(port, 115200, timeout=0.3)
    time.sleep(2.5)                          # Nano startuje po resecie
    ser.reset_input_buffer()
    ser.write(b"RUN\n")
    deadline = time.time() + 3
    text = ""
    while time.time() < deadline:
        text += ser.read(200).decode(errors="replace")
        m = re.search(r"ID=(\d+) HZ=([\d.]+) ramka=[\d.]+s chipy=([01]+)", text)
        if m:
            drone_id, hz, chips = int(m.group(1)), float(m.group(2)), m.group(3)
            # Arduino i Python muszą mieć tę samą tablicę kodów (gen_codes.py)
            expected = "".join(map(str, encode_frame(drone_id)))
            if chips != expected:
                raise SystemExit(f"Arduino nadaje inny kod niż oczekuje Python dla ID {drone_id}:\n"
                                 f"  Arduino: {chips}\n  Python:  {expected}\n"
                                 "Wgraj aktualne firmware (firmware/led_beacon) - patrz README.")
            return ser, drone_id, hz
    ser.close()
    return None


def banner(frame, text: str, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    x, y = (frame.shape[1] - tw) // 2, frame.shape[0] - 30
    cv2.rectangle(frame, (x - 12, y - th - 14), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--no-arduino", action="store_true",
                    help="nie łącz się z Arduino (nadajnik działa sam, np. firmware/led_blink_id5)")
    ap.add_argument("--id", type=int, help="bez Arduino: oczekiwane ID nadajnika (domyślnie pokazuje każde odczytane)")
    ap.add_argument("--camera", type=int, help="indeks kamery (domyślnie pierwsza nie-czarna)")
    ap.add_argument("--color", default="none", choices=["red", "green", "blue", "yellow", "nonblue", "none"],
                    help="śledzenie po kolorze diody (ułatwienie); none = tylko jasność i mruganie, jak docelowo z IR")
    ap.add_argument("--hz", type=float, help="częstotliwość chipów nadajnika (z Arduino: domyślnie dobierana do "
                    "FPS kamery; bez Arduino: domyślnie 10, jak w firmware/led_blink_id5)")
    ap.add_argument("--threshold", type=float, default=ReceiverConfig.energy_threshold,
                    help="min energia mrugania kandydata na diodę (mniej = czulej)")
    ap.add_argument("--no-record", action="store_true", help="nie zapisuj sesji do recordings/")
    args = ap.parse_args()

    ser, port, true_id, chip_hz = None, None, args.id, None
    if not args.no_arduino:
        port = args.port or find_port()
        if port is None:
            print("Nie widzę Arduino na USB - działam bez niego (--no-arduino).")
        else:
            got = query_arduino(port)
            if got is None:
                if args.port:
                    raise SystemExit(f"Arduino nie odpowiada na {port}. Bez Arduino: --no-arduino")
                print(f"Arduino na {port} nie odpowiada - działam bez niego (--no-arduino).")
                port = None
            else:
                ser, true_id, chip_hz = got
                print(f"Arduino na {port}: nadaje ID={true_id}, {chip_hz:g} Hz")

    cam = CameraThread(open_camera(args.camera))   # odczyt w tle - wolniejsza klatka nie gubi następnej
    cam_fps = cam.measure_fps()
    if ser is not None:
        # ~3 klatki na chip: 30 fps -> 10 Hz, 60 fps -> 20 Hz
        want_hz = 20.0 if cam_fps >= 50 else 10.0
        if args.hz:
            want_hz = args.hz
        if want_hz != chip_hz:
            ser.write(f"HZ {want_hz:g}\n".encode())
            time.sleep(0.3)
            ser.read(500)
            chip_hz = want_hz
        print(f"Kamera: {cam_fps:.1f} fps -> Arduino nadaje z {chip_hz:g} Hz ({cam_fps / chip_hz:.1f} klatki/chip)")
    else:
        # bez Arduino nie da się zmienić tempa nadajnika - dekoder musi znać je z góry
        chip_hz = args.hz or 10.0
        target = f"ID {true_id}" if true_id is not None else "dowolne ID"
        print(f"Kamera: {cam_fps:.1f} fps, nadajnik {chip_hz:g} Hz ({cam_fps / chip_hz:.1f} klatki/chip), szukam: {target}")
        if cam_fps / chip_hz < 2.5:
            print("  Uwaga: mniej niż ~3 klatki na chip - ustaw wolniejszy nadajnik albo --hz zgodne z nadajnikiem.")

    cfg = ReceiverConfig(chip_hz=chip_hz, energy_threshold=args.threshold,
                         led_color=None if args.color == "none" else args.color)
    rx = LedIdReceiver(cfg)
    rx.debug = True                           # od razu podgląd "co widzi maszyna" (klawisz d)

    rec = None if args.no_record else SessionRecorder({
        "arduino": {"port": port, "id": true_id, "hz": chip_hz,
                    "chips": "".join(map(str, encode_frame(true_id))) if true_id is not None else None},
        "camera": {"index": args.camera, "fps": round(cam_fps, 2)},
        "config": config_dict(cfg),
    })

    state = {"first_match_ts": None, "first_confirmed_ts": None, "ok": 0, "bad": 0, "bad_tentative": 0}
    try:
        loop(cam, rx, cfg, rec, true_id, state)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if ser is not None:
            ser.close()
        cv2.destroyAllWindows()
    first = f"{state['first_match_ts'] / 1000:.1f}s" if state["first_match_ts"] else "nigdy"
    conf = f"{state['first_confirmed_ts'] / 1000:.1f}s" if state["first_confirmed_ts"] else "nigdy"
    print(f"Podsumowanie: pierwsze trafienie {first}, potwierdzone {conf}, dopasowania zgodne {state['ok']}, "
          f"błędne potwierdzone {state['bad']}, błędne wstępne {state['bad_tentative']}")
    if rec is not None:
        out = rec.close({"first_match_s": first, "first_confirmed_s": conf,
                         "matches_ok": state["ok"], "matches_bad_confirmed": state["bad"],
                         "matches_bad_tentative": state["bad_tentative"], "dropped_frames": cam.dropped})
        print(f"Sesja zapisana: {out}")


def loop(cam, rx, cfg, rec, true_id, state) -> None:
    proc_ms = 0.0
    seen_frames = {}                          # tid -> liczba dopasowań, żeby liczyć tylko nowe
    seen_ids = {}                             # bez znanego ID: id -> czy już potwierdzone (do wypisywania)
    while True:
        item = cam.read()
        if item is None:
            break
        frame, ts = item                      # czas = moment zrobienia klatki, nie przetworzenia
        t_start = time.perf_counter()
        drones = rx.process(frame, ts)
        fps = cam.fps
        if rec is not None:
            rec.frame(frame, ts, rx, drones)
        for d in drones:
            if (true_id is None or d.drone_id == true_id) and d.confirmed and state["first_confirmed_ts"] is None:
                state["first_confirmed_ts"] = ts
                print(f"[{ts / 1000:6.2f}s] POTWIERDZONE ID {d.drone_id}")

        # każde nowe dopasowanie z każdej diody: zgodne czy nie. Błędne liczymy
        # osobno dla wstępnych (pojedyncze dopasowanie) i potwierdzonych.
        for t in rx.tracks:
            n = t.decoder.frames_ok
            if n > seen_frames.get(t.tid, 0):
                seen_frames[t.tid] = n
                dec = t.decoder
                if true_id is None:           # bez znanego ID: tylko wypisz, co widać
                    if state["first_match_ts"] is None:
                        state["first_match_ts"] = ts
                    state["ok"] += 1
                    if dec.confirmed_id != dec.last_id or not seen_ids.get(dec.last_id):
                        seen_ids[dec.last_id] = dec.confirmed_id == dec.last_id
                        print(f"[{ts / 1000:6.2f}s] ID {dec.last_id}"
                              + (" (potwierdzone)" if seen_ids[dec.last_id] else " (wstepnie)")
                              + f" - tor T{t.tid}, dopas {dec.last_score:.2f}, przewaga {dec.last_margin:.2f}")
                elif dec.last_id == true_id:
                    state["ok"] += 1
                    if state["first_match_ts"] is None:
                        state["first_match_ts"] = ts
                        print(f"[{ts / 1000:6.2f}s] DETECTED (wstepnie): ID {true_id}")
                elif dec.confirmed_id == dec.last_id:
                    state["bad"] += 1
                    print(f"[{ts / 1000:6.2f}s] MISMATCH (potwierdzone): kamera {dec.last_id} != Arduino {true_id} "
                          f"(tor T{t.tid}, dopas {dec.last_score:.2f}, przewaga {dec.last_margin:.2f})")
                else:
                    state["bad_tentative"] += 1
                    print(f"[{ts / 1000:6.2f}s] wstepnie inne ID {dec.last_id} (nie potwierdzone; tor T{t.tid}, "
                          f"dopas {dec.last_score:.2f}, przewaga {dec.last_margin:.2f})")

        # podgląd rysujemy tylko, gdy nie ma zaległości - dekodowanie ma pierwszeństwo
        if cam.backlog > 1:
            continue
        view = rx.small.copy()
        draw(view, rx, drones, fps, debug=rx.debug)
        if true_id is None:
            mine, wrong_conf, wrong_tent = list(drones), [], []
        else:
            mine = [d for d in drones if d.drone_id == true_id]
            wrong_conf = [d for d in drones if d.drone_id != true_id and d.confirmed]
            wrong_tent = [d for d in drones if d.drone_id != true_id and not d.confirmed]
        if mine:
            d = max(mine, key=lambda d: d.confirmed)
            label = f"DETECTED: MATCHING ID {true_id}" if true_id is not None else f"DETECTED: ID {d.drone_id}"
            banner(view, label + ("" if d.confirmed else " (wstepnie)"),
                   (0, 255, 0) if d.confirmed else (0, 200, 200))
            cv2.putText(view, f"az {d.azimuth_deg:+.1f} deg  el {d.elevation_deg:+.1f} deg",
                        (10, 110), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2)
        elif wrong_conf:
            banner(view, f"MISMATCH: kamera {wrong_conf[0].drone_id} != oczekiwane {true_id}", (0, 0, 255))
        elif wrong_tent:
            banner(view, f"wstepnie {wrong_tent[0].drone_id}? (czeka na potwierdzenie)", (0, 200, 255))
        else:
            banner(view, f"SZUKAM... (oczekiwane ID {true_id})" if true_id is not None else "SZUKAM...",
                   (0, 200, 255))
        proc_ms = 0.9 * proc_ms + 0.1 * (time.perf_counter() - t_start) * 1000.0
        cv2.putText(view, f"dopasowania zgodne {state['ok']}  bledne potw. {state['bad']}  wstepne {state['bad_tentative']}"
                    f"   {proc_ms:.0f} ms/klatke"
                    f"  zgubione klatki {cam.dropped}",
                    (10, 88), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

        cv2.imshow("vision_id - weryfikacja", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return
        if key == ord("d"):
            rx.debug = not rx.debug
        if key in (ord("+"), ord("=")):
            cfg.energy_threshold = min(200, cfg.energy_threshold + 3)
        if key == ord("-"):
            cfg.energy_threshold = max(3, cfg.energy_threshold - 3)



if __name__ == "__main__":
    main()
