"""Live LED ID decoding from the Pi camera at 60 fps with the C++ receiver.

    python3 live.py [--seconds 30] [--exposure 250] [--keep 10] [--no-flip]

Prints fps, detector time and CPU load once a second. On exit (or Ctrl+C) the session is saved
to ~/recordings/live_<time>/: raw.avi + ts.npy + info.json of the last --keep seconds (the same
format as station recordings, so `build/ledrx_rec DIR` replays it) and log.jsonl with every
frame's result.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import MappedArray, Picamera2

sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))
import ledrx  # noqa: E402

W, H, FPS = 1332, 990, 60


def read_transmitter():
    try:
        import serial
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.3)
        time.sleep(2.0)
        ser.read(4096)
        ser.write(b"RUN\n")
        time.sleep(0.3)
        return ser.read(4096).decode(errors="replace").strip()
    except Exception as e:
        return f"no transmitter ({e})"


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def save(out: Path, frames, ts, n_kept, info, log):
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    if n_kept:
        order = np.argsort(ts[:n_kept])
        writer = cv2.VideoWriter(str(out / "raw.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        for i in order:
            writer.write(frames[i])
        writer.release()
        np.save(out / "ts.npy", ts[order] - ts[order[0]])
    (out / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--exposure", type=int, default=250, help="exposure time [us]")
    ap.add_argument("--chip-hz", type=float, default=20.0)
    ap.add_argument("--keep", type=float, default=10, help="seconds of raw video kept for the session (0 = none)")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--root", default=str(Path.home() / "recordings"))
    args = ap.parse_args()

    transmitter = read_transmitter()
    print("transmitter:", transmitter)
    cam = Picamera2()
    frame_us = int(1e6 / FPS)
    cam.configure(cam.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"}, sensor={"output_size": (W, H), "bit_depth": 10},
        transform=Transform() if args.no_flip else Transform(hflip=1, vflip=1), buffer_count=8,
        controls={"FrameDurationLimits": (frame_us, frame_us), "AeEnable": False, "AwbEnable": False,
                  "ExposureTime": args.exposure, "AnalogueGain": 1.0, "ColourGains": (2.0, 2.0)}))
    cam.start()

    rx = ledrx.Receiver(args.chip_hz)
    n_ring = int(args.keep * FPS)
    frames = np.empty((n_ring, H, W, 3), np.uint8) if n_ring else None
    ring_ts = np.zeros(n_ring)
    log = []
    t_first = None
    n = 0
    sec = {"frames": 0, "detect": 0.0, "wall": time.perf_counter(), "cpu": cpu_seconds(), "ids": set()}
    try:
        while True:
            req = cam.capture_request()
            try:
                ts = req.get_metadata()["SensorTimestamp"] / 1e6
                t_first = ts if t_first is None else t_first
                t = ts - t_first
                with MappedArray(req, "main") as m:
                    t0 = time.perf_counter()
                    drones = rx.process(m.array, t)
                    dt = time.perf_counter() - t0
                    if n_ring:
                        frames[n % n_ring] = m.array
                        ring_ts[n % n_ring] = t
            finally:
                req.release()
            n += 1
            if args.seconds and t >= args.seconds * 1000:
                break
            log.append({"ts": round(t, 3), "detect_ms": round(dt * 1000, 3),
                        "drones": [[d.drone_id, round(d.x, 2), round(d.y, 2), d.confirmed] for d in drones]})
            sec["frames"] += 1
            sec["detect"] += dt
            sec["ids"] |= {f"{d.drone_id}{'' if d.confirmed else '?'}" for d in drones}
            wall = time.perf_counter() - sec["wall"]
            if wall >= 1.0:
                cpu = cpu_seconds() - sec["cpu"]
                print(f"{sec['frames'] / wall:5.1f} fps  detect {1000 * sec['detect'] / sec['frames']:4.2f} ms  "
                      f"process CPU {100 * cpu / wall:5.1f}% of one core  IDs: {' '.join(sorted(sec['ids'])) or '-'}",
                      flush=True)
                sec = {"frames": 0, "detect": 0.0, "wall": time.perf_counter(), "cpu": cpu_seconds(), "ids": set()}
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
    n_kept = min(n, n_ring)
    info = {"size": [W, H], "fps_set": FPS, "frames": n_kept, "exposure_us": args.exposure,
            "arduino": transmitter, "source": "rt/live.py"}
    save(Path(args.root) / time.strftime("live_%Y%m%d_%H%M%S"), frames, ring_ts, n_kept, info, log)


if __name__ == "__main__":
    main()
