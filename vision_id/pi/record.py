"""
Nagranie testowe z ręczną ekspozycją, w formacie vision_id/recorder.py:
    OUT/raw.avi (MJPG), OUT/ts.npy (czas klatki z sensora, ms), OUT/info.json
Odtworzenie na Macu:  uv run receiver.py --video raw.avi --ts ts.npy --chip-hz 20

    python3 record.py OUT --seconds 10 --exposure 250 [--gain 1] [--fps 60] [--arduino RUN]

Klatki trzymamy w RAM i kodujemy dopiero po nagraniu: kodowanie MJPG 1332x990 na
bieżąco nie wyrabia 60 fps. 10 s = ok. 2.4 GB RAM, więc nagrania do ~20 s.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import serial
from libcamera import Transform
from picamera2 import Picamera2

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=10)
ap.add_argument("--exposure", type=int, default=250, help="czas naświetlania [us]")
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--fps", type=float, default=60)
ap.add_argument("--size", default="1332x990")
ap.add_argument("--sensor", default="1332x990", help="tryb sensora (1332x990 do 120 fps, 2028x1080 do 50 fps)")
ap.add_argument("--lead", type=float, default=0, help="dodatkowe sekundy przed odliczaniem (czas na dojście)")
ap.add_argument("--countdown", type=float, default=0, help="ile sekund po sygnale ODLICZANIE zaczyna się nagranie")
ap.add_argument("--arduino", default="RUN", help="komenda dla nadajnika przed nagraniem ('' = nie ruszaj)")
ap.add_argument("--rot", type=int, default=180, choices=[0, 180], help="obrót obrazu: 180 = kamera zamontowana do góry nogami, 0 = bez obrotu")
args = ap.parse_args()
if args.seconds > 20:
    raise SystemExit("Najwyżej 20 s (klatki są w RAM)")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
w, h = map(int, args.size.split("x"))
sw, sh = map(int, args.sensor.split("x"))

ard_cfg = None
if args.arduino:
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.3)
    time.sleep(2.0)
    ser.read(4096)
    ser.write((args.arduino + "\n").encode())
    time.sleep(0.3)
    ard_cfg = ser.read(4096).decode(errors="replace").strip()

cam = Picamera2()
frame_us = int(1e6 / args.fps)
cam.configure(cam.create_video_configuration(
    main={"size": (w, h), "format": "RGB888"},
    sensor={"output_size": (sw, sh), "bit_depth": 10 if (sw, sh) == (1332, 990) else 12},
    transform=Transform(hflip=int(args.rot == 180), vflip=int(args.rot == 180)), buffer_count=8,
    controls={"FrameDurationLimits": (frame_us, frame_us), "AeEnable": False, "AwbEnable": False,
              "ExposureTime": args.exposure, "AnalogueGain": args.gain, "ColourGains": (2.0, 2.0)}))
cam.start()
time.sleep(1.0 + args.lead)
if args.countdown:
    print("ODLICZANIE", flush=True)       # Mac gra pierwszy dźwięk
    time.sleep(args.countdown)
print("START", flush=True)

n = int(args.seconds * args.fps)
frames = np.empty((n, h, w, 3), np.uint8)
ts = np.empty(n)
md = {}
for i in range(n):
    req = cam.capture_request()
    frames[i] = req.make_array("main")
    md = req.get_metadata()
    ts[i] = md["SensorTimestamp"] / 1e6
    req.release()
print("KONIEC", flush=True)               # Mac gra drugi dźwięk
sensor_mode = cam.camera_configuration()["sensor"]
cam.stop()

writer = cv2.VideoWriter(str(out / "raw.avi"), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))
for f in frames:
    writer.write(f)
writer.release()

ts -= ts[0]
np.save(out / "ts.npy", ts)
dt = np.diff(ts)
info = {"sensor_mode": sensor_mode, "exposure_us": md.get("ExposureTime"), "gain": md.get("AnalogueGain"),
        "size": [w, h], "fps_set": args.fps, "frames": n, "fps_measured": round(1000 / dt.mean(), 2),
        "dt_max_ms": round(float(dt.max()), 2), "gaps": int((dt > 1.5 * 1000 / args.fps).sum()),
        "arduino": ard_cfg}
(out / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
print(json.dumps(info, ensure_ascii=False))
