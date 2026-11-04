"""
Seria czasów naświetlania dla testu diod (kamera i nadajnik nieruchome).

1. Nadajnik w trybie RUN (mruga kodem), 90 klatek przy stałej ekspozycji ->
   max i min każdego piksela; różnica max-min pokazuje, gdzie są diody.
2. Nadajnik ON (świeci stale), po jednej klatce na każdy czas naświetlania.
Zapis PNG (bez strat - przy krótkich czasach liczą się małe wartości) do OUT.
   python3 sweep.py [OUT]
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import serial
from picamera2 import Picamera2

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep")
EXPOSURES = [8000, 4000, 2000, 1000, 500, 250, 125, 60]
FIXED = {"AeEnable": False, "AnalogueGain": 1.0, "AwbEnable": False, "ColourGains": (2.0, 2.0)}
OUT.mkdir(parents=True, exist_ok=True)

ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.3)
time.sleep(2.0)
ser.read(4096)


def ard(cmd: str) -> None:
    ser.write((cmd + "\n").encode())
    time.sleep(0.3)
    ser.read(4096)


cam = Picamera2()
cam.configure(cam.create_video_configuration(main={"size": (1332, 990), "format": "RGB888"},
                                             controls={"FrameDurationLimits": (16667, 16667)}))
cam.start()


def settle(us=None) -> dict:
    cam.set_controls({"AeEnable": True} if us is None else {**FIXED, "ExposureTime": us})
    time.sleep(1.0 if us is None else 0.3)
    for _ in range(30):
        md = cam.capture_metadata()
        if us is None or abs(md["ExposureTime"] - us) <= max(20, 0.05 * us):
            return md
    return md


info = {}
ard("RUN")
md = settle(4000)
frames = [cam.capture_array("main") for _ in range(90)]
st = np.stack(frames)
cv2.imwrite(str(OUT / "blink_max.png"), st.max(axis=0))
cv2.imwrite(str(OUT / "blink_min.png"), st.min(axis=0))
info["blink"] = {"ExposureTime": md["ExposureTime"], "AnalogueGain": md["AnalogueGain"]}

ard("ON")
md = settle(None)
cv2.imwrite(str(OUT / "exp_auto.png"), cam.capture_array("main"))
info["auto"] = {"ExposureTime": md["ExposureTime"], "AnalogueGain": md["AnalogueGain"]}
for us in EXPOSURES:
    md = settle(us)
    cv2.imwrite(str(OUT / f"exp_{us}.png"), cam.capture_array("main"))
    info[str(us)] = {"ExposureTime": md["ExposureTime"], "AnalogueGain": md["AnalogueGain"]}
cam.stop()
(OUT / "info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info))
