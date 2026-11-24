# vision_id_rt

Real-time drone ID from blinking LEDs, in C++. It finds the LEDs in each camera frame, tracks
them, decodes the 26-chip ID code (128 IDs) and groups LEDs with the same ID into a drone
with its bearing angle. The algorithm is the same as the Python detector on the
`vision-id-detection` branch (station settings), and the output matches it frame by frame.

Raspberry Pi 5, 1332x990 @ 60 fps: 3.5 ms per frame, 21% of one CPU core (Python: 25-34 ms).

```
src/receiver.*    LED detection, tracking, grouping into drones
src/decoder.*     ID decoding (template matching over all codes, phases and clock rates)
src/kernels.hpp   whole-image passes, NEON
src/recording.*   station recordings: MJPG AVI, ts.npy
src/ledrx_rec.cpp command-line tool for recordings
src/pyledrx.cpp   Python module (pybind11) used by live.py
live.py           live decoding from the Pi camera
```

## Build

```bash
sudo apt install build-essential libturbojpeg0-dev   # macOS: brew install jpeg-turbo
make            # build/ledrx_rec

sudo apt install pybind11-dev python3-dev python3-picamera2 python3-opencv
make python     # module for live.py
```

## Run

Live from the Pi camera with a web preview at http://<pi>:8000 (stop `station.py` first, only
one program can use the camera):

```bash
python3 live.py --seconds 30
```

It prints fps, detector time, CPU load and the IDs it sees once a second (camera + detector:
31% of one core at 60 fps). The session is saved to `~/recordings/live_<time>/`: the last
`--keep` seconds of video in the station format, plus `log.jsonl` with each frame's result.

Recordings:

```bash
build/ledrx_rec ~/recordings/03_ruch           # as fast as possible
build/ledrx_rec ~/recordings/03_ruch --pace    # at the recording's frame times, like a camera
```

It reads `raw.avi`, `ts.npy` and `info.json` (the transmitter ID from `ID=5`), prints timing
and CPU load, and writes `result_rt.json`. `--dump frames.jsonl` saves every frame's tracks
and drones.

In code:

```python
import ledrx
rx = ledrx.Receiver(chip_hz=20)
drones = rx.process(bgr_frame, ts_ms)   # HxWx3 uint8 numpy array, padded rows are fine
```

```cpp
ledrx::Config cfg;            // cfg.chip_hz = 20 for a 60 fps camera
ledrx::Receiver rx(cfg);
auto drones = rx.process(bgr, width, height, stride, ts_ms);
```
