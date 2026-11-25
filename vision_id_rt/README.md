# vision_id_rt

Real-time drone ID from blinking LEDs, in C++, with the flight controller gyro. This is the
algorithm from the `kalman-and-hungarian` branch (Kalman filter + Hungarian assignment,
constellations, FC gyro compensation) in the fast C++ receiver from `vision-id-rt`. On the same
recordings it reports the same drones and confirmations as that branch's Python code in every frame.

Raspberry Pi 5, 1332x990 @ 60 fps: 3.8 ms per frame, 23% of one CPU core (Python: 25-34 ms).

Per frame:

1. camera rotation since the last frame from the FC gyro (or, without it, the image shift from
   phase correlation, `--motion-comp`) moves the history and the tracks,
2. blink energy of the LED colour map (max(R, G) - B) and its peaks, in one NEON pass,
3. tracks: Kalman prediction, Hungarian assignment of peaks, correction with the peak and then
   with the exact LED point,
4. constellations: tracks that switch on and off in the same frames are grouped before any ID
   is read; a group shares its ID, and two agreeing reads confirm it at once,
5. ID decoding (template matching over all codes, phases and clock rates), drones and bearings.

```
src/receiver.*       pipeline above
src/tracking.*       Kalman filter, Hungarian algorithm
src/constellation.*  grouping tracks by blinking
src/decoder.*        ID decoding
src/motion.*         phase correlation, rotation homography, image warping
src/fc.*             flight controller over MSP: gyro, attitude, calibration
src/kernels.hpp      whole-image passes, NEON
src/recording.*      station recordings: MJPG AVI, ts.npy
src/ledrx_rec.cpp    command-line tool for recordings
src/pyledrx.cpp      Python module used by live.py and fc_check.py
live.py              live decoding from the Pi camera, web preview
fc_check.py          flight controller link and mounting check
```

## Build (on the Pi)

```bash
sudo apt install build-essential libturbojpeg0-dev pybind11-dev python3-dev python3-picamera2 python3-opencv
make all python
```

## Run

Live from the camera, with a preview at http://<pi>:8000 (stop `station.py` first):

```bash
python3 live.py                      # without the flight controller
python3 live.py --fc                 # with the FC gyro (USB, /dev/ttyACM*)
```

The preview's Save button stores the last 15 s (video, timestamps and a per-frame log of tracks,
constellations and the gyro rotation) in `~/recordings/live_<time>/`.

## Flight controller

The FC (Betaflight / INAV) must answer MSP on its USB port, or on a UART with MSP enabled
(`--fc /dev/ttyAMA0`). Close Betaflight Configurator first.

1. `python3 fc_check.py` prints the gyro in FC axes and in camera axes. Pan the camera right, tilt
   it up and roll it clockwise (seen from behind): each should be positive in its own camera
   column only.
2. If the FC is not mounted flat with the camera looking forward, or the signs are wrong:
   `python3 live.py --fc-calibrate 15`. Point the camera at a still scene with detail and turn
   it up/down and left/right. It tries all 24 right-angle mountings and the latency between gyro
   and image, prints the agreement (trust it above ~0.7) and saves the result to `fc_mount.json`,
   which `live.py --fc` and `fc_check.py` then use. It also measures the gyro scale: the Betaflight
   build tested here sends raw sensor units over MSP (scale ~0.069) rather than deg/s.
3. The mounting can also be given by hand: `--fc-mount x=+pitch,y=+yaw,z=+roll` (camera x right,
   y down, z forward, each as a signed FC axis).

## Recordings

```bash
build/ledrx_rec ~/recordings/03_ruch --pace    # at the recording's frame times, like a camera
```

It reads `raw.avi`, `ts.npy` and `info.json`, prints timing and CPU load and writes `result_rt.json`.
