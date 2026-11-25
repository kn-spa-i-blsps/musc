# Live recordings with the flight controller gyro (2026-11-25)

Pi 5 + IMX477, 1332x990, exposure 250 us, transmitter ID 5 at 20 Hz (red + green LED). The
Betaflight FC was taped to the camera (upside down, perpendicular to it) and its gyro compensated
the camera rotation. The light and the camera were moved at the same time, which is where the ID
was lost most often. `fc_mount.json` is the FC calibration these were recorded with.

| clip | real fps | camera speed (median / 90%) | LED visible | ID while visible (confirmed) | false IDs |
|---|---|---|---|---|---|
| live_20261125_110140 | 46 | 16 / 41 deg/s | 99% | 89% (84%) | 0 |
| live_20261125_110404 | 46 | 12 / 27 deg/s | 100% | 84% (84%) | 0 |

The frame rate dropped to 46 fps because warping the history under camera motion was still slow
when these were recorded.

Replayed offline (same detector, frames with ID 5):

| clip | with gyro | without gyro | with gyro, `snap_px` 30 |
|---|---|---|---|
| live_20261125_110140 | 78% | 47% | 81% |
| live_20261125_110404 | 86% | 59% | 94% |

Files per clip:

- `raw.avi`: MJPG video, `ts.npy`: frame times [ms, float64], `info.json`: capture settings
- `log.jsonl`: what the live decoder saw, one line per frame:
  - `ts` [ms], `detect_ms`
  - `drones`: `[id, x, y, confirmed]`
  - `tracks`: `[tid, x, y, id, confirmed, level, lo, hi, group]`
  - `cam_rot`: camera rotation since the previous frame from the gyro (rotation vector in camera
    axes, rad: x right, y down, z forward), or null
- `overlay.mp4`: the video with the decoder overlay and an ID timeline (green ID, yellow
  tentative, red none)

Replay with the gyro: `python3 -c` around `ledrx.Receiver().process(frame, ts, cam_rot)`; the
frames read with `cv2.VideoCapture(path, cv2.CAP_OPENCV_MJPEG)` match what `build/ledrx_rec` decodes.
