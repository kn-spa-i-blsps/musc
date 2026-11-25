"""Check the flight controller link and how it is mounted relative to the camera.

    python3 fc_check.py [--port /dev/ttyACM0] [--mount x=+pitch,y=+yaw,z=+roll]

Prints the gyro four times a second in FC axes and in camera axes. Move the camera and check the
camera columns: pan right, tilt up and roll clockwise (seen from behind the camera) should each
give a positive number in their own column and about zero in the others. If not, run
`python3 live.py --fc-calibrate 15`, which finds the mounting from the image.

The mounting is saved by --fc-calibrate in fc_mount.json next to this script and used by default.
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))
import ledrx  # noqa: E402

AXES = ("roll", "pitch", "yaw")
CAM = ("x", "y", "z")
MOUNT_FILE = Path(__file__).resolve().parent / "fc_mount.json"


def parse_mount(text):
    """'x=+pitch,y=+yaw,z=+roll' -> row-major 3x3 (camera = M * (roll, pitch, yaw))."""
    m = [0.0] * 9
    for part in text.split(","):
        cam, fc = part.split("=")
        sign = -1.0 if fc.startswith("-") else 1.0
        m[3 * CAM.index(cam.strip()) + AXES.index(fc.strip().lstrip("+-"))] = sign
    return m


def format_mount(m):
    parts = []
    for r, cam in enumerate(CAM):
        row = m[3 * r:3 * r + 3]
        k = max(range(3), key=lambda i: abs(row[i]))
        parts.append(f"{cam}={'+' if row[k] > 0 else '-'}{AXES[k]}")
    return ",".join(parts)


def load_mount():
    """(mount matrix, latency ms, gyro scale) saved by live.py --fc-calibrate, or the defaults."""
    try:
        d = json.loads(MOUNT_FILE.read_text())
        return parse_mount(d["mount"]), float(d["latency_ms"]), float(d.get("gyro_scale", 1.0))
    except (OSError, ValueError, KeyError):
        return ledrx.mount_matrix(0.0), 0.0, 1.0


def find_port():
    """A /dev/serial/by-id path when there is one: it survives the FC re-enumerating after a USB reset."""
    ports = sorted(glob.glob("/dev/serial/by-id/*Betaflight*") + glob.glob("/dev/serial/by-id/*INAV*"))
    ports += sorted(glob.glob("/dev/ttyACM*"))
    if not ports:
        raise SystemExit("No flight controller on USB (/dev/ttyACM*). Use --port for a UART, e.g. /dev/ttyAMA0.")
    return ports[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mount", help="camera axes from FC axes, e.g. x=+pitch,y=+yaw,z=+roll")
    args = ap.parse_args()

    mount, latency, scale = load_mount()
    if args.mount:
        mount = parse_mount(args.mount)
    port = args.port or find_port()
    fc = ledrx.FcTelemetry(port, args.baud, mount, latency, scale)
    print(f"FC {fc.variant} on {port}, mount {format_mount(mount)}, gyro scale {scale:.4g}. Ctrl+C to stop.")
    print("            FC [deg/s]: roll  pitch    yaw |  camera [deg/s]: tilt up  pan right  roll cw | attitude r/p/y")
    try:
        last = ledrx.monotonic_ms()
        while True:
            time.sleep(0.25)
            now = ledrx.monotonic_ms()
            body, cam = fc.rotation_body(last, now), fc.rotation(last, now)
            dt = (now - last) / 1000
            last = now
            if body is None:
                print(f"{fc.rate_hz:5.0f} Hz  no gyro data (errors {fc.errors})")
                continue
            b = "".join(f"{math.degrees(v) / dt:7.1f}" for v in body)
            c = "".join(f"{math.degrees(v) / dt:10.1f}" for v in cam)
            att = fc.attitude
            a = " ".join(f"{v:6.1f}" for v in att) if att else "-"
            print(f"{fc.rate_hz:5.0f} Hz  {b}  |  {c} | {a}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        fc.close()


if __name__ == "__main__":
    main()
