"""Live LED ID decoding from the Pi camera at 60 fps with the C++ receiver.

    python3 live.py [--seconds 30] [--exposure 250] [--keep 10] [--port 8000] [--no-flip]

Open http://<pi>:8000 for the preview with the decoder overlay. Stats are also printed once a
second. On exit (Ctrl+C) the session is saved to ~/recordings/live_<time>/: raw.avi + ts.npy +
info.json of the last --keep seconds (replay with `build/ledrx_rec DIR`) and log.jsonl with
every frame's result.
"""
import argparse
import json
import resource
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import MappedArray, Picamera2

sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))
import ledrx  # noqa: E402

W, H, FPS = 1332, 990, 60
PREVIEW_W, PREVIEW_EVERY = 888, 4
GAMMA = np.array([min(255, int(255 * (i / 255) ** 0.45)) for i in range(256)], np.uint8)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LED ID live</title>
<style>
  body { background:#111; color:#ddd; font-family:system-ui, sans-serif; margin:12px; }
  img { max-width:100%; width:888px; background:#000; border:1px solid #333; display:block; }
  #ids { font-size:30px; font-weight:700; color:#81c784; min-height:38px; }
  #stat, #tx { color:#999; margin:4px 0; }
  button { background:#444; color:#fff; border:0; border-radius:4px; padding:6px 12px; margin-right:4px; cursor:pointer; }
  button.on { background:#2d5d8a; }
</style></head><body>
<div id="ids">-</div>
<div id="stat">connecting...</div>
<div id="tx"></div>
<div style="margin:8px 0">Exposure [us]: <span id="exp"></span></div>
<img src="/stream" alt="preview">
<script>
const EXPS = ["auto", "4000", "1000", "250", "125"];
async function setExp(v) { await fetch("/exposure?v=" + v, {method: "POST"}); }
async function poll() {
  try {
    const s = await (await fetch("/status")).json();
    ids.textContent = s.ids.length ? s.ids.map(i => "DRONE " + i).join("   ") : "-";
    stat.textContent = `${s.fps} fps   detector ${s.detect_ms} ms/frame   process CPU ${s.cpu}% of one core`;
    tx.textContent = "transmitter: " + s.transmitter;
    exp.innerHTML = EXPS.map(v => `<button class="${v == s.exposure ? 'on' : ''}" onclick="setExp('${v}')">${v}</button>`).join("");
  } catch (e) { stat.textContent = "no connection"; }
  setTimeout(poll, 500);
}
poll();
</script></body></html>
"""


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
        return f"none ({e})"


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Preview:
    """Renders every PREVIEW_EVERY-th frame with the overlay into a JPEG for the web page."""

    def __init__(self):
        self.cond = threading.Condition()
        self.pending = None
        self.jpeg = b""
        self.seq = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def offer(self, frame, tracks, drones, peaks):
        with self.cond:
            if self.pending is None:
                self.pending = (frame.copy(), tracks, drones, peaks)
                self.cond.notify_all()

    def wait_jpeg(self, seen):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seen, timeout=2.0)
            return self.jpeg, self.seq

    def _loop(self):
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.pending is not None)
                frame, tracks, drones, peaks = self.pending
            img = self._draw(frame, tracks, drones, peaks)
            jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            with self.cond:
                self.jpeg, self.pending = jpeg, None
                self.seq += 1
                self.cond.notify_all()

    @staticmethod
    def _draw(frame, tracks, drones, peaks):
        s = PREVIEW_W / frame.shape[1]
        img = cv2.LUT(cv2.resize(frame, (PREVIEW_W, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA), GAMMA)
        for x, y, _ in peaks:
            cv2.circle(img, (int(x * s), int(y * s)), 1, (110, 110, 110), -1)
        for t in tracks:
            if t["drone_id"] is None and t["contrast"] < 6:
                continue
            color = (0, 200, 255) if t["drone_id"] is None else (0, 220, 0) if t["id_confirmed"] else (0, 160, 160)
            p = (int(t["x"] * s), int(t["y"] * s))
            cv2.circle(img, p, int(max(6, 2 * t["radius"] * s)), color, 1)
            bits = "".join(map(str, t["states"][-24:]))
            cv2.putText(img, bits, (p[0] + 10, p[1] + 4), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)
        for d in drones:
            c = (int(d.x * s), int(d.y * s))
            r = int(max(20, d.spread_px * s / 2 + 15))
            color = (0, 255, 0) if d.confirmed else (0, 200, 200)
            cv2.circle(img, c, r, color, 2)
            cv2.putText(img, f"DRONE {d.drone_id}" + ("" if d.confirmed else "?"), (c[0] - r, c[1] + r + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, f"az {d.azimuth_deg:+.1f} el {d.elevation_deg:+.1f}", (c[0] - r, c[1] + r + 40),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)
        return img


def serve(port, preview, status, set_exposure):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/status":
                self._send(json.dumps(status).encode(), "application/json")
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        jpeg, seen = preview.wait_jpeg(seen)
                        if jpeg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == "/exposure":
                set_exposure(parse_qs(url.query).get("v", ["250"])[0])
                self._send(b"ok", "text/plain")
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()


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
    ap.add_argument("--exposure", default="250", help="exposure time [us] or auto")
    ap.add_argument("--chip-hz", type=float, default=20.0)
    ap.add_argument("--keep", type=float, default=10, help="seconds of raw video kept for the session (0 = none)")
    ap.add_argument("--port", type=int, default=8000, help="web preview port (0 = off)")
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
        controls={"FrameDurationLimits": (frame_us, frame_us)}))
    cam.start()

    status = {"fps": 0, "detect_ms": 0, "cpu": 0, "ids": [], "transmitter": transmitter, "exposure": args.exposure}

    def set_exposure(v):
        if v == "auto":
            cam.set_controls({"AeEnable": True, "AwbEnable": True})
        else:
            cam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": int(v),
                              "AnalogueGain": 1.0, "ColourGains": (2.0, 2.0)})
        status["exposure"] = v

    set_exposure(args.exposure)
    preview = Preview() if args.port else None
    if preview:
        serve(args.port, preview, status, set_exposure)
        print(f"preview: http://{__import__('socket').gethostname()}.local:{args.port}")

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
                    if preview and n % PREVIEW_EVERY == 0:
                        preview.offer(m.array, rx.tracks(), drones, rx.peaks)
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
                status.update(fps=round(sec["frames"] / wall, 1), detect_ms=round(1000 * sec["detect"] / sec["frames"], 2),
                              cpu=round(100 * cpu / wall, 1), ids=sorted(sec["ids"]))
                print(f"{status['fps']:5.1f} fps  detect {status['detect_ms']:4.2f} ms  "
                      f"process CPU {status['cpu']:5.1f}% of one core  IDs: {' '.join(status['ids']) or '-'}", flush=True)
                sec = {"frames": 0, "detect": 0.0, "wall": time.perf_counter(), "cpu": cpu_seconds(), "ids": set()}
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
    n_kept = min(n, n_ring)
    info = {"size": [W, H], "fps_set": FPS, "frames": n_kept, "exposure_us": status["exposure"],
            "arduino": transmitter, "source": "live.py"}
    save(Path(args.root) / time.strftime("live_%Y%m%d_%H%M%S"), frames, ring_ts, n_kept, info, log)


if __name__ == "__main__":
    main()
