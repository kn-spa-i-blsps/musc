"""
Stanowisko testowe na Raspberry Pi: podgląd kamery w przeglądarce, nagrywanie z dźwiękami
i automatyczny odczyt ID na Pi po każdym nagraniu.

    python3 station.py [--port 8000] [--exposure 250]
    przeglądarka (na Macu): http://169.254.12.2:8000

Kamera: IMX477 1332x990 @ 60 fps (wycinek środka sensora), obraz obrócony o 180°.
"Nagraj": X s na dojście -> dźwięk (3 s do startu) -> N s nagrania -> dźwięk końca.
Dźwięki gra przeglądarka na laptopie (zaplanowane od razu po kliknięciu, więc
działają też przy karcie w tle). Nagranie trafia do ~/recordings/NN_nazwa/:
    raw.avi + ts.npy + info.json    (ten sam format co record.py / vision_id/recorder.py)
Potem decode_rec.py liczy ID (na Pi ~29 klatek/s, czyli ~2x wolniej niż nagranie)
i robi film z nakładką - wynik pojawia się na stronie, gdy będzie gotowy.

Nadajnik (Arduino na /dev/ttyUSB0) jest otwarty przez cały czas, więc się nie resetuje.
Kamerę może mieć tylko jeden program: przed record.py / sweep.py zatrzymaj station.py.
"""

import argparse
import json
import shutil
import subprocess
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2

HERE = Path(__file__).resolve().parent
W, H, FPS = 1332, 990, 60
MIN_FREE_MB = 500
GAMMA = np.array([min(255, int(255 * (i / 255) ** 0.45)) for i in range(256)], np.uint8)

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--exposure", type=int, default=250, help="czas naświetlania nagrań [us]")
ap.add_argument("--root", default=str(Path.home() / "recordings"))
ap.add_argument("--rot", type=int, default=180, choices=[0, 180], help="obrót obrazu: 180 = kamera zamontowana do góry nogami, 0 = bez obrotu")
args = ap.parse_args()
ROOT = Path(args.root)
ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ nadajnik
ard_cfg = None
try:
    import serial
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.3)
    time.sleep(2.0)                      # otwarcie portu resetuje Nano
    ser.read(4096)
    ser.write(b"RUN\n")
    time.sleep(0.3)
    ard_cfg = ser.read(4096).decode(errors="replace").strip()
except Exception as e:                   # bez nadajnika też działa (np. nagranie samego tła)
    ard_cfg = f"brak nadajnika ({e})"

# ------------------------------------------------------------------ kamera
cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (W, H), "format": "RGB888"}, sensor={"output_size": (W, H), "bit_depth": 10},
    transform=Transform(hflip=int(args.rot == 180), vflip=int(args.rot == 180)), buffer_count=8,
    controls={"FrameDurationLimits": (int(1e6 / FPS), int(1e6 / FPS))}))
cam.start()

lock = threading.Lock()
st = {"mode": "idle", "msg": "", "preview_exp": str(args.exposure), "rec": None}
latest = {"frame": None, "n": 0, "md": {}}
jpeg = {"data": b"", "n": 0}
jpeg_cond = threading.Condition()


def set_exposure(v: str) -> None:
    if v == "auto":
        cam.set_controls({"AeEnable": True, "AwbEnable": True})
    else:
        cam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": int(v),
                          "AnalogueGain": 1.0, "ColourGains": (2.0, 2.0)})


set_exposure(st["preview_exp"])


def free_mb() -> int:
    return shutil.disk_usage(ROOT).free // 2**20


def capture_loop() -> None:
    while True:
        req = cam.capture_request()
        frame = req.make_array("main")
        md = req.get_metadata()
        req.release()
        latest["frame"], latest["md"] = frame, md
        latest["n"] += 1
        r = st["rec"]
        if r is not None and st["mode"] == "recording" and r["i"] < r["n"]:
            r["frames"][r["i"]] = frame
            r["ts"][r["i"]] = md["SensorTimestamp"] / 1e6
            r["i"] += 1
            if r["i"] == r["n"]:
                with lock:
                    st["mode"] = "saving"
                threading.Thread(target=save_and_decode, args=(r,), daemon=True).start()


def preview_loop() -> None:
    """Podgląd ~15 kl/s w 888 px (rozjaśniony, żeby przy krótkiej ekspozycji było cokolwiek widać)."""
    seen = 0
    while True:
        time.sleep(1 / 15)
        if latest["n"] == seen or latest["frame"] is None:
            continue
        seen = latest["n"]
        small = cv2.resize(latest["frame"], (888, 660), interpolation=cv2.INTER_AREA)
        if st["preview_exp"] != "auto":
            small = cv2.LUT(small, GAMMA)
        md = latest["md"]
        txt = f"exp {md.get('ExposureTime', 0)} us  gain {md.get('AnalogueGain', 0):.1f}  |  {status_line()}"
        cv2.putText(small, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        with jpeg_cond:
            jpeg["data"], jpeg["n"] = data, jpeg["n"] + 1
            jpeg_cond.notify_all()


def status_line() -> str:
    r, now = st["rec"], time.monotonic()
    if st["mode"] == "countdown" and r:
        return f"START za {r['t_start'] - now:4.1f} s"
    if st["mode"] == "recording" and r:
        return f"NAGRYWAM {r['i'] / FPS:4.1f} / {r['n'] / FPS:.0f} s"
    return {"saving": "zapisuję...", "idle": "gotowe"}.get(st["mode"], st["mode"])


# ------------------------------------------------------------------ nagranie
def schedule(lead: float, dur: float, name: str) -> dict:
    with lock:
        if st["mode"] != "idle":
            return {"error": f"zajęte: {st['mode']}"}
        if free_mb() < MIN_FREE_MB:
            return {"error": f"za mało miejsca na dysku ({free_mb()} MB)"}
        n = int(dur * FPS)
        num = 1 + max([int(p.name[:2]) for p in ROOT.iterdir() if p.name[:2].isdigit()] or [0])
        safe = "".join(c for c in name if c.isalnum() or c in "-_")[:30] or "nagranie"
        r = {"dir": ROOT / f"{num:02d}_{safe}", "n": n, "i": 0, "lead": lead, "dur": dur,
             "frames": np.empty((n, H, W, 3), np.uint8), "ts": np.empty(n),
             "t_start": time.monotonic() + lead + 3.0}
        st["rec"], st["mode"] = r, "countdown"
    set_exposure(str(args.exposure))           # nagrania zawsze z ręczną, krótką ekspozycją
    threading.Thread(target=countdown, args=(r,), daemon=True).start()
    return {"beep1_in": lead, "start_in": lead + 3.0, "end_in": lead + 3.0 + dur, "dir": r["dir"].name}


def countdown(r: dict) -> None:
    while time.monotonic() < r["t_start"]:
        if st["rec"] is not r:
            return                               # anulowane
        time.sleep(0.002)
    with lock:
        if st["rec"] is r:
            st["mode"] = "recording"


def save_and_decode(r: dict) -> None:
    d = r["dir"]
    d.mkdir(parents=True)
    # JPEG w 3 wątkach (cv2 zwalnia GIL), potem ffmpeg tylko przepakowuje do AVI bez przekodowania
    with ThreadPoolExecutor(3) as ex:
        jpgs = list(ex.map(lambda f: cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(),
                           r["frames"]))
    with open(d / "frames.mjpg", "wb") as f:
        for j in jpgs:
            f.write(j)
    del jpgs
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "mjpeg", "-framerate", str(FPS),
                    "-i", str(d / "frames.mjpg"), "-c", "copy", str(d / "raw.avi")], check=False)
    if (d / "raw.avi").exists():
        (d / "frames.mjpg").unlink()
    ts = r["ts"] - r["ts"][0]
    np.save(d / "ts.npy", ts)
    dt = np.diff(ts)
    info = {"exposure_us": args.exposure, "size": [W, H], "fps_set": FPS, "frames": r["n"],
            "fps_measured": round(1000 / dt.mean(), 2), "dt_max_ms": round(float(dt.max()), 2),
            "gaps": int((dt > 1.5 * 1000 / FPS).sum()), "arduino": ard_cfg}
    (d / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    with lock:
        st["rec"], st["mode"] = None, "idle"   # można nagrywać następne, dekoder liczy w tle
    set_exposure(st["preview_exp"])
    del r["frames"]
    subprocess.Popen(["nice", "-n", "5", "python3", str(HERE / "decode_rec.py"), str(d)],
                     stdout=open(d / "decode.log", "w"), stderr=subprocess.STDOUT)


def rec_dir(name: str):
    """Katalog nagrania po nazwie ze strony - tylko bezpośrednio w ROOT (bez ../)."""
    d = ROOT / Path(name).name
    return d if d.is_dir() and d.parent == ROOT else None


def overlay_mp4(d: Path) -> Path:
    """Film z nakładką jako H.264 (odtworzy go każdy odtwarzacz); robiony przy pierwszym pobraniu."""
    out = d / "wynik.mp4"
    if not out.exists() and (d / "overlay.mjpg").exists():
        tmp = d / "wynik.tmp.mp4"                # pod docelową nazwą dopiero gotowy plik
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "mjpeg", "-framerate", str(FPS // 2),
                        "-i", str(d / "overlay.mjpg"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # H.264: parzyste wymiary
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(tmp)],
                       check=False)
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.rename(out)
    return out


def recordings() -> list:
    out = []
    for d in sorted(ROOT.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        e = {"name": d.name}
        for f in ("info", "result", "progress"):
            p = d / f"{f}.json"
            if p.exists():
                try:
                    e[f] = json.loads(p.read_text())
                except json.JSONDecodeError:
                    pass
        out.append(e)
    return out


# ------------------------------------------------------------------ strona
PAGE = (HERE / "station.html").read_text()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg(self, frames):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            for jpg in frames:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/record":
            res = schedule(float(q.get("lead", 15)), min(20.0, float(q.get("dur", 15))), q.get("name", ""))
            self._json(res, 400 if "error" in res else 200)
        elif u.path == "/cancel":
            with lock:
                if st["mode"] == "countdown":
                    st["rec"], st["mode"] = None, "idle"
            set_exposure(st["preview_exp"])
            self._json({"ok": True})
        elif u.path == "/delete":
            d = rec_dir(q.get("name", ""))
            busy = st["rec"] is not None and st["rec"]["dir"] == d
            if d is None or busy:
                self._json({"error": "nie ma takiego nagrania albo właśnie się zapisuje"}, 400)
            else:
                shutil.rmtree(d)
                self._json({"ok": True, "free_mb": free_mb()})
        elif u.path == "/exposure":
            st["preview_exp"] = q.get("us", "250")
            if st["mode"] == "idle":
                set_exposure(st["preview_exp"])
            self._json({"ok": True})
        else:
            self._json({"error": "?"}, 404)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/status":
            self._json({"mode": st["mode"], "line": status_line(), "free_mb": free_mb(), "arduino": ard_cfg,
                        "exposure_rec": args.exposure, "preview_exp": st["preview_exp"], "recordings": recordings()})
        elif u.path == "/stream":
            def live():
                n = -1
                while True:
                    with jpeg_cond:
                        jpeg_cond.wait_for(lambda: jpeg["n"] != n)
                        data, n = jpeg["data"], jpeg["n"]
                    yield data
            self._mjpeg(live())
        elif u.path.startswith("/download/"):
            parts = u.path[len("/download/"):].split("/")
            d = rec_dir(parts[0]) if len(parts) == 2 else None
            if d is None:
                self._json({"error": "nie ma takiego nagrania"}, 404)
                return
            kind = parts[1]
            if kind == "paczka.zip":
                # wszystko do analizy na Macu; AVI jest już skompresowane, więc zip bez kompresji
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{d.name}.zip"')
                self.end_headers()
                with zipfile.ZipFile(self.wfile, "w", zipfile.ZIP_STORED) as z:
                    for f in ("raw.avi", "ts.npy", "info.json", "result.json"):
                        if (d / f).exists():
                            z.write(d / f, f"{d.name}/{f}")
                return
            f = overlay_mp4(d) if kind == "wynik.mp4" else d / "raw.avi" if kind == "raw.avi" else None
            if f is None or not f.exists():
                self._json({"error": "brak pliku (dekoder jeszcze liczy?)"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4" if f.suffix == ".mp4" else "video/x-msvideo")
            self.send_header("Content-Length", str(f.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{d.name}_{kind}"')
            self.end_headers()
            with open(f, "rb") as fh:
                shutil.copyfileobj(fh, self.wfile)
        elif u.path.startswith("/replay/"):
            d = ROOT / Path(u.path[len("/replay/"):]).name
            if not (d / "overlay_idx.npy").exists():
                self._json({"error": "jeszcze nie gotowe"}, 404)
                return
            idx = np.load(d / "overlay_idx.npy")
            blob = (d / "overlay.mjpg").read_bytes()

            def replay():
                start = time.monotonic()
                for k, (off, t) in enumerate(idx):
                    end = int(idx[k + 1][0]) if k + 1 < len(idx) else len(blob)
                    wait = start + (t - idx[0][1]) / 1000 - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)            # odtwarzanie w czasie rzeczywistym
                    yield blob[int(off):end]
            self._mjpeg(replay())
        else:
            self._json({"error": "?"}, 404)


threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=preview_loop, daemon=True).start()
print(f"Stanowisko: http://169.254.12.2:{args.port}  (nadajnik: {ard_cfg})", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
