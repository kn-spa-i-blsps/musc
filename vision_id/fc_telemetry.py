"""
Telemetria z kontrolera lotu (Mamba F722 MK4 - Betaflight / INAV) po MSP:
żyroskop, akcelerometr i orientacja. Odbiornik używa żyroskopu do kompensacji
ruchu WŁASNEJ kamery: wie, o ile kamera obróciła się między klatkami, więc tory
diod przesuwa razem z obrazem, zamiast gubić je przy każdym szarpnięciu drona.

    uv run fc_telemetry.py --port /dev/ttyACM0                 # podgląd: żyroskop, akcelerometr, kąty, Hz
    uv run fc_telemetry.py --port /dev/ttyACM0 --calibrate     # kamera + FC: znaki osi i opóźnienie
    uv run receiver.py --fc /dev/ttyACM0 --fc-tilt 20          # odbiornik z kompensacją z żyroskopu

Podłączenie: USB kontrolera (port /dev/ttyACM*) albo wolny UART z włączonym MSP
(Betaflight: zakładka Ports -> "Configuration/MSP" na tym UART, 115200).
Potrzebny tylko pyserial.

Co z czego:
  * ŻYROSKOP (MSP_RAW_IMU, deg/s) - to on daje kompensację. Obrót kamery o 1 stopień
    przesuwa cały obraz o ~15 px (przy 65 stopni FOV i 960 px), niezależnie od
    odległości do diody. Całkujemy prędkość kątową między czasami dwóch klatek.
  * AKCELEROMETR - sam nie nadaje się do przesuwania torów: położenie z podwójnego
    całkowania odpływa w ułamku sekundy, a przesunięcie kamery o kilka cm zmienia
    obraz diody odległej o kilka metrów o ułamek piksela. FC używa go do liczenia
    orientacji (roll/pitch względem grawitacji) - tę bierzemy gotową z MSP_ATTITUDE
    (FcTelemetry.attitude), np. do przeliczenia kąta namiaru na poziom/pion świata.

NIE SPRAWDZONE NA SPRZĘCIE: znaki osi żyroskopu i opóźnienie względem kamery
zależą od firmware'u i montażu. Przed lotem: --calibrate (kilka sekund ruszania
dronem przed nieruchomą sceną) - wypisze --fc-signs i --fc-latency do wpisania.
"""

import argparse
import math
import struct
import threading
import time
from collections import deque
from typing import Deque, Optional, Sequence, Tuple

import numpy as np

MSP_API_VERSION = 1
MSP_FC_VARIANT = 2
MSP_RAW_IMU = 102          # 9 x int16: akcelerometr xyz, żyroskop xyz [deg/s], magnetometr xyz
MSP_ATTITUDE = 108         # int16 roll [0.1 deg], pitch [0.1 deg], yaw [deg]


def msp_frame(cmd: int, payload: bytes = b"") -> bytes:
    """Ramka zapytania MSP v1: $M< rozmiar komenda dane suma (xor rozmiaru, komendy i danych)."""
    body = bytes([len(payload), cmd]) + payload
    chk = 0
    for b in body:
        chk ^= b
    return b"$M<" + body + bytes([chk])


def msp_parse(buf: bytes) -> Optional[Tuple[int, bytes, int]]:
    """
    Szuka w buforze pierwszej pełnej odpowiedzi MSP v1. Zwraca (komenda, dane,
    ile bajtów bufora zużyto) albo None, gdy ramka jeszcze niepełna. Odpowiedź
    błędu ($M!) i zła suma kontrolna dają dane = b"" z komendą -1.
    """
    i = buf.find(b"$M")
    if i < 0 or len(buf) < i + 6:
        return None
    size, cmd = buf[i + 3], buf[i + 4]
    end = i + 5 + size + 1
    if len(buf) < end:
        return None
    payload = buf[i + 5:i + 5 + size]
    chk = size ^ cmd
    for b in payload:
        chk ^= b
    if buf[i + 2:i + 3] != b">" or chk != buf[end - 1]:
        return -1, b"", end
    return cmd, payload, end


def body_to_cam(rpy: Sequence[float], tilt_deg: float = 0.0, signs: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Wektor (roll, pitch, yaw) z FC -> osie kamery (x w prawo, y w dół, z do przodu).
    Założenie przy znakach (1, 1, 1): roll + = prawa strona w dół, pitch + = nos w górę,
    yaw + = nos w prawo. Kamera patrzy do przodu, podniesiona o tilt_deg w górę.
    """
    p, q, r = (float(v) * float(s) for v, s in zip(rpy, signs))
    c, s = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    return np.array([q, r * c + p * s, p * c - r * s])


class GyroIntegrator:
    """Próbki prędkości kątowej z czasem -> kąt obrotu między dwiema chwilami (wzór trapezów)."""

    def __init__(self, maxlen: int = 4000):
        self._buf: Deque[Tuple[float, float, float, float]] = deque(maxlen=maxlen)   # t_ms, wx, wy, wz [rad/s]
        self._lock = threading.Lock()

    def add(self, t_ms: float, rate_rad_s: Sequence[float]) -> None:
        with self._lock:
            self._buf.append((float(t_ms), float(rate_rad_s[0]), float(rate_rad_s[1]), float(rate_rad_s[2])))

    def rotation(self, t0_ms: float, t1_ms: float, max_gap_ms: float = 60.0) -> Optional[np.ndarray]:
        """
        Kąt obrotu [rad, 3 osie] w przedziale [t0, t1]. None, gdy próbki nie
        pokrywają przedziału (przerwa w telemetrii) - wtedy lepiej nie zgadywać.
        """
        if t1_ms <= t0_ms:
            return None
        with self._lock:
            a = np.array(self._buf)
        if len(a) < 2 or a[0, 0] > t0_ms + max_gap_ms or a[-1, 0] < t1_ms - max_gap_ms:
            return None
        lo = max(int(np.searchsorted(a[:, 0], t0_ms)) - 1, 0)
        hi = min(int(np.searchsorted(a[:, 0], t1_ms)) + 1, len(a))
        a = a[lo:hi]
        if len(a) < 2 or float(np.diff(a[:, 0]).max()) > max_gap_ms:
            return None
        grid = np.unique(np.clip(np.concatenate([[t0_ms, t1_ms], a[:, 0]]), t0_ms, t1_ms))
        out = np.zeros(3)
        for k in range(3):
            w = np.interp(grid, a[:, 0], a[:, k + 1])          # poza próbkami: ostatnia znana wartość
            out[k] = float(((w[1:] + w[:-1]) * 0.5 * np.diff(grid)).sum()) / 1000.0
        return out


class FcTelemetry:
    """
    Wątek, który w kółko pyta FC o żyroskop (i co kilka zapytań o orientację).

    t0 = początek zegara w sekundach time.perf_counter(); podaj CameraThread.t0,
    żeby czasy próbek były na tej samej osi co czasy klatek.
    latency_ms = o ile obraz z kamery jest spóźniony względem żyroskopu
    (naświetlanie + przesył klatki); wychodzi z --calibrate.
    """

    def __init__(self, port: str, baud: int = 115200, t0: Optional[float] = None, tilt_deg: float = 0.0,
                 signs: Sequence[float] = (1.0, 1.0, 1.0), latency_ms: float = 0.0, attitude_every: int = 10):
        import serial                                 # tylko tutaj - reszta modułu działa bez portu
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.t0 = time.perf_counter() if t0 is None else t0
        self.tilt_deg, self.signs, self.latency_ms = tilt_deg, tuple(signs), latency_ms
        self.gyro = GyroIntegrator()                  # prędkości w osiach FC (roll, pitch, yaw) [rad/s]
        self.acc_raw: Tuple[int, int, int] = (0, 0, 0)        # surowy akcelerometr (skala zależy od firmware'u)
        self.attitude: Optional[Tuple[float, float, float]] = None   # roll, pitch, yaw [deg] z FC
        self.variant = ""
        self.samples = 0
        self.errors = 0
        self._attitude_every = attitude_every
        self._buf = b""
        self._running = True
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        got = self._request(MSP_FC_VARIANT)
        if got is None:
            self.ser.close()
            raise RuntimeError(f"FC na {port} nie odpowiada na MSP (zły port? MSP wyłączone na tym UART? "
                               "otwarty Betaflight Configurator?)")
        self.variant = got.decode("ascii", "replace")
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def _request(self, cmd: int, timeout: float = 0.1) -> Optional[bytes]:
        self.ser.write(msp_frame(cmd))
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            chunk = self.ser.read(max(self.ser.in_waiting, 1))
            if chunk:
                self._buf += chunk
            while True:
                got = msp_parse(self._buf)
                if got is None:
                    if len(self._buf) > 512:
                        self._buf = self._buf[-64:]
                    break
                rcmd, payload, used = got
                self._buf = self._buf[used:]
                if rcmd == cmd:
                    return payload
        return None

    def _run(self) -> None:
        k = 0
        while self._running:
            t_send = self._now_ms()
            try:
                raw = self._request(MSP_RAW_IMU)
            except Exception:
                raw = None
            if raw is None or len(raw) < 12:
                self.errors += 1
                continue
            v = struct.unpack("<6h", raw[:12])
            self.acc_raw = v[:3]
            # czas próbki = środek między zapytaniem a odpowiedzią
            self.gyro.add(0.5 * (t_send + self._now_ms()), [math.radians(x) for x in v[3:6]])
            self.samples += 1
            k += 1
            if k % self._attitude_every == 0:
                try:
                    att = self._request(MSP_ATTITUDE)
                except Exception:
                    att = None
                if att is not None and len(att) >= 6:
                    r, p, y = struct.unpack("<3h", att[:6])
                    self.attitude = (r / 10.0, p / 10.0, float(y))

    @property
    def rate_hz(self) -> float:
        return self.samples / max(time.perf_counter() - self._started, 1e-3)

    def rotation_body(self, t0_ms: float, t1_ms: float) -> Optional[np.ndarray]:
        """Obrót FC między czasami dwóch klatek [rad] w jego osiach (roll, pitch, yaw), bez znaków i montażu."""
        return self.gyro.rotation(t0_ms - self.latency_ms, t1_ms - self.latency_ms)

    def rotation(self, t0_ms: float, t1_ms: float) -> Optional[np.ndarray]:
        """Obrót KAMERY między czasami dwóch klatek - do LedIdReceiver.process(cam_rot=...). None = brak danych."""
        rot = self.rotation_body(t0_ms, t1_ms)
        return None if rot is None else body_to_cam(rot, self.tilt_deg, self.signs)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.ser.close()


def fit_calibration(frame_ts: np.ndarray, shifts: np.ndarray, gyro: GyroIntegrator, fx: float, tilt_deg: float,
                    lags: Sequence[float] = tuple(range(-120, 121, 10))) -> Optional[dict]:
    """
    Dopasowanie żyroskopu do ruchu obrazu. shifts[i] = przesunięcie obrazu (dx, dy) [px]
    między klatką i-1 a i (z korelacji fazowej na nieruchomej scenie). Szuka opóźnienia
    i znaków osi pitch / yaw, przy których żyroskop najlepiej przewiduje to przesunięcie.
    (Znak roll nie przesuwa środka obrazu - z tego pomiaru go nie widać.)
    """
    best = None
    for lag in lags:
        rows = []
        for i in range(1, len(frame_ts)):
            rot = gyro.rotation(frame_ts[i - 1] - lag, frame_ts[i] - lag)
            if rot is not None and np.all(np.isfinite(shifts[i])):
                rows.append((rot, shifts[i]))
        if len(rows) < 30:
            continue
        rots = np.array([r for r, _ in rows])
        meas = np.array([m for _, m in rows])
        for sp in (1.0, -1.0):
            for sy in (1.0, -1.0):
                cam = np.array([body_to_cam(r, tilt_deg, (1.0, sp, sy)) for r in rots])
                pred = np.stack([-fx * cam[:, 1], fx * cam[:, 0]], axis=1)     # obrót w prawo -> obraz w lewo
                num = float((pred * meas).sum())
                den = math.sqrt(float((pred ** 2).sum()) * float((meas ** 2).sum()))
                corr = num / den if den > 0 else 0.0
                scale = num / float((pred ** 2).sum()) if float((pred ** 2).sum()) > 0 else 0.0
                if best is None or corr > best["corr"]:
                    best = {"corr": corr, "latency_ms": float(lag), "signs": (1.0, sp, sy), "scale": scale, "n": len(rows)}
    return best


def _calibrate(fc: FcTelemetry, cam, seconds: float, hfov_deg: float) -> None:
    import cv2
    print(f"Kalibracja {seconds:.0f} s: obracaj dronem (góra-dół, lewo-prawo) przed NIERUCHOMĄ sceną z wyraźnymi "
          "szczegółami. Nie machaj niczym przed kamerą.")
    ts, shifts, prev, hann, width = [], [], None, None, 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        item = cam.read()
        if item is None:
            break
        frame, t = item
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = gray.shape[1]
        g = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA).astype(np.float32)
        if hann is None:
            hann = cv2.createHanningWindow((320, 240), cv2.CV_32F)
        shift = (np.nan, np.nan)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g, hann)
            if resp > 0.2:
                shift = (dx * gray.shape[1] / 320.0, dy * gray.shape[0] / 240.0)
        prev = g
        ts.append(t)
        shifts.append(shift)
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    res = fit_calibration(np.array(ts), np.array(shifts), fc.gyro, fx, fc.tilt_deg)
    if res is None:
        print("Za mało dobrych klatek (scena bez szczegółów albo brak telemetrii).")
        return
    sg = "".join("+" if s > 0 else "-" for s in res["signs"])
    print(f"Zgodność żyroskopu z obrazem: {res['corr']:.2f} (1.0 = idealna; poniżej ~0.7 nie ufaj wynikowi), "
          f"{res['n']} klatek")
    print(f"  --fc-signs={sg} --fc-latency {res['latency_ms']:.0f}")
    print(f"  skala ruchu obraz/żyroskop: {res['scale']:.2f} (daleko od 1.0 -> popraw --hfov: "
          f"spróbuj {2 * math.degrees(math.atan(math.tan(math.radians(hfov_deg) / 2) / max(res['scale'], 1e-3))):.0f})")


def parse_signs(text: str) -> Tuple[float, float, float]:
    if len(text) != 3 or set(text) - set("+-"):
        raise argparse.ArgumentTypeError("znaki osi roll/pitch/yaw jako 3 znaki, np. +-+")
    return tuple(1.0 if c == "+" else -1.0 for c in text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="port FC, np. /dev/ttyACM0 (USB) albo /dev/ttyAMA0 (UART)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--calibrate", action="store_true", help="kamera + FC: znajdź znaki osi i opóźnienie")
    ap.add_argument("--camera", type=int, help="indeks kamery do kalibracji")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--tilt", type=float, default=0.0, help="podniesienie kamery względem drona [deg]")
    ap.add_argument("--hfov", type=float, default=65.0, help="poziome pole widzenia kamery [deg]")
    args = ap.parse_args()

    if args.calibrate:
        from camera import CameraThread, open_camera
        cam = CameraThread(open_camera(args.camera))
        fc = FcTelemetry(args.port, args.baud, t0=cam.t0, tilt_deg=args.tilt)
        print(f"FC: {fc.variant}")
        try:
            _calibrate(fc, cam, args.seconds, args.hfov)
        finally:
            fc.close()
            cam.release()
        return

    fc = FcTelemetry(args.port, args.baud, tilt_deg=args.tilt)
    print(f"FC: {fc.variant}. Ctrl+C kończy.")
    try:
        last = fc._now_ms()
        while True:
            time.sleep(0.25)
            now = fc._now_ms()
            rot = fc.rotation_body(last, now)
            last = now
            rate = "brak danych" if rot is None else "  ".join(f"{math.degrees(v) / 0.25:+7.1f}" for v in rot)
            print(f"{fc.rate_hz:5.0f} Hz | żyro roll/pitch/yaw [deg/s]: {rate} | akc. surowy {fc.acc_raw} | "
                  f"kąty roll/pitch/yaw [deg]: {fc.attitude} | błędy {fc.errors}")
    except KeyboardInterrupt:
        pass
    finally:
        fc.close()


if __name__ == "__main__":
    main()
