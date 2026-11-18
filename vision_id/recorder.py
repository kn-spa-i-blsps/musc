"""
Zapis sesji do późniejszej analizy: surowe klatki, czasy klatek, log dekodera.

    recordings/<data_godzina>/
        raw.avi      surowe klatki z kamery (MJPG)
        ts.npy       czas każdej klatki [ms] - ten sam, który dostał detektor
        log.jsonl    jedna linia na klatkę: tory, ich jasność, dopasowanie, ID
        info.json    ustawienia (Arduino, kamera, konfiguracja) + podsumowanie

Odtworzenie offline dokładnie tak, jak było na żywo:
    uv run receiver.py --video recordings/X/raw.avi --ts recordings/X/ts.npy --debug
"""

import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class SessionRecorder:
    def __init__(self, info: dict, root: Optional[Path] = None):
        root = root or Path(__file__).parent / "recordings"
        self.dir = root / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.info = dict(info)
        self._ts = []
        self._log = open(self.dir / "log.jsonl", "w")
        self._writer: Optional[cv2.VideoWriter] = None
        # zapis klatek w osobnym wątku - kodowanie JPEG nie spowalnia dekodowania
        self._q: "queue.Queue" = queue.Queue(maxsize=120)
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()
        self.frames_lost = 0
        self._save_info()

    def _write_loop(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                return
            if self._writer is None:
                h, w = frame.shape[:2]
                self._writer = cv2.VideoWriter(str(self.dir / "raw.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                               30, (w, h))
            self._writer.write(frame)

    def frame(self, frame: np.ndarray, ts: float, rx, drones) -> None:
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.frames_lost += 1
            return
        self._ts.append(ts)
        tracks = []
        for t in rx.tracks:
            d = t.decoder
            tracks.append({
                "tid": t.tid, "x": round(t.x, 1), "y": round(t.y, 1), "r": round(t.radius, 1),
                "lvl": round(t.levels[-1], 1) if t.levels else None, "c": round(t.contrast, 1),
                "id": t.drone_id, "conf": t.id_confirmed, "g": t.gid,
                "sc": round(d.last_score, 3), "mg": round(d.last_margin, 3), "cov": round(d.coverage, 2),
                "dec": d.last_id, "dec_conf": d.confirmed_id,
            })
        line = {"i": len(self._ts) - 1, "ts": round(ts, 1), "spots": len(rx.last_events),
                "motion": [round(rx.last_motion[0], 2), round(rx.last_motion[1], 2)],
                "tracks": tracks,
                "drones": [[d.drone_id, d.confirmed, round(d.x, 1), round(d.y, 1)] for d in drones]}
        self._log.write(json.dumps(line) + "\n")

    def _save_info(self) -> None:
        (self.dir / "info.json").write_text(json.dumps(self.info, indent=2, default=str))

    def close(self, summary: dict) -> Path:
        self._q.put(None)
        self._thread.join(timeout=10)
        if self._writer is not None:
            self._writer.release()
        self._log.close()
        np.save(self.dir / "ts.npy", np.array(self._ts))
        self.info["summary"] = dict(summary, frames=len(self._ts), frames_not_saved=self.frames_lost)
        self._save_info()
        return self.dir


def config_dict(cfg) -> dict:
    return asdict(cfg)
