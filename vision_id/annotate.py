"""
Ręczne oznaczanie, gdzie naprawdę jest dioda, na losowych klatkach nagrania.
Z tych etykiet bench.py liczy, czy kółko z ID siedzi NA diodzie (a nie na
odblasku, ekranie laptopa itp.).

    uv run annotate.py ~/Downloads/IMG_5655.MOV            # 150 losowych klatek
    uv run annotate.py recordings/20261017_165150 --n 80   # sesja z verify_live

Klatki są losowane równomiernie po całym nagraniu (po jednej z każdego
odcinka). Postęp zapisuje się po każdej klatce - można przerwać (q) i wrócić
tym samym poleceniem, zacznie od pierwszej nieoznaczonej.

Mysz / klawisze:
    lewy klik    środek świecącej diody (źródła, nie odblasku); kilka diod = kilka klików
    prawy klik/u cofnij ostatni klik
    spacja       zatwierdź i dalej
    x            diody nie ma w kadrze / całkiem zasłonięta (też ważne - liczy fałszywe ID)
    s            nie wiem / pomiń
    , .          sąsiednia klatka -/+ (gdy dioda akurat zgaszona); etykieta trafia do tej klatki
    b            wróć do poprzedniej próbki (poprawka)
    q / Esc      zapisz i wyjdź

Lupa w rogu powiększa okolicę kursora 5x - klikaj w najjaśniejszy środek diody.

Etykiety: plik.mov -> labels/<nazwa>.json, sesja recordings/X -> recordings/X/labels.json.
Numery klatek = kolejność czytania przez cv2.VideoCapture, tak samo jak w bench.py.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

NEIGHBORS = 2                 # ile sąsiednich klatek w każdą stronę da się podejrzeć
VIEW_W = 1280                 # szerokość okna
ZOOM_SRC, ZOOM = 41, 5        # lupa: wycinek 41x41 px oryginału, powiększony 5x


def video_and_labels_path(arg: str):
    p = Path(arg).expanduser()
    if p.is_dir():
        return p / "raw.avi", p / "labels.json"
    return p, Path(__file__).parent / "labels" / f"{p.stem}.json"


def pick_samples(n_frames: int, n: int, seed: int):
    """Po jednej losowej klatce z każdego z n równych odcinków nagrania."""
    rnd = random.Random(seed)
    edges = np.linspace(0, n_frames, n + 1).astype(int)
    return [rnd.randrange(a, max(a + 1, b)) for a, b in zip(edges[:-1], edges[1:])]


def preload(video: Path, wanted: set):
    """Jedno przejście po nagraniu; trzyma w pamięci (JPEG) tylko potrzebne klatki."""
    cap = cv2.VideoCapture(str(video))
    frames, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        i += 1
        if i % 300 == 0:
            print(f"\r  wczytuję klatki... {i}", end="", flush=True)
    print(f"\r  wczytano {i} klatek nagrania")
    return frames


class Annotator:
    def __init__(self, video: Path, out: Path, n: int, seed: int):
        self.out = out
        cap = cv2.VideoCapture(str(video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if out.exists():
            self.data = json.loads(out.read_text())
            print(f"Wznawiam {out} ({len(self.data['labels'])} etykiet)")
        else:
            self.data = {"video": str(video), "frames": n_frames, "fps": fps,
                         "samples": pick_samples(n_frames, n, seed), "labels": []}
        self.samples = self.data["samples"]
        wanted = {min(max(s + d, 0), n_frames - 1) for s in self.samples for d in range(-NEIGHBORS, NEIGHBORS + 1)}
        self.frames = preload(video, wanted)
        self.n_frames = max(self.frames) + 1
        self.mouse = None
        self.pts = []

    def label_for(self, sample: int):
        return next((l for l in self.data["labels"] if l["sample"] == sample), None)

    def set_label(self, sample: int, frame: int, state: str, pts) -> None:
        self.data["labels"] = [l for l in self.data["labels"] if l["sample"] != sample]
        self.data["labels"].append({"sample": sample, "frame": frame, "state": state,
                                    "pts": [[round(x, 1), round(y, 1)] for x, y in pts]})
        self.data["labels"].sort(key=lambda l: l["sample"])
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.data, indent=1))

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.mouse = (x / self.scale, y / self.scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append(self.mouse)
        elif event == cv2.EVENT_RBUTTONDOWN and self.pts:
            self.pts.pop()

    def render(self, img: np.ndarray, k: int, frame: int) -> np.ndarray:
        h, w = img.shape[:2]
        view = cv2.resize(img, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_AREA)
        for x, y in self.pts:
            c = (int(x * self.scale), int(y * self.scale))
            cv2.circle(view, c, 12, (0, 255, 0), 1)
            cv2.drawMarker(view, c, (0, 255, 0), cv2.MARKER_CROSS, 6, 1)
        if self.mouse is not None:                       # lupa
            mx, my = int(self.mouse[0]), int(self.mouse[1])
            r = ZOOM_SRC // 2
            pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT)
            crop = pad[my:my + ZOOM_SRC, mx:mx + ZOOM_SRC]
            if crop.shape[:2] == (ZOOM_SRC, ZOOM_SRC):
                big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
                for x, y in self.pts:
                    c = (int((x - mx + r + 0.5) * ZOOM), int((y - my + r + 0.5) * ZOOM))
                    cv2.drawMarker(big, c, (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
                cv2.drawMarker(big, (r * ZOOM + ZOOM // 2,) * 2, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                s = big.shape[0]
                view[8:8 + s, view.shape[1] - s - 8:view.shape[1] - 8] = big
                cv2.rectangle(view, (view.shape[1] - s - 9, 7), (view.shape[1] - 7, 9 + s), (255, 255, 255), 1)
        sample = self.samples[k]
        done = sum(self.label_for(s) is not None for s in self.samples)
        prev = self.label_for(sample)
        off = frame - sample
        lines = [f"probka {k + 1}/{len(self.samples)}  (oznaczone {done})   klatka {frame}"
                 f"  t={frame / self.data['fps']:.2f}s" + (f"  [sasiednia {off:+d}]" if off else ""),
                 "klik = dioda   spacja = OK   x = brak diody   s = pomin   , . = sasiednia   b = wstecz   q = koniec"]
        if prev is not None:
            lines.append(f"poprzednio: {prev['state']} (klatka {prev['frame']})")
        band = view[:14 + 22 * len(lines), :800]
        band[:] = (band * 0.35).astype(np.uint8)          # przyciemniony pasek pod tekstem
        for i, t in enumerate(lines):
            cv2.putText(view, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    def run(self) -> None:
        win = "annotate - kliknij diode"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        k = next((i for i, s in enumerate(self.samples) if self.label_for(s) is None), len(self.samples))
        off = 0
        self.pts = []
        while k < len(self.samples):
            sample = self.samples[k]
            frame = min(max(sample + off, 0), self.n_frames - 1)
            if frame not in self.frames:                 # licznik klatek w pliku bywa zawyżony
                if sample not in self.frames:
                    self.set_label(sample, sample, "skip", [])
                    k, off = k + 1, 0
                    continue
                frame, off = sample, 0
            img = cv2.imdecode(self.frames[frame], cv2.IMREAD_COLOR)
            self.scale = min(1.0, VIEW_W / img.shape[1])
            cv2.imshow(win, self.render(img, k, frame))
            key = cv2.waitKey(20) & 0xFF
            nxt = None
            if key in (ord("q"), 27):
                break
            if key == ord("u") and self.pts:
                self.pts.pop()
            elif key == ord(","):
                off = max(off - 1, -NEIGHBORS)
            elif key == ord("."):
                off = min(off + 1, NEIGHBORS)
            elif key in (ord(" "), 13) and self.pts:
                self.set_label(sample, frame, "led", self.pts)
                nxt = k + 1
            elif key == ord("x"):
                self.set_label(sample, frame, "absent", [])
                nxt = k + 1
            elif key == ord("s"):
                self.set_label(sample, frame, "skip", [])
                nxt = k + 1
            elif key == ord("b"):
                nxt = max(k - 1, 0)
            if nxt is not None:
                k, off = nxt, 0
                prev = self.label_for(self.samples[k]) if k < len(self.samples) else None
                self.pts = [tuple(p) for p in prev["pts"]] if prev else []
                if prev:
                    off = prev["frame"] - prev["sample"]
        cv2.destroyAllWindows()
        states = [l["state"] for l in self.data["labels"]]
        print(f"Zapisano {self.out}: dioda {states.count('led')}, brak {states.count('absent')}, "
              f"pominiete {states.count('skip')}, zostalo {len(self.samples) - len(states)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="plik wideo albo sesja recordings/X")
    ap.add_argument("--n", type=int, default=150, help="ile losowych klatek (tylko przy pierwszym uruchomieniu)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="plik etykiet (domyślnie labels/<nazwa>.json)")
    args = ap.parse_args()
    video, out = video_and_labels_path(args.video)
    Annotator(video, Path(args.out) if args.out else out, args.n, args.seed).run()


if __name__ == "__main__":
    main()
