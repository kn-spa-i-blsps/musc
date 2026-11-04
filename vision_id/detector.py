"""
Odbiornik: znajduje mrugające diody, śledzi je, dekoduje ID i łączy diody
o tym samym ID w drona (z kątem namiaru z kamery). Zero ML - OpenCV + numpy.

Potok na klatkę:
  1. pomniejszenie do ~960 px szerokości
  2. kompensacja ruchu kamery (korelacja fazowa) - przesuwamy historię i tory
  3. ENERGIA MRUGANIA: |zmiana jasności piksela| z pamięcią ~1 s. Ściana,
     plakaty, twarz - zero; mrugająca dioda - najsilniejsze miejsce w obrazie
     (jej odblaski na płytce/dłoni też mrugają, ale słabiej).
     Kandydaci na diodę = lokalne szczyty energii.
  4. śledzenie toru:
     a) RUCH z przepływu optycznego (Lucas-Kanade) na teksturze wokół diody
        (płytka, elektronika, palce) - tor jedzie razem z obiektem także wtedy,
        gdy dioda jest zgaszona, a jej mruganie nie zaburza pomiaru ruchu
     b) KOREKTA: łagodne przyciągnięcie do najsilniejszego szczytu energii tuż
        obok (kilka px) - tor zostaje na samej diodzie, a nieruchoma dioda nie
        "odpływa" na odblaski
  5. nowy tor na silnym szczycie energii, przy którym nie ma jeszcze toru
  6. sygnał toru = średnia jasność w oknie wokół diody (surowa - światło tylko
     dodaje jasności, więc zapalona jest zawsze jaśniejsza)
  7. TemplateDecoder (protocol.py): dopasowanie do wszystkich kodów i faz ->
     najpierw ID wstępne ("DRON 5?"), po 2 zgodnych dopasowaniach potwierdzone;
     ID wygasa, gdy dekoder przestaje je potwierdzać
  8. tory z tym samym ID = jeden dron; pozycja z najsilniejszych źródeł
     (sama dioda, nie jej odblaski) -> azymut/elewacja

Tor znika, gdy od ~1 s nie mruga albo nie ma przy nim energii. Gdy w tym samym
miejscu zaraz powstanie nowy tor, przejmuje ID wstępnie i musi je potwierdzić.

Na dronie z detektorem obiektów (YOLO itp.) kandydatów można dodatkowo
ograniczyć do ramek dronów - reszta potoku zostaje bez zmian.

Współrzędne torów i dronów są w układzie obrazu POMNIEJSZONEGO
(LedIdReceiver.small to ten obraz - na nim rysuje podgląd).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from protocol import DEFAULT_CHIP_HZ, FRAME_CHIPS, TemplateDecoder

Peak = Tuple[float, float, float, float]          # x, y, promień, energia


@dataclass
class ReceiverConfig:
    chip_hz: float = DEFAULT_CHIP_HZ
    proc_width: int = 960            # szerokość obrazu, na którym liczymy
    led_color: Optional[str] = None  # "red"/"green"/"blue"/"yellow"/"nonblue" -> mapa koloru zamiast jasności
    energy_threshold: float = 15.0   # min energia mrugania szczytu-kandydata
    spawn_threshold: float = 25.0    # min energia, żeby założyć NOWY tor
    energy_decay: float = 0.95       # pamięć energii (0.95 na klatkę ~ 1.3 s przy 30 fps)
    energy_filter: str = "blur"      # "blur" albo "max" (filtr maksimum; w testach gorszy)
    feature: str = "max"             # jasność: "max" (max z B,G,R) albo "luma" (w testach gorsza)
    peak_nms_px: int = 11            # szczyty energii nie bliżej niż to
    snap_px: float = 8.0             # korekta tylko do szczytu bliżej niż to (+ promień diody)
    snap_gain: float = 0.4           # jak mocno przyciągać do szczytu (0..1)
    min_track_sep_px: float = 15.0   # nowy tor nie bliżej niż to od istniejącego
    flow_ring: Tuple[int, int] = (8, 45)   # punkty przepływu: pierścień wokół diody [px]
    flow_points: int = 30
    lost_ms: float = 1000.0          # tor bez energii przez tyle -> zgubiony
    search_grow: float = 0.08        # zgubiony tor szuka szerzej: +px na każdą ms bez diody
    source_search_px: int = 35       # w tym promieniu tor szuka najjaśniejszego mrugającego punktu
    sync_corr: float = 0.85          # tory mrugające tak zgodnie = to samo źródło światła
    seek_source: bool = True         # przesuwaj tor na najjaśniejszy mrugający punkt w okolicy
    source_margin: float = 40.0      # o ile jaśniejszy musi być punkt obok, żeby tor tam przeszedł
    sync_adopt: bool = True          # tor zsynchronizowany z torem z ID przejmuje to ID
    sample_pad: int = 3              # okno sygnału: promień diody + tyle px
    sample_max: int = 12             # ...ale najwyżej tyle
    min_contrast: float = 12.0       # różnica jasności on/off w oknie; poniżej - tor "nie mruga"
    max_tracks: int = 24
    motion_comp: bool = True         # kompensacja ruchu kamery
    hfov_deg: float = 65.0           # poziome pole widzenia kamery (do kąta)
    group_radius_px: float = 150.0   # diody jednego drona są bliżej niż to
    core_lock: bool = True           # gdy dioda świeci: tor na środek jej białego, prześwietlonego rdzenia
    core_px: int = 16                # promień szukania rdzenia wokół toru [px]
    core_tol: float = 8.0            # rdzeń = piksele najwyżej o tyle ciemniejsze od najjaśniejszego
    anchor: str = "core"             # pozycja drona: "core" (tor z najjaśniejszym rdzeniem) / "mean" (stare)
    envelope: bool = False           # dekoder: próg względem lokalnego poziomu "zapalona" (tylko z ramką drona)
    anchor_own_only: bool = True     # pozycja drona tylko z torów z własnym odczytem (nie z przejętym ID)
    signal: str = "gray"             # sygnał do dekodera: "gray" (luminancja - biały rdzeń i kolor) / "white" / "feat"
    core_drop: float = 15.0          # nie podawaj pozycji, gdy najjaśniejszy tor z ID jest o tyle ciemniejszy
                                     # niż źródło tego ID przed chwilą (zostały same odblaski); 0 = wył.
    # ruch toru między klatkami: "flow" - tylko przepływ optyczny wokół diody;
    # "cv" - dodatkowo stała prędkość z ostatnich położeń diody (jak HT4D w UVDAR:
    # w krótkim oknie dioda jedzie po prostej), więc tor jedzie dalej, gdy dioda
    # gaśnie albo gdy wokół niej nie ma tekstury (krótka ekspozycja, tło nieba)
    motion_model: str = "flow"
    cv_window_ms: float = 300.0      # z ilu ostatnich ms położeń liczymy prędkość
    cv_max_gap_ms: float = 250.0     # dłużej bez diody -> tor przestaje jechać (stoi i szuka)
    cv_max_px_per_ms: float = 2.0    # górna granica prędkości (2 px/ms = 33 px/klatkę przy 60 fps)
    # jak w UVDAR: w każdej klatce dioda to JASNY PUNKT w mapie cechy (np. kolor nonblue),
    # tor przeskakuje na niego w oknie wokół przewidzianego położenia; bez tego tor
    # celuje w energię mrugania, która przy ruchu rozmazuje się w smugę
    point_lock: bool = False
    point_min: float = 20.0          # najsłabszy punkt, który jeszcze uznajemy za zapaloną diodę
    point_rel: float = 0.5           # ...i co najmniej taki ułamek jasności, jaką ta dioda już miała
    flow: bool = True                # przepływ optyczny wokół diody (na Pi ~16 ms/klatkę; z point_lock zbędny?)

    @property
    def chip_ms(self) -> float:
        return 1000.0 / self.chip_hz


# Gotowe zestawy ustawień (receiver.py --preset NAZWA). Pola jak w ReceiverConfig.
PRESETS: Dict[str, Dict[str, object]] = {
    # Pi 5 + IMX477 1332x990 @ 60 fps, ekspozycja ~250 us, kolorowe diody (czerwona/zielona)
    # na tle nieba, nadajnik w ruchu. Nagrania testowe 2026-11-03: nieruchome 100%,
    # ręka w ruchu ~88% klatek z ID, zero fałszywych ID.
    "pi_niebo": dict(led_color="nonblue", proc_width=1332, motion_model="cv", point_lock=True,
                     snap_px=20.0, min_track_sep_px=30.0, core_lock=False, signal="feat", min_contrast=6.0),
}


@dataclass
class LedTrack:
    tid: int
    x: float
    y: float
    created_ts: float
    last_seen_ts: float
    decoder: TemplateDecoder
    radius: float = 3.0
    pts: Optional[np.ndarray] = None             # punkty przepływu (N x 1 x 2, float32)
    pts_frame: int = 0                           # kiedy ostatnio odświeżone
    levels: Deque[float] = field(default_factory=lambda: deque(maxlen=90))
    states: Deque[int] = field(default_factory=lambda: deque(maxlen=90))
    lohi: Optional[Tuple[float, float]] = None   # poziomy "zgaszona"/"zapalona" (percentyle 10/90)
    contrast: float = 0.0
    roi: Tuple[int, int, int, int] = (0, 0, 0, 0)  # okno pomiaru (do podglądu)
    drone_id: Optional[int] = None
    id_confirmed: bool = False
    inherited_ts: Optional[float] = None         # ID przejęte po zgubionym torze (czeka na odczyt)
    on_peak: float = 0.0                         # jasność najjaśniejszego punktu toru, gdy dioda świeci
    synced_with: Optional[int] = None            # ID przejęte przez synchronizację z innym torem
    core: float = 0.0                            # jasność białego rdzenia, gdy świeci (źródło > odblask)
    obs: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=40))  # (ts, x, y), gdy widać diodę
    vx: float = 0.0                              # prędkość z obs [px/ms] (motion_model="cv")
    vy: float = 0.0
    flow_moved: bool = False                     # przepływ optyczny przesunął tor w tej klatce
    point_peak: float = 0.0                      # jasność punktu diody, gdy świeci (point_lock)


@dataclass
class DroneObs:
    drone_id: int
    x: float                         # pozycja diody/konstelacji [px obrazu pomniejszonego]
    y: float
    azimuth_deg: float               # + w prawo od osi optycznej
    elevation_deg: float             # + w górę
    n_leds: int
    spread_px: float                 # rozpiętość konstelacji (do zgrubnej odległości)
    track_ids: List[int]
    confirmed: bool = True           # False = ID wstępne (1 dopasowanie albo przejęte po zgubieniu)


def fresh_ok(dec: TemplateDecoder, ts: float, frame_ms: float) -> bool:
    return ts - dec.last_ok_ts < 1.5 * frame_ms


def brightness(img: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img)
    return cv2.max(cv2.max(b, g), r)


def chroma(img: np.ndarray, color: str) -> np.ndarray:
    """Jak bardzo piksel ma kolor diody: np. dla czerwonej R - max(G, B), obcięte do 0..255."""
    # cv2.subtract na uint8 obcina do 0..255 sam (to samo co int16 + clip, ~10x szybciej na Pi)
    b, g, r = cv2.split(img)
    if color == "red":
        return cv2.subtract(r, cv2.max(g, b))
    if color == "green":
        return cv2.subtract(g, cv2.max(r, b))
    if color == "blue":
        return cv2.subtract(b, cv2.max(r, g))
    if color == "yellow":
        return cv2.subtract(cv2.min(r, g), b)
    if color == "nonblue":
        # każda dioda "cieplejsza" od nieba (czerwona, pomarańczowa, zielona, żółta) naraz;
        # niebo jest niebieskie, a białe chmury i ściany mają R ~ G ~ B, więc dają ~0
        return cv2.subtract(cv2.max(r, g), b)
    raise ValueError(f"nieznany kolor diody: {color}")


def _even_dft(n: int) -> int:
    """
    Najbliższy rozmiar >= n, który jest parzysty i "optymalny" dla FFT.
    cv2.phaseCorrelate dopełnia obraz do optymalnego rozmiaru; jeśli wyjdzie
    nieparzysty, zwraca stałe przesunięcie 0.5 px nawet dla identycznych obrazów.
    """
    while n % 2 or cv2.getOptimalDFTSize(n) != n:
        n += 1
    return n


class LedIdReceiver:
    def __init__(self, cfg: Optional[ReceiverConfig] = None):
        self.cfg = cfg or ReceiverConfig()
        self.tracks: List[LedTrack] = []
        self._next_tid = 1
        self._frame_idx = 0
        self._prev_gray: Optional[np.ndarray] = None       # do przepływu optycznego
        self._prev_small_gray: Optional[np.ndarray] = None  # do kompensacji ruchu
        self._prev_feat: Optional[np.ndarray] = None
        self._hann: Optional[np.ndarray] = None
        self._last_ts: Optional[float] = None
        self._frame_dt = 33.3                    # średni odstęp klatek [ms]
        # tory z potwierdzonym ID, które niedawno zginęły: (czas, x, y, id)
        self._lost: List[Tuple[float, float, float, int]] = []
        self._src_core: Dict[int, Tuple[float, float]] = {}   # ID -> (jasność rdzenia źródła, czas)
        self.energy: Optional[np.ndarray] = None
        self.small: Optional[np.ndarray] = None
        self.last_events: List[Tuple[float, float, float, int]] = []   # szczyty energii (do podglądu)
        self.last_motion: Tuple[float, float] = (0.0, 0.0)
        self.debug = False                       # zapisuj maski do podglądu (dbg_*)
        self.dbg_on = self.dbg_off = self.dbg_lit = None

    # ------------------------------------------------------------ ruch kamery
    def _compensate_motion(self, gray: np.ndarray) -> None:
        h, w = gray.shape
        g = cv2.resize(gray, (_even_dft(w // 4), _even_dft(h // 4)), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._hann is None or self._hann.shape != g.shape:
            self._hann = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        prev, self._prev_small_gray = self._prev_small_gray, g
        self.last_motion = (0.0, 0.0)
        if prev is None or prev.shape != g.shape:
            return
        (dx, dy), response = cv2.phaseCorrelate(prev, g, self._hann)
        dx, dy = dx * w / g.shape[1], dy * h / g.shape[0]
        if response < 0.2 or abs(dx) > w / 10 or abs(dy) > h / 10 or (abs(dx) < 0.5 and abs(dy) < 0.5):
            return
        self.last_motion = (dx, dy)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        if self._prev_feat is not None:
            self._prev_feat = cv2.warpAffine(self._prev_feat, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if self.energy is not None:
            self.energy = cv2.warpAffine(self.energy, m, (w, h))
        # tory przesuwa przepływ optyczny (zawiera już ruch kamery)

    # -------------------------------------------------------- energia mrugania
    def _update_energy(self, feat: np.ndarray) -> None:
        f = feat.astype(np.float32)
        if self._prev_feat is not None:
            # automatyczna ekspozycja zmienia jasność CAŁEGO obrazu - bez wyrównania
            # każda krawędź w kadrze wyglądałaby jak mruganie
            sub = (slice(None, None, 8), slice(None, None, 8))
            cur_s, prev_s = f[sub], self._prev_feat[sub]
            ok = (cur_s > 20) & (cur_s < 235) & (prev_s > 20) & (prev_s < 235)
            gain = float(np.median(cur_s[ok] / prev_s[ok])) if ok.sum() > 100 else 1.0
            prev = self._prev_feat if gain == 1.0 else self._prev_feat * gain
            # filtr maksimum zamiast rozmycia: mała, odległa dioda (kilka px)
            # zachowuje pełną siłę i wygrywa z dużą, ale słabszą plamą odblasku
            d = cv2.absdiff(f, prev)
            if self.cfg.energy_filter == "max":
                d = cv2.dilate(cv2.GaussianBlur(d, (3, 3), 0), np.ones((3, 3), np.uint8))
            else:
                d = cv2.GaussianBlur(d, (5, 5), 0)
            self.energy = d if self.energy is None else cv2.max(d, self.energy * self.cfg.energy_decay)
            if self.debug:
                diff = f - prev
                self.dbg_on, self.dbg_off = diff > 25, diff < -25
        self._prev_feat = f

    def _peaks(self) -> List[Peak]:
        """Lokalne szczyty energii mrugania: [(x, y, promień plamy, energia)], od najsilniejszego."""
        cfg = self.cfg
        if self.energy is None:
            return []
        e = self.energy
        k = cfg.peak_nms_px
        # to samo co (e >= dilate(e)) & (e >= próg) + np.nonzero, ale operacjami OpenCV (Pi: 16 -> ~7 ms)
        is_max = cv2.bitwise_and(cv2.compare(e, cv2.dilate(e, np.ones((k, k), np.uint8)), cv2.CMP_GE),
                                 cv2.compare(e, cfg.energy_threshold, cv2.CMP_GE))
        if self.debug:
            self.dbg_lit = e >= cfg.energy_threshold
        nz = cv2.findNonZero(is_max)
        nz = nz.reshape(-1, 2) if nz is not None else np.empty((0, 2), int)   # (N,1,2) albo (N,2) zależnie od wersji
        xs, ys = nz[:, 0], nz[:, 1]
        h, w = e.shape
        peaks = []
        for x, y in zip(xs, ys):
            v = float(e[y, x])
            # zwarta plamka (dioda), a nie fragment linii (krawędź przy ruchu albo
            # zmianie ekspozycji): na pierścieniu wokół szczytu energia wyraźnie spada
            ring = e[max(y - 14, 0):y + 15:7, max(x - 14, 0):x + 15:7]
            if np.median(ring) > 0.5 * v:
                continue
            x0, x1, y0, y1 = max(x - 3, 0), min(x + 4, w), max(y - 3, 0), min(y + 4, h)
            win = e[y0:y1, x0:x1]
            wsum = float(win.sum())
            gy, gx = np.mgrid[y0:y1, x0:x1]
            cx, cy = float((gx * win).sum() / wsum), float((gy * win).sum() / wsum)
            # promień plamy: ile pikseli w okolicy ma co najmniej połowę energii szczytu
            r0, r1, c0, c1 = max(y - 12, 0), min(y + 13, h), max(x - 12, 0), min(x + 13, w)
            area = int((e[r0:r1, c0:c1] >= 0.5 * v).sum())
            peaks.append((cx, cy, math.sqrt(area / math.pi), v))
        peaks.sort(key=lambda p: -p[3])
        return peaks[:60]

    # ---------------------------------------------------- przepływ optyczny
    def _refresh_points(self, t: LedTrack, gray: np.ndarray) -> None:
        """Punkty do przepływu: wyraźne narożniki w pierścieniu wokół diody (liczone tylko w wycinku)."""
        cfg = self.cfg
        if not cfg.flow:
            t.pts = None
            t.pts_frame = self._frame_idx
            return
        h, w = gray.shape
        r_in, r_out = cfg.flow_ring
        r_in = max(r_in, int(1.5 * t.radius))
        cx, cy = int(round(t.x)), int(round(t.y))
        x0, x1 = max(cx - r_out, 0), min(cx + r_out + 1, w)
        y0, y1 = max(cy - r_out, 0), min(cy + r_out + 1, h)
        t.pts_frame = self._frame_idx
        if x1 - x0 < 8 or y1 - y0 < 8:
            t.pts = None
            return
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.circle(mask, (cx - x0, cy - y0), r_out, 255, -1)
        cv2.circle(mask, (cx - x0, cy - y0), r_in, 0, -1)
        if self.energy is not None:
            mask[self.energy[y0:y1, x0:x1] >= cfg.energy_threshold] = 0   # mrugające piksele się nie nadają
        pts = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], cfg.flow_points, 0.01, 4, mask=mask)
        t.pts = (pts + np.array([x0, y0], np.float32)).astype(np.float32) if pts is not None else None

    def _flow(self, gray: np.ndarray) -> None:
        """Przesuwa każdy tor o ruch tekstury wokół jego diody (Lucas-Kanade, z kontrolą w przód i w tył)."""
        if self._prev_gray is None or not self.cfg.flow:
            return
        tracks = [t for t in self.tracks if t.pts is not None and len(t.pts) >= 3]
        if not tracks:
            return
        allp = np.concatenate([t.pts for t in tracks])
        lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, allp, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, nxt, None, **lk)
        good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((back - allp)[:, 0], axis=1) < 1.0)
        gx, gy = self.last_motion
        k = 0
        for t in tracks:
            n = len(t.pts)
            g = good[k:k + n]
            flow = (nxt[k:k + n] - t.pts)[:, 0][g]
            new_pts = nxt[k:k + n][g]
            k += n
            if len(flow) < 3:
                t.pts = new_pts if len(new_pts) else None
                continue
            # punkty tła (ściana za płytką) stoją, punkty obiektu się ruszają -
            # jeśli wyraźna część punktów się rusza, bierzemy ruch tej części
            rel = flow - np.array([gx, gy], np.float32)
            moving = np.linalg.norm(rel, axis=1) > 0.7
            sel = flow[moving] if moving.sum() >= max(3, 0.3 * len(flow)) else flow
            dx, dy = np.median(sel, axis=0)
            t.x += float(dx)
            t.y += float(dy)
            t.pts = new_pts
            t.flow_moved = True

    # ------------------------------------------------ ruch ze stałą prędkością
    def _predict(self, ts: float, dt: float) -> None:
        """
        motion_model="cv": tor, którego nie przesunął przepływ, jedzie dalej ze
        swoją prędkością. Dioda w kodzie gaśnie na ~1 chip; bez tego szuka jej
        tam, gdzie zgasła, a zapala się już kilkanaście px dalej -> nowy tor
        i odczyt od zera.
        """
        if self.cfg.motion_model != "cv" or dt <= 0:
            return
        for t in self.tracks:
            if not t.flow_moved and ts - t.last_seen_ts < self.cfg.cv_max_gap_ms:
                t.x += t.vx * dt
                t.y += t.vy * dt

    def _lock_point(self, t: LedTrack, feat: np.ndarray, ts: float) -> None:
        """
        point_lock: najjaśniejszy punkt cechy w oknie wokół (przewidzianego) toru.
        Jasny = dioda świeci -> tor na środek punktu, tor "widział" diodę. Ciemno =
        dioda zgaszona -> tor zostaje na przewidywaniu (motion_model="cv").
        """
        cfg = self.cfg
        h, w = feat.shape
        R = int(cfg.snap_px + t.radius + min(cfg.search_grow * (ts - t.last_seen_ts), 60.0))
        cx, cy = int(round(t.x)), int(round(t.y))
        x0, x1, y0, y1 = max(cx - R, 0), min(cx + R + 1, w), max(cy - R, 0), min(cy + R + 1, h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return
        win = feat[y0:y1, x0:x1]
        m = float(win.max())
        if m < max(cfg.point_min, cfg.point_rel * t.point_peak):
            t.point_peak *= 0.995                # powoli zapominamy (dioda może się oddalać)
            return
        t.point_peak = m if t.point_peak == 0 else 0.9 * t.point_peak + 0.1 * m
        ys, xs = np.nonzero(win >= 0.7 * m)
        wts = win[ys, xs].astype(np.float32)
        t.x = x0 + float((xs * wts).sum() / wts.sum())
        t.y = y0 + float((ys * wts).sum() / wts.sum())
        t.last_seen_ts = ts

    def _update_velocity(self, t: LedTrack, ts: float) -> None:
        """Prędkość = prosta dopasowana do położeń diody z ostatnich cv_window_ms (najmniejsze kwadraty)."""
        cfg = self.cfg
        if t.last_seen_ts == ts:
            t.obs.append((ts, t.x, t.y))
        pts = [o for o in t.obs if ts - o[0] <= cfg.cv_window_ms]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < 3 * self._frame_dt:
            t.vx = t.vy = 0.0
            return
        a = np.array(pts)
        tt = a[:, 0] - a[:, 0].mean()
        den = float((tt * tt).sum())
        vx = float((tt * (a[:, 1] - a[:, 1].mean())).sum() / den)
        vy = float((tt * (a[:, 2] - a[:, 2].mean())).sum() / den)
        sp = math.hypot(vx, vy)
        if sp > cfg.cv_max_px_per_ms:
            vx, vy = vx * cfg.cv_max_px_per_ms / sp, vy * cfg.cv_max_px_per_ms / sp
        t.vx, t.vy = vx, vy

    # -------------------------------------------------------- dopasowanie
    def _snap_and_spawn(self, peaks: List[Peak], ts: float, gray: np.ndarray) -> None:
        cfg = self.cfg
        used = set()
        # starsze tory wybierają pierwsze; każdy bierze najsilniejszy szczyt tuż obok
        for t in sorted(self.tracks, key=lambda t: t.created_ts):
            # tor, który od jakiegoś czasu nie widzi diody, szuka jej coraz szerzej
            reach = cfg.snap_px + t.radius + min(cfg.search_grow * (ts - t.last_seen_ts), 60.0)
            best = None
            for pi, (px, py, pr, pe) in enumerate(peaks):
                if pi in used or math.hypot(px - t.x, py - t.y) > reach:
                    continue
                if best is None or pe > peaks[best][3]:
                    best = pi
            if best is None:
                continue
            used.add(best)
            px, py, pr, _ = peaks[best]
            t.x += cfg.snap_gain * (px - t.x)
            t.y += cfg.snap_gain * (py - t.y)
            t.radius = 0.9 * t.radius + 0.1 * max(pr, 1.0)
            t.last_seen_ts = ts

        for pi, (px, py, pr, pe) in enumerate(peaks):
            if pi in used or pe < cfg.spawn_threshold:
                continue
            if any(math.hypot(px - t.x, py - t.y) < cfg.min_track_sep_px for t in self.tracks):
                continue
            if len(self.tracks) >= cfg.max_tracks:
                # wyrzucamy tor bez postępu w odczycie (bez ID, najmniejszy kontrast)
                idle = [t for t in self.tracks if t.drone_id is None and t.decoder.frames_ok == 0]
                if not idle:
                    break
                self.tracks.remove(min(idle, key=lambda t: (t.contrast, -t.created_ts)))
            t = LedTrack(self._next_tid, px, py, ts, ts,
                         TemplateDecoder(cfg.chip_hz, min_contrast=cfg.min_contrast,
                                         envelope=cfg.envelope), radius=max(pr, 2.0))
            self._refresh_points(t, gray)
            self._inherit_id(t, ts)
            self.tracks.append(t)
            self._next_tid += 1

    def _inherit_id(self, t: LedTrack, ts: float) -> None:
        """Nowy tor tuż przy miejscu, gdzie przed chwilą zginął tor z potwierdzonym ID -> ID wstępne."""
        live = {k.drone_id for k in self.tracks if k.drone_id is not None}
        best = None
        for lt, lx, ly, lid in self._lost:
            d = math.hypot(t.x - lx, t.y - ly)
            if lid not in live and ts - lt < 800.0 and d < 60.0:
                if best is None or d < best[0]:
                    best = (d, lid)
        if best is not None:
            t.drone_id, t.id_confirmed, t.inherited_ts = best[1], False, ts

    # ------------------------------------------------------------- źródło
    def _seek_source(self, t: LedTrack, feat: np.ndarray) -> None:
        """
        Gdy dioda świeci: przesuń tor na NAJJAŚNIEJSZY mrugający punkt w okolicy.
        Odblask (płytka, obudowa drona, dłoń) jest zawsze ciemniejszy od diody,
        która go oświetla - tor, który zsunął się na odblask, sam wraca na źródło.
        """
        if self.energy is None or t.lohi is None or not t.states or not t.states[-1]:
            return
        h, w = feat.shape
        R = self.cfg.source_search_px
        cx, cy = int(round(t.x)), int(round(t.y))
        x0, x1, y0, y1 = max(cx - R, 0), min(cx + R + 1, w), max(cy - R, 0), min(cy + R + 1, h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return
        e = self.energy[y0:y1, x0:x1]
        emax = float(e.max())
        if emax < self.cfg.energy_threshold:
            return
        # kandydaci: punkty, które mrugają MOCNO (nie tylko trochę) - prześwietlona
        # biała płytka obok diody ma 255, ale mruga słabo i tu odpada
        blinking = e >= max(self.cfg.energy_threshold, 0.6 * emax)
        patch = np.where(blinking, feat[y0:y1, x0:x1], 0).astype(np.float32)
        peak = float(patch.max())
        # jasność tam, gdzie tor siedzi teraz
        r = 4
        own_win = feat[max(cy - r, 0):min(cy + r + 1, h), max(cx - r, 0):min(cx + r + 1, w)]
        own = float(own_win.max()) if own_win.size else 0.0
        t.on_peak = 0.8 * t.on_peak + 0.2 * own if t.on_peak else own
        # przesuwamy tylko, gdy obok jest punkt WYRAŹNIE jaśniejszy - tor siedzący
        # na nasyconej diodzie stoi w miejscu (nie skacze między równie jasnymi pikselami)
        if peak < own + self.cfg.source_margin:
            return
        ys, xs = np.nonzero(patch >= peak - 4)      # środek najjaśniejszej części
        t.x, t.y = x0 + float(xs.mean()), y0 + float(ys.mean())

    def _lock_core(self, t: LedTrack, white: np.ndarray) -> None:
        """
        Źródło światła przepala matrycę do bieli (wszystkie kanały), odblask jest
        słabszy i zabarwiony. Gdy w okolicy toru jest rdzeń tak jasny jak ten,
        który ten tor już widział (dioda świeci), tor przeskakuje na jego środek -
        trzyma się samej diody także w ruchu. Zgaszona dioda: tylko przepływ.
        Okno jest małe, więc tor nie przeskakuje na INNY obiekt.
        """
        h, w = white.shape
        R = int(self.cfg.core_px)
        cx, cy = int(round(t.x)), int(round(t.y))
        x0, x1, y0, y1 = max(cx - R, 0), min(cx + R + 1, w), max(cy - R, 0), min(cy + R + 1, h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return
        win = white[y0:y1, x0:x1]
        m = float(win.max())
        t.core = max(m, t.core - 0.5)            # pamięć najjaśniejszego rdzenia, powoli zapominana
        if m < t.core - self.cfg.core_tol:
            return                               # dioda zgaszona albo to nie jej rdzeń
        ys, xs = np.nonzero(win >= m - self.cfg.core_tol)
        t.x, t.y = x0 + float(xs.mean()), y0 + float(ys.mean())

    def _adopt_synced_ids(self, ts: float) -> None:
        """
        Tor bez ID, który mruga DOKŁADNIE tak jak tor z potwierdzonym ID (to samo
        światło: dioda i jej odblask, albo dioda złapana od nowa po zgubieniu),
        dostaje to ID od razu - bez czekania na pełny odczyt.
        """
        n = int(0.6 * FRAME_CHIPS * self.cfg.chip_ms / max(self._frame_dt, 1.0))
        refs = [k for k in self.tracks if k.id_confirmed and len(k.levels) >= n]
        if not refs:
            return
        for t in self.tracks:
            if t.drone_id is not None or len(t.levels) < n or t.contrast < self.cfg.min_contrast:
                continue
            a = np.array(list(t.levels)[-n:])
            for k in refs:
                if math.hypot(k.x - t.x, k.y - t.y) > self.cfg.group_radius_px:
                    continue
                b = np.array(list(k.levels)[-n:])
                if a.std() < 1e-3 or b.std() < 1e-3:
                    continue
                if float(np.corrcoef(a, b)[0, 1]) >= self.cfg.sync_corr:
                    t.drone_id, t.id_confirmed, t.synced_with = k.drone_id, True, k.tid
                    t.last_seen_ts = ts
                    break

    # ------------------------------------------------------------------ sygnał
    def _sample(self, t: LedTrack, feat: np.ndarray) -> float:
        """
        Średnia jasność w oknie wokół diody (z poświatą). Surowa jasność:
        światło diody tylko DODAJE jasności, więc zapalona jest zawsze
        jaśniejsza. Zwraca surową wartość (normalizuje dekoder).
        """
        h, w = feat.shape
        r = int(min(max(round(t.radius) + self.cfg.sample_pad, 2), self.cfg.sample_max))
        x0, x1 = max(int(round(t.x)) - r, 0), min(int(round(t.x)) + r + 1, w)
        y0, y1 = max(int(round(t.y)) - r, 0), min(int(round(t.y)) + r + 1, h)
        t.roi = (x0, y0, x1, y1)
        level = float(feat[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0
        t.levels.append(level)
        if len(t.levels) >= 6 and (len(t.levels) % 3 == 0 or t.lohi is None):
            t.lohi = tuple(np.percentile(np.array(t.levels), (10, 90)))
            t.contrast = float(t.lohi[1] - t.lohi[0])
        return level

    def _still_blinking(self, t: LedTrack, ts: float) -> bool:
        """Tor, który od ~1 s nie mruga (dioda zniknęła, tor wisi na tle), jest usuwany."""
        n = int(1000.0 / max(self._frame_dt, 1.0))
        if ts - t.created_ts < 1000.0 or len(t.levels) < n:
            return True
        recent = list(t.levels)[-n:]
        return max(recent) - min(recent) >= self.cfg.min_contrast

    # ------------------------------------------------------------------ główne
    def process(self, frame_bgr: np.ndarray, ts: float,
                roi: Optional[Tuple[float, float, float, float]] = None) -> List[DroneObs]:
        """
        roi = (x0, y0, x1, y1) w układzie obrazu pomniejszonego, np. ramka drona z
        detektora obiektów: nowi kandydaci na diodę tylko w niej. None = cały kadr.
        """
        cfg = self.cfg
        h, w = frame_bgr.shape[:2]
        if w > cfg.proc_width:
            img = cv2.resize(frame_bgr, (cfg.proc_width, int(round(h * cfg.proc_width / w))),
                             interpolation=cv2.INTER_AREA)
        else:
            img = frame_bgr
        self.small = img
        self._frame_idx += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(img)
        white = cv2.min(cv2.min(b, g), r)   # biel: źródło światła przepala wszystkie kanały
        if cfg.led_color:
            feat = chroma(img, cfg.led_color)
        elif cfg.feature == "max":
            feat = brightness(img)
        elif cfg.feature == "min":
            feat = white
        else:
            # luminancja, nie max(B,G,R): nasycona dioda (zapalona 255, zgaszona
            # soczewka nadal ~230 w R) zmienia się głównie w G i B - max tego nie widzi
            feat = gray

        dt = ts - self._last_ts if self._last_ts is not None else 0.0
        self._last_ts = ts
        if dt > 0:
            self._frame_dt = 0.9 * self._frame_dt + 0.1 * dt
        if cfg.motion_comp:
            self._compensate_motion(gray)
        for t in self.tracks:
            t.flow_moved = False
        self._flow(gray)
        self._predict(ts, dt)
        self._update_energy(feat)
        peaks = self._peaks()
        if roi is not None:
            x0, y0, x1, y1 = roi
            peaks = [p for p in peaks if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
        self.last_events = [(px, py, pr, 1) for px, py, pr, _ in peaks]
        self._snap_and_spawn(peaks, ts, gray)

        frame_ms = FRAME_CHIPS * cfg.chip_ms
        for t in self.tracks:
            if t.pts is None or len(t.pts) < 6 or self._frame_idx - t.pts_frame >= 15:
                self._refresh_points(t, gray)
            if cfg.seek_source:
                self._seek_source(t, feat)
            if cfg.core_lock:
                self._lock_core(t, white)
            if cfg.point_lock:
                self._lock_point(t, feat, ts)
            if cfg.motion_model == "cv":
                self._update_velocity(t, ts)
            level = self._sample(t, {"white": white, "gray": gray}.get(cfg.signal, feat))
            t.states.append(int(t.lohi is not None and level > 0.5 * sum(t.lohi)))
            t.decoder.push(ts, level)
            dec = t.decoder
            if t.synced_with is not None:
                # ID z synchronizacji trzyma się, dopóki tor dalej mruga zgodnie
                if dec.last_id is not None and fresh_ok(dec, ts, frame_ms) and dec.last_id != t.drone_id:
                    t.synced_with = None          # własny odczyt mówi co innego - wygrywa odczyt
                else:
                    continue
            # ID jest ważne, dopóki dekoder je ponownie dopasowuje (dioda mruga dalej)
            fresh = ts - dec.last_ok_ts < 1.5 * frame_ms
            if not fresh:
                dec.confirmed_id = None
            if dec.confirmed_id is not None and dec.last_id == dec.confirmed_id:
                t.drone_id, t.id_confirmed, t.inherited_ts = dec.confirmed_id, True, None
            elif dec.last_id is not None and fresh:
                t.drone_id, t.id_confirmed = dec.last_id, False
            elif t.inherited_ts is not None and ts - t.inherited_ts < 1.3 * frame_ms:
                pass                             # przejęte ID czeka na odczyt
            else:
                t.drone_id, t.id_confirmed, t.inherited_ts = None, False, None
        if cfg.sync_adopt:
            self._adopt_synced_ids(ts)
        self._prev_gray = gray

        H, W = feat.shape
        alive = []
        for t in self.tracks:
            if ts - t.last_seen_ts < cfg.lost_ms and 0 <= t.x < W and 0 <= t.y < H and self._still_blinking(t, ts):
                alive.append(t)
            elif t.drone_id is not None and t.id_confirmed:
                self._lost.append((ts, t.x, t.y, t.drone_id))
        self.tracks = alive
        live_tids = {t.tid for t in self.tracks}
        for t in self.tracks:
            if t.synced_with is not None and t.synced_with not in live_tids and not t.decoder.confirmed_id:
                # tor-wzór zniknął: zostaje przy ID, dopóki sam go nie potwierdzi (jak przejęte ID)
                t.synced_with, t.id_confirmed, t.inherited_ts = None, False, ts
        self._lost = [e for e in self._lost if ts - e[0] < 800.0]
        self._dedupe()
        return self._group((H, W), ts)

    def _dedupe(self) -> None:
        """
        Dwa tory na tej samej diodzie -> zostaje ten, który jest dalej z odczytem
        (potwierdzone ID, więcej dopasowań, większy kontrast), a nie po prostu
        starszy - inaczej scalenie z torem na odblasku kasowało postęp odczytu.
        """
        keep: List[LedTrack] = []
        order = sorted(self.tracks, key=lambda t: (not t.id_confirmed, t.drone_id is None,
                                                   -t.decoder.frames_ok, -t.contrast, t.created_ts))
        for t in order:
            if all(math.hypot(t.x - k.x, t.y - k.y) > max(6.0, k.radius) for k in keep):
                keep.append(t)
        self.tracks = keep

    def _group(self, shape, ts: float) -> List[DroneObs]:
        h, w = shape
        fx = (w / 2) / math.tan(math.radians(self.cfg.hfov_deg) / 2)
        by_id: Dict[int, List[LedTrack]] = {}
        for t in self.tracks:
            # ID przejęte (po zgubionym torze) to tylko
            # podpowiedź dla dekodera - w testach w ~95% stało poza diodą, więc
            # pozycję drona wyznaczają tylko tory, które same odczytały kod
            if t.drone_id is not None and (t.inherited_ts is None or not self.cfg.anchor_own_only):
                by_id.setdefault(t.drone_id, []).append(t)
        drones = []
        for drone_id, members in by_id.items():
            # dioda oświetla też otoczenie (płytka, dłoń, twarz) i te odblaski mrugają
            # tym samym kodem - pozycję bierzemy z najsilniejszych źródeł z tym ID
            # pozycja = samo źródło światła: tor z najjaśniejszym punktem w stanie
            # "zapalona" (odblask jest zawsze ciemniejszy od diody, która go oświetla)
            if self.cfg.anchor in ("core", "core_conf"):
                # źródło = tor z najjaśniejszym białym rdzeniem; odblaski (ciemniejsze)
                # nie przesuwają pozycji. Kilka diod drona = kilka równie jasnych źródeł.
                key = (lambda t: (t.id_confirmed, t.core)) if self.cfg.anchor == "core_conf" else (lambda t: t.core)
                anchor = max(members, key=key)
                group = [t for t in members
                         if math.hypot(t.x - anchor.x, t.y - anchor.y) < self.cfg.group_radius_px
                         and t.core >= anchor.core - 3]
                if self.cfg.core_drop:
                    # źródło nie ciemnieje nagle o kilkadziesiąt poziomów - jeśli najjaśniejszy
                    # tor z tym ID jest dużo ciemniejszy niż źródło przed chwilą, dioda
                    # zniknęła (zasłonięta, poza kadrem), a zostały odblaski: brak pozycji
                    prev, pts_ = self._src_core.get(drone_id, (0.0, ts))
                    src = max(anchor.core, prev - 0.01 * (ts - pts_))     # zapominanie 10 poziomów/s
                    self._src_core[drone_id] = (src, ts)
                    if anchor.core < src - self.cfg.core_drop:
                        continue
            else:
                anchor = max(members, key=lambda t: (t.on_peak, t.contrast))
                group = [t for t in members
                         if math.hypot(t.x - anchor.x, t.y - anchor.y) < self.cfg.group_radius_px
                         and t.on_peak >= anchor.on_peak - 10 and t.contrast >= 0.5 * anchor.contrast]
            xs = np.array([t.x for t in group])
            ys = np.array([t.y for t in group])
            cx, cy = float(xs.mean()), float(ys.mean())
            drones.append(DroneObs(
                drone_id=drone_id, x=cx, y=cy,
                azimuth_deg=math.degrees(math.atan((cx - w / 2) / fx)),
                elevation_deg=math.degrees(math.atan((h / 2 - cy) / fx)),
                n_leds=len(group), spread_px=float(math.hypot(np.ptp(xs), np.ptp(ys))),
                track_ids=[t.tid for t in group],
                confirmed=any(t.id_confirmed for t in group)))
        return sorted(drones, key=lambda d: d.drone_id)
