"""
Protokół optycznej identyfikacji drona: diody nadają w kółko kod z codebook.py.

Kod (gen_codes.py): cykliczny ciąg CHIPS = 26 chipów, 128 różnych ID.
  * dioda świeci ~75% czasu i gaśnie najwyżej na 1 chip -> zawsze prawie
    widoczna, tor nie gubi jej przy ruchu
  * dwa różne ID różnią się w >= 4 chipach przy KAŻDYM przesunięciu w czasie
    -> nie trzeba synchronizacji ani znajomości fazy

Dekodowanie = dopasowanie wzorca (matched filter), bez ML:
  dla każdego ID i każdego przesunięcia w czasie liczymy, jak bardzo
  zmierzony przebieg jasności z ostatnich ~1.2 ramki zgadza się z kodem.
  Wygrywa najlepsze dopasowanie, jeśli jest wysokie i wyraźnie lepsze niż
  najlepsze INNE ID. Pojedyncze złe klatki tylko trochę obniżają wynik
  (w starym dekoderze jeden zły odcinek kasował cały odczyt).

Wystarczy ~3 klatki na chip (30 fps -> CHIP_HZ = 10, 60 fps -> 20).
Kody MUSZĄ się zgadzać z firmware/led_beacon/codes.h (oba z gen_codes.py).
"""

from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from codebook import CHIPS, CODES, MAX_DARK

N_IDS = len(CODES)
DEFAULT_CHIP_HZ = 10.0
FRAME_CHIPS = CHIPS              # ramka = jeden pełny obieg kodu

PHASE_STEPS = 4                  # rozdzielczość szukania fazy: 1/4 chipu
# zegar nadajnika (rezonator Arduino ~0.5%) i kamery nie są idealne - na oknie
# ~3 s daje to do kilkudziesięciu ms rozjazdu, więc sprawdzamy kilka temp
RATES = (0.993, 1.0, 1.007)
EDGE_ZONE = 0.2                  # próbka bliżej niż 0.2 chipu od zbocza waży mniej
EDGE_WEIGHT = 0.3


def encode_frame(drone_id: int) -> List[int]:
    """Chipy jednego obiegu kodu (1 = dioda świeci), w kolejności nadawania."""
    if not 0 <= drone_id < N_IDS:
        raise ValueError(f"ID musi być w zakresie 0..{N_IDS - 1}")
    code = CODES[drone_id]
    return [(code >> (CHIPS - 1 - i)) & 1 for i in range(CHIPS)]


_SIGNS = np.array([[2 * c - 1 for c in encode_frame(i)] for i in range(N_IDS)], dtype=np.float32)


class TemplateDecoder:
    """
    Dekoder dla jednego źródła światła. Dostaje (czas_ms, jasność) z każdej
    klatki - surową, w dowolnej skali. Przy każdej ocenie sam normalizuje
    próbki do [-1, 1] progiem z percentyli 10/90 DOKŁADNIE tych próbek, które
    dopasowuje (próg policzony z pierwszych kilku klatek bywa zły, a próbki
    znormalizowane nim na stałe psułyby całe okno).
    """

    def __init__(self, chip_hz: float = DEFAULT_CHIP_HZ, candidates: Optional[Sequence[int]] = None,
                 min_score: float = 0.8, min_margin: float = 0.06, min_contrast: float = 0.0,
                 min_coverage: float = 0.8, envelope: bool = False):
        self.chip_ms = 1000.0 / chip_hz
        self.frame_ms = CHIPS * self.chip_ms
        self.ids = np.array(candidates if candidates is not None else range(N_IDS))
        self.signs = _SIGNS[self.ids]
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_contrast = min_contrast          # w jednostkach jasności; mniej = "nie mruga"
        self.min_coverage = min_coverage          # od jakiej części ramki próbujemy decydować
        self.envelope = envelope                  # próg względem lokalnego poziomu "zapalona" (patrz _normalize)
        self.samples: Deque[Tuple[float, float]] = deque()

        self.last_id: Optional[int] = None        # ostatnie dopasowane ID (wstępne)
        self.confirmed_id: Optional[int] = None   # ID potwierdzone 2 zgodnymi dopasowaniami
        self.confirmed_ts: float = float("-inf")
        self._cand_since = float("-inf")
        self.last_score = 0.0                     # dopasowanie najlepszego ID (0..1)
        self.last_margin = 0.0                    # przewaga nad najlepszym innym ID
        self.coverage = 0.0                       # jaką część ramki obejmują próbki
        self.frames_ok = 0                        # ile razy potwierdzono ID (max 1 na pół ramki)
        self.frames_bad = 0                       # ile razy przy pełnej ramce wynik był niejednoznaczny
        self.last_ok_ts = float("-inf")          # kiedy ostatnio ID zostało (ponownie) dopasowane
        self._n_since_eval = 0

    def push(self, ts_ms: float, value: float) -> Optional[int]:
        """Dodaje próbkę jasności. Zwraca ID, gdy dopasowanie jest pewne."""
        self.samples.append((float(ts_ms), float(value)))
        while self.samples and ts_ms - self.samples[0][0] > 1.2 * self.frame_ms:
            self.samples.popleft()
        self.coverage = min(1.0, (ts_ms - self.samples[0][0]) / self.frame_ms)
        self._n_since_eval += 1
        if self.coverage < self.min_coverage or self._n_since_eval < 2:
            return None
        self._n_since_eval = 0
        return self._evaluate(ts_ms)

    def push_series(self, ts_ms: float, t: np.ndarray, v: np.ndarray) -> Optional[int]:
        """
        Ocena na gotowym oknie próbek (czasy t rosnąco, ostatnia = teraz) zamiast
        dokładania po jednej: sygnał wspólny konstelacji składany co klatkę od nowa
        z torów, które akurat do niej należą. Stan potwierdzania ID zostaje.
        """
        keep = t >= ts_ms - 1.2 * self.frame_ms
        self.samples = deque(zip(t[keep].tolist(), v[keep].tolist()))
        if not self.samples:
            self.coverage = 0.0
            return None
        self.coverage = min(1.0, (ts_ms - self.samples[0][0]) / self.frame_ms)
        if self.coverage < self.min_coverage:
            return None
        return self._evaluate(ts_ms)

    def reset(self) -> None:
        self.samples.clear()
        self.coverage = 0.0

    def _looks_like_beacon(self, t: np.ndarray, s: np.ndarray) -> bool:
        """
        Szybki test zgodności z protokołem, zanim w ogóle zaczniemy dopasowywać:
        nadajnik nigdy nie gaśnie na dłużej niż MAX_DARK chip i świeci 62-80%
        czasu. Losowe mruganie (ręka, odblask), migotanie lampy czy szum tego
        nie spełniają - dzięki temu nie dostaną przypadkowego ID.
        """
        dark = s < 0
        duty = 1.0 - dark.mean()
        if not 0.5 <= duty <= 0.92:
            return False
        dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
        start = None
        for ti, d in zip(t, dark):
            if d and start is None:
                start = ti
            elif not d and start is not None:
                if ti - start > (MAX_DARK + 0.6) * self.chip_ms + dt:
                    return False
                start = None
        return True

    def _normalize(self, t: np.ndarray, v: np.ndarray) -> Optional[np.ndarray]:
        """
        Próbki jasności -> [-1 zgaszona, +1 zapalona].

        envelope=False: stały próg z percentyli 10/90 całego okna.
        envelope=True: nadajnik nigdy nie gaśnie na dłużej niż MAX_DARK chip, więc
        w każdym oknie +-(MAX_DARK + 0.5) chipu jest stan "zapalona" - lokalne
        maksimum to AKTUALNY poziom zapalonej diody, nawet gdy tło za diodą czy
        jej odległość zmieniają się w ruchu. Zgaszona = wyraźny spadek względem
        niego. Na całym kadrze daje to ID losowym rzeczom - używać w ramce drona.
        """
        if not self.envelope:
            lo, hi = np.percentile(v, (10, 90))
            if hi - lo <= max(self.min_contrast, 1e-6):
                return None
            return np.clip((v - 0.5 * (lo + hi)) / (0.5 * (hi - lo)), -1.0, 1.0)
        half = (MAX_DARK + 0.5) * self.chip_ms
        a = np.searchsorted(t, t - half, side="left")
        b = np.searchsorted(t, t + half, side="right")
        env = np.array([v[i:j].max() for i, j in zip(a, b)], dtype=np.float32)
        d = v - env                                       # 0 = zapalona, < 0 = spadek
        depth = -float(np.percentile(d, 10))              # typowa głębokość zgaszenia
        if depth <= max(self.min_contrast, 1e-6):
            return None
        s = np.clip(1.0 + 2.0 * d / depth, -1.0, 1.0)
        # nadajnik ma DWA stany; szum i migotanie mają dużo wartości pośrednich
        if (np.abs(s) < 0.5).mean() > 0.25:
            return None
        return s

    def _id_scores(self, now: float) -> Optional[np.ndarray]:
        """Dopasowanie okna próbek do każdego ID (najlepsze po fazach i tempach), None = to nie nadajnik."""
        t = np.array([x[0] for x in self.samples], dtype=np.float64) - now
        v = np.array([x[1] for x in self.samples], dtype=np.float32)
        s = self._normalize(t, v)
        if s is None or not self._looks_like_beacon(t, s):
            return None
        phases = np.arange(CHIPS * PHASE_STEPS) * (self.chip_ms / PHASE_STEPS)
        # hipotezy (tempo, faza) razem: pozycja próbki w kodzie, w chipach
        pos = np.concatenate([(t[None, :] * r + phases[:, None]) / self.chip_ms for r in RATES])
        idx = np.floor(pos).astype(np.int64) % CHIPS
        frac = pos - np.floor(pos)
        w = np.where((frac > EDGE_ZONE) & (frac < 1 - EDGE_ZONE), 1.0, EDGE_WEIGHT).astype(np.float32)
        ws = w * s[None, :]                                              # H x n
        # scores[k, h] = sum_i ws[h, i] * znak_kodu_k[idx[h, i]]
        scores = np.einsum("khn,hn->kh", self.signs[:, idx], ws)
        scores /= (w * np.abs(s)[None, :]).sum(axis=1)[None, :]
        return scores.max(axis=1)

    def _evaluate(self, now: float) -> Optional[int]:
        per_id = self._id_scores(now)
        if per_id is None:
            self.last_score = self.last_margin = 0.0
            return None
        order = np.argsort(per_id)[::-1]
        best = per_id[order[0]]
        second = per_id[order[1]] if len(order) > 1 else -1.0
        self.last_score, self.last_margin = float(best), float(best - second)
        # przy niepełnej ramce inne ID mogą pasować tak samo dobrze - wymagamy więcej
        full = self.coverage >= 0.95
        margin_needed = self.min_margin * (1.0 if full else 1.5)
        score_needed = self.min_score if full else max(self.min_score, 0.9)
        if best < score_needed or best - second < margin_needed:
            if self.coverage >= 0.95:
                self.frames_bad += 1
            return None
        drone_id = int(self.ids[order[0]])
        if drone_id != self.last_id:
            # nowy kandydat - potwierdzi go dopiero kolejne zgodne dopasowanie
            # co najmniej pół ramki później (w dużej części na nowych próbkach)
            self._cand_since = now
            if self.confirmed_id is not None and self.confirmed_id != drone_id:
                self.confirmed_id = None
        elif now - self._cand_since >= 0.5 * self.frame_ms:
            if self.confirmed_id != drone_id:
                self.confirmed_ts = now
            self.confirmed_id = drone_id
        if drone_id != self.last_id or now - self.last_ok_ts > 0.5 * self.frame_ms:
            self.frames_ok += 1
            self.last_ok_ts = now
        self.last_id = drone_id
        return drone_id
