"""
Konstelacje: grupowanie diod, które mrugają TYM SAMYM kodem - na bieżąco,
klatka po klatce, zanim dekoder odczyta jakiekolwiek ID.

Wszystkie diody jednego drona dostają ten sam sygnał, więc zapalają się i gasną
w tych samych klatkach. Diody dwóch różnych dronów nadają inne kody (i mają
inną fazę), więc prędzej czy później jedna zgaśnie, a druga będzie dalej
świecić. Dla każdej pary torów zbieramy więc dwa rodzaje dowodów:

  * ZA  - wspólne zdarzenie: oba tory zapaliły się albo oba zgasły w tej samej
          klatce (+-1 klatka: klatka naświetlana na zboczu daje diodę
          "pół-jasną" i progi dwóch torów mogą ją odczytać różnie),
  * PRZECIW - jeden tor świeci, a drugi jest zgaszony przez >= 2 klatki z rzędu
          (pojedyncza klatka niezgody to zwykle zbocze, a nie inny kod).
          To jest "to, co zgasło, i to, co dalej świeci, nie mogą być w jednej
          grupie".

Para jest ZGODNA po kilku wspólnych zdarzeniach bez żadnego dowodu przeciw
(i gdy ta zgodność utrzyma się jeszcze przez 1 chip);
dowód przeciw kasuje zebrane dowody za i blokuje parę. Oba liczniki powoli
wygasają (połowa po jednym obiegu kodu), więc jednorazowa pomyłka śledzenia nie
rozdziela diod na zawsze, a tory z różnymi kodami są blokowane wciąż od nowa.

Grupy = łączenie zgodnych par od najmocniejszej, ale nigdy tak, żeby w jednej
grupie znalazła się zablokowana para. Numer grupy (gid) jest trwały: grupa
zachowuje numer, dopóki zostaje w niej większość torów.

Pierwsza wspólna "dziura" w świeceniu (zgaśnięcie + zapalenie) wystarcza, czyli
grupa powstaje po kilku chipach (~0.3-0.8 s przy 10 Hz), a nie po całym obiegu
kodu i potwierdzeniu (~3.5 s).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class BlinkGrouper:
    def __init__(self, code_ms: float, chip_ms: float, min_events: float = 2.0,
                 radius_px: float = 150.0, block_level: float = 0.5, near_weight: float = 0.5):
        self.code_ms = code_ms                  # czas jednego obiegu kodu (pamięć dowodów)
        self.chip_ms = chip_ms
        self.min_events = min_events            # ile wspólnych zdarzeń łączy parę
        self.radius_px = radius_px              # diody jednego drona są bliżej niż to
        self.near_weight = near_weight          # waga wspólnego zdarzenia przesuniętego o 1 klatkę (w tej samej = 1)
        self.block_level = block_level          # para zablokowana, dopóki "przeciw" >= tyle
        self._tids: List[int] = []
        self._last = np.zeros(0, np.int8)       # ostatni pewny stan toru (+1 świeci, -1 zgaszony, 0 nieznany)
        self._ev_prev = np.zeros(0, np.int8)    # zdarzenie toru w poprzedniej klatce
        self._run = np.zeros((0, 0), np.int32)  # ile klatek z rzędu para się nie zgadza
        self._sync = np.zeros((0, 0))           # dowody za (wspólne zdarzenia, wygasające)
        self._strikes = np.zeros((0, 0))        # dowody przeciw (wygasające)
        self._ok_ms = np.zeros((0, 0))          # od ilu ms para jest zgodna bez przerwy
        self._gid: Dict[int, int] = {}
        self._next_gid = 1
        self._last_ts: Optional[float] = None
        self._frame_dt = 33.3
        self.groups: Dict[int, List[int]] = {}  # gid -> tory (także grupy jednoelementowe)

    # ---------------------------------------------------------------- stan
    def _reindex(self, tids: Sequence[int]) -> None:
        """Przestawia tablice na nową listę torów (tory mogły zniknąć, dojść, zmienić kolejność)."""
        if list(tids) == self._tids:
            return
        pos = {t: i for i, t in enumerate(self._tids)}
        n = len(tids)
        new = np.array([i for i, t in enumerate(tids) if t in pos], dtype=np.intp)
        old = np.array([pos[t] for t in tids if t in pos], dtype=np.intp)
        last, ev = np.zeros(n, np.int8), np.zeros(n, np.int8)
        last[new], ev[new] = self._last[old], self._ev_prev[old]
        self._last, self._ev_prev = last, ev
        for name in ("_run", "_sync", "_strikes", "_ok_ms"):
            m_old = getattr(self, name)
            m = np.zeros((n, n), m_old.dtype)
            if len(new):
                m[np.ix_(new, new)] = m_old[np.ix_(old, old)]
            setattr(self, name, m)
        self._tids = list(tids)
        live = set(tids)
        self._gid = {t: g for t, g in self._gid.items() if t in live}

    def block(self, tid_a: int, tid_b: int) -> None:
        """Twardy dowód z zewnątrz, że dwa tory to różne nadajniki (np. dekoder odczytał dwa różne ID)."""
        if tid_a in self._tids and tid_b in self._tids and tid_a != tid_b:
            i, j = self._tids.index(tid_a), self._tids.index(tid_b)
            self._strikes[i, j] = self._strikes[j, i] = max(self._strikes[i, j], 2.0)
            self._sync[i, j] = self._sync[j, i] = 0.0

    def evidence(self, tid_a: int, tid_b: int) -> Tuple[float, float]:
        """(dowody za, dowody przeciw) dla pary torów - do podglądu i testów."""
        i, j = self._tids.index(tid_a), self._tids.index(tid_b)
        return float(self._sync[i, j]), float(self._strikes[i, j])

    # -------------------------------------------------------------- klatka
    def update(self, ts: float, tids: Sequence[int], xy: np.ndarray, states: Sequence[int]) -> Dict[int, int]:
        """
        Jedna klatka. states[i]: +1 dioda toru świeci, -1 zgaszona, 0 nie wiadomo
        (tor jeszcze bez progu albo jasność pośrednia). Zwraca tor -> numer grupy.
        """
        self._reindex(tids)
        n = len(tids)
        dt = ts - self._last_ts if self._last_ts is not None else 0.0
        self._last_ts = ts
        if dt > 0:
            self._frame_dt = 0.9 * self._frame_dt + 0.1 * dt
        if n == 0:
            self.groups = {}
            return {}
        s = np.asarray(states, dtype=np.int8)
        # zdarzenie: pewny stan inny niż ostatni pewny (+1 zapaliła się, -1 zgasła)
        ev = np.where((s != 0) & (self._last != 0) & (s != self._last), s, 0).astype(np.int8)
        self._last = np.where(s != 0, s, self._last).astype(np.int8)

        prod = s[:, None].astype(np.int32) * s[None, :]
        dis, agr = prod < 0, prod > 0
        self._run = np.where(dis, self._run + 1, np.where(agr, 0, self._run))
        # dioda gaśnie na 1 chip; "przeciw" = niezgoda przez pół chipu, ale co najmniej 2 klatki
        need = max(2, int(round(0.5 * self.chip_ms / max(self._frame_dt, 1.0))))
        strike = dis & (self._run == need)

        e_i, e_j = ev[:, None], ev[None, :]
        p_i, p_j = self._ev_prev[:, None], self._ev_prev[None, :]
        same = (e_i != 0) & (e_j == e_i)
        near = ((e_i != 0) & (e_j == 0) & (p_j == e_i)) | ((e_j != 0) & (e_i == 0) & (p_i == e_j))
        co = same + self.near_weight * near
        self._ev_prev = ev

        decay = 0.5 ** (max(dt, 0.0) / self.code_ms)
        self._sync = self._sync * decay + co
        self._strikes = self._strikes * decay + 2.0 * strike
        self._sync[self._strikes >= self.block_level] = 0.0
        np.fill_diagonal(self._sync, 0.0)

        xy = np.asarray(xy, dtype=np.float64).reshape(n, 2)
        dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        blocked = self._strikes >= self.block_level
        ok = (self._sync >= self.min_events - 0.5) & ~blocked & (dist <= self.radius_px)
        # zgodność musi się utrzymać przez 1 chip: przypadkowo wspólna "dziura" dwóch różnych
        # kodów zwykle rozjeżdża się zaraz potem, zanim para zdąży się połączyć
        self._ok_ms = np.where(ok, self._ok_ms + max(dt, 0.0), 0.0)
        edge = ok & (self._ok_ms >= self.chip_ms)
        return self._cluster(tids, edge, blocked)

    def _cluster(self, tids: Sequence[int], edge: np.ndarray, blocked: np.ndarray) -> Dict[int, int]:
        n = len(tids)
        comp = list(range(n))                      # tor -> numer składowej
        members: Dict[int, List[int]] = {i: [i] for i in range(n)}
        ii, jj = np.nonzero(np.triu(edge, 1))
        for k in np.argsort(-self._sync[ii, jj], kind="stable"):
            a, b = comp[ii[k]], comp[jj[k]]
            if a == b or blocked[np.ix_(members[a], members[b])].any():
                continue
            for m in members[b]:
                comp[m] = a
            members[a] += members.pop(b)

        # trwałe numery: większa grupa wybiera pierwsza, bierze numer większości swoich torów
        out: Dict[int, int] = {}
        taken = set()
        for mem in sorted(members.values(), key=lambda m: (-len(m), min(tids[i] for i in m))):
            votes: Dict[int, int] = {}
            for i in mem:
                g = self._gid.get(tids[i])
                if g is not None and g not in taken:
                    votes[g] = votes.get(g, 0) + 1
            if votes:
                gid = max(votes, key=lambda g: (votes[g], -g))
            else:
                gid = self._next_gid
                self._next_gid += 1
            taken.add(gid)
            for i in mem:
                out[tids[i]] = gid
        self._gid = out
        self.groups = {}
        for t, g in out.items():
            self.groups.setdefault(g, []).append(t)
        return out
