"""
Śledzenie torów diod: filtr Kalmana (stała prędkość) + algorytm węgierski.

Zastępuje przepływ optyczny (Lucas-Kanade):
  * KalmanCV przewiduje, gdzie dioda będzie w tej klatce - także gdy dioda
    jest zgaszona (bit 0 w kodzie) albo wokół niej nie ma tekstury (niebo,
    krótka ekspozycja), i wygładza położenie z pomiarów,
  * hungarian() przydziela szczyty energii mrugania torom globalnie
    (minimalny łączny koszt), a nie zachłannie - dwa tory blisko siebie nie
    kradną sobie nawzajem diod.

Bez scipy (na Pi nie ma po co go doinstalowywać) - macierze są małe
(~24 tory x ~60 szczytów), wersja O(n^2 m) w numpy wystarcza z zapasem.
"""

from typing import List, Tuple

import numpy as np

_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


class KalmanCV:
    """
    Stan [x, y, vx, vy] w px i px/ms. Model: stała prędkość, szum = losowe
    przyspieszenie o odchyleniu accel [px/ms^2] (ruch ręką / dronem).
    """

    __slots__ = ("x", "P", "accel")

    def __init__(self, x: float, y: float, pos_std: float = 3.0, vel_std: float = 0.3, accel: float = 0.01):
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([pos_std ** 2, pos_std ** 2, vel_std ** 2, vel_std ** 2])
        self.accel = accel

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0.0], [0.0, dt * dt / 2], [dt, 0.0], [0.0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (self.accel ** 2) * (G @ G.T)

    def correct(self, zx: float, zy: float, meas_std: float) -> None:
        S = _H @ self.P @ _H.T + np.eye(2) * meas_std ** 2
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([zx, zy]) - _H @ self.x)
        # postać Josepha - P zostaje symetryczna i dodatnio określona
        IKH = np.eye(4) - K @ _H
        self.P = IKH @ self.P @ IKH.T + K @ (np.eye(2) * meas_std ** 2) @ K.T

    def shift(self, dx: float, dy: float) -> None:
        """Przesunięcie całego obrazu (kompensacja ruchu kamery)."""
        self.x[0] += dx
        self.x[1] += dy

    def stop(self) -> None:
        """Długo bez pomiaru: tor przestaje jechać i stoi (szuka diody w miejscu)."""
        self.x[2:] = 0.0

    def clamp_speed(self, vmax: float) -> None:
        sp = float(np.hypot(self.x[2], self.x[3]))
        if sp > vmax:
            self.x[2:] *= vmax / sp

    @property
    def pos_std(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))


def hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    Przydział o minimalnym łącznym koszcie dla macierzy n x m (dowolny kształt).
    Zwraca pary (wiersz, kolumna); każdy wiersz i każda kolumna najwyżej raz,
    przydzielonych jest min(n, m) par. Zakazane pary: duży koszt + odfiltrowanie
    po stronie wywołującego.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    transposed = n > m
    if transposed:
        cost = cost.T
        n, m = m, n
    inf = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)      # p[j] = wiersz przydzielony kolumnie j (1-indeksowane, 0 = brak)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j]]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)
