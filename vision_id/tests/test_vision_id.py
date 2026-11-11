import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codebook import CHIPS, CODES, MAX_DARK, MIN_DIST  # noqa: E402
from protocol import N_IDS, TemplateDecoder, encode_frame  # noqa: E402
from simulate import run  # noqa: E402
from tracking import KalmanCV, hungarian  # noqa: E402


def _rot(bits, r):
    return bits[r:] + bits[:r]


def feed(drone_id, fps=30, hz=10, seconds=8, noise=0.2, flip=0.03, seed=0):
    """Symuluje kamerę (pół-klatkowa ekspozycja, szum, przekłamania) -> (ID wstępne, ID potwierdzone, czas)."""
    rng = random.Random(seed)
    dec = TemplateDecoder(hz)
    chips = encode_frame(drone_id)
    chip_ms = 1000.0 / hz
    phase = rng.uniform(0, 3000)
    for f in range(int(seconds * fps)):
        t = f * 1000.0 / fps + rng.uniform(-2, 2)
        on = sum(chips[int((t + phase + k * (500.0 / fps) / 6) // chip_ms) % CHIPS] for k in range(6)) / 6
        s = 2 * on - 1 + rng.gauss(0, noise)
        if rng.random() < flip:
            s = -s
        dec.push(t, max(-1.0, min(1.0, s)))
        if dec.confirmed_id is not None:
            return dec.last_id, dec.confirmed_id, t
    return dec.last_id, None, None


def test_codebook_properties():
    assert len(CODES) == N_IDS == 128
    codes = [encode_frame(i) for i in range(N_IDS)]
    for c in codes:
        duty = sum(c) / CHIPS
        assert 0.6 <= duty <= 0.82
        loop = "".join(map(str, c * 2))
        assert "0" * (MAX_DARK + 1) not in loop        # nigdy nie gaśnie na dłużej niż MAX_DARK chip
    # dowolne dwa ID różnią się w >= MIN_DIST chipach przy każdym przesunięciu
    arr = np.array(codes)
    rots = np.array([[_rot(list(c), r) for r in range(CHIPS)] for c in codes])   # K x R x N
    for i in range(N_IDS):
        d = (rots[i][None, :, :] != arr[:, None, :]).sum(axis=2).min(axis=1)
        d[i] = CHIPS
        assert d.min() >= MIN_DIST, i


def test_firmware_codes_match_python():
    text = (ROOT / "firmware" / "led_beacon" / "codes.h").read_text()
    fw = [int(x, 16) for x in re.findall(r"0x([0-9A-F]+)UL", text)]
    assert fw == list(CODES)


def test_every_id_decodes_from_any_phase():
    wrong = missed = 0
    for i in range(N_IDS):
        _, conf, _ = feed(i, seed=i)
        missed += conf is None
        wrong += conf is not None and conf != i
    assert wrong == 0 and missed == 0


def test_fast_enough_at_30_and_60_fps():
    t30 = [feed(i, fps=30, hz=10, seed=i)[2] for i in range(0, N_IDS, 8)]
    t60 = [feed(i, fps=60, hz=20, seed=i)[2] for i in range(0, N_IDS, 8)]
    assert np.median(t30) < 4500 and np.median(t60) < 2500


def test_junk_never_confirms_an_id():
    rng = random.Random(9)
    for j in range(150):
        dec = TemplateDecoder(10)
        level, kind = 0, j % 3
        for f in range(300):
            t = f * 1000.0 / 30
            if kind == 0:
                s = rng.uniform(-1, 1)                           # szum
            elif kind == 1:
                if rng.random() < 0.15:                          # losowe mruganie (ręka, odblask)
                    level = 1 - level
                s = 2 * level - 1 + rng.gauss(0, 0.2)
            else:
                s = np.sin(t / 150.0) + rng.gauss(0, 0.3)        # migotanie lampy
            dec.push(t, max(-1.0, min(1.0, s)))
            assert dec.confirmed_id is None, (j, kind)


def test_end_to_end_video():
    res = run(fps=30, chip_hz=10, seconds=10, ids=[3, 17, 60, 120], seed=5, verbose=False)
    assert res["found"] == {3, 17, 60, 120} and not res["wrong"]


def test_hungarian_is_optimal():
    import itertools
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = (int(v) for v in rng.integers(1, 6, size=2))
        c = rng.random((n, m)) * 10
        pairs = hungarian(c)
        assert len(pairs) == min(n, m)
        assert len({r for r, _ in pairs}) == len(pairs) == len({k for _, k in pairs})
        got = sum(c[r, k] for r, k in pairs)
        if n <= m:
            best = min(sum(c[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        else:
            best = min(sum(c[p[j], j] for j in range(m)) for p in itertools.permutations(range(n), m))
        assert abs(got - best) < 1e-9


def test_kalman_follows_constant_velocity_through_gaps():
    kf = KalmanCV(100.0, 50.0, pos_std=3.0, accel=0.01)
    vx, vy, dt = 0.5, -0.2, 16.7                       # px/ms, 60 fps
    for k in range(1, 60):
        kf.predict(dt)
        if k % 4:                                       # co 4. klatka dioda zgaszona - bez pomiaru
            kf.correct(100.0 + vx * k * dt, 50.0 + vy * k * dt, 2.0)
    assert abs(kf.x[2] - vx) < 0.05 and abs(kf.x[3] - vy) < 0.05
    for _ in range(3):                                   # 3 klatki bez pomiaru: przewidywanie
        kf.predict(dt)
    assert abs(kf.x[0] - (100.0 + vx * 62 * dt)) < 2.0
