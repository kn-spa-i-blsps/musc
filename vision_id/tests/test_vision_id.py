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


# ---------------------------------------------------------------- konstelacje
def _states(drone_id, phase_ms, t, hz=10.0, exposure_ms=16.0):
    """Stan diody widziany przez kamerę w chwili t: +1 / -1, a 0, gdy klatka trafiła na zbocze."""
    chips = encode_frame(drone_id)
    chip_ms = 1000.0 / hz
    on = sum(chips[int((t + phase_ms + k * exposure_ms / 6) // chip_ms) % CHIPS] for k in range(6)) / 6
    return 1 if on > 0.65 else -1 if on < 0.35 else 0


def test_constellations_group_by_blinking_before_any_id():
    from constellation import BlinkGrouper
    rng = random.Random(3)
    worst = 0.0
    for trial in range(30):
        ids = rng.sample(range(N_IDS), 3)
        phases = [rng.uniform(0, 2600) for _ in ids]
        owner = [0, 0, 0, 1, 1, 1, 2, 2]                      # 8 torów: 3 + 3 + 2 diody trzech dronów
        xy = np.array([[10.0 * i, 0.0] for i in range(len(owner))])   # przemieszane w obrazie - liczy się mruganie
        g = BlinkGrouper(code_ms=2600.0, chip_ms=100.0)
        done = None
        for f in range(240):
            t = f * 1000.0 / 30 + rng.uniform(-1.5, 1.5)
            st = [_states(ids[o], phases[o], t) for o in owner]
            st[rng.randrange(len(st))] = 0                    # co klatkę jeden tor bez pewnego stanu
            gid = g.update(t, list(range(len(owner))), xy, st)
            good = all((gid[a] == gid[b]) == (owner[a] == owner[b])
                       for a in range(len(owner)) for b in range(a))
            if good and done is None:
                done = t
            elif not good and t > 2600.0:
                raise AssertionError((trial, f, gid))         # po jednym obiegu kodu podział ma być stabilny
            elif not good:
                done = None
        assert done is not None, trial
        worst = max(worst, done)
    assert worst < 2600.0                                     # zawsze szybciej niż jeden obieg kodu


def test_constellation_keeps_group_number_and_respects_block():
    from constellation import BlinkGrouper
    g = BlinkGrouper(code_ms=2600.0, chip_ms=100.0)
    xy = np.zeros((3, 2))
    gids = []
    for f in range(90):
        t = f * 1000.0 / 30
        gid = g.update(t, [7, 8, 9], xy, [_states(5, 0.0, t)] * 3)
        gids.append(gid[7])
    assert gid[7] == gid[8] == gid[9] and len(set(gids[30:])) == 1
    g.block(8, 9)                                             # np. dekoder odczytał na nich dwa różne ID
    gid = g.update(3000.0, [7, 8, 9], xy, [1, 1, 1])
    assert gid[8] != gid[9]
    gid = g.update(3033.0, [7, 9], xy[:2], [1, 1])            # tor 8 zniknął - reszta trzyma numer grupy
    assert gid[7] == gids[-1]


def test_video_constellation_forms_before_id_is_read():
    res = run(fps=30, chip_hz=10, seconds=6, ids=[3, 17, 60, 120], seed=5, verbose=False)
    assert set(res["first_group"]) == {3, 17, 60, 120}
    for i, t_id in res["first_seen"].items():
        assert res["first_group"][i] < t_id


# ---------------------------------------------------------- telemetria FC
def test_msp_frames():
    from fc_telemetry import MSP_RAW_IMU, msp_frame, msp_parse
    assert msp_frame(MSP_RAW_IMU) == b"$M<\x00\x66\x66"
    payload = bytes(range(18))
    chk = 18 ^ MSP_RAW_IMU
    for b in payload:
        chk ^= b
    reply = b"$M>" + bytes([18, MSP_RAW_IMU]) + payload + bytes([chk])
    assert msp_parse(b"xx" + reply[:10]) is None              # niepełna ramka - czekamy na resztę
    assert msp_parse(b"xx" + reply + b"$M") == (MSP_RAW_IMU, payload, 2 + len(reply))
    assert msp_parse(reply[:-1] + b"\x00")[0] == -1           # zła suma kontrolna


def test_gyro_integration_and_camera_axes():
    from fc_telemetry import GyroIntegrator, body_to_cam
    gi = GyroIntegrator()
    for k in range(200):                                      # 200 Hz, yaw 0.5 rad/s, pitch rośnie liniowo
        t = 5.0 * k
        gi.add(t, (0.0, 0.001 * t, 0.5))
    rot = gi.rotation(101.3, 134.6)
    assert abs(rot[2] - 0.5 * 0.0333) < 1e-6
    assert abs(rot[1] - 0.001 * (134.6 ** 2 - 101.3 ** 2) / 2 / 1000) < 1e-6
    assert gi.rotation(990.0, 1100.0) is None                 # telemetria nie sięga tak daleko - nie zgadujemy
    assert np.allclose(body_to_cam((0.1, 0.2, 0.3)), (0.2, 0.3, 0.1))          # pitch -> x, yaw -> y, roll -> z
    assert np.allclose(body_to_cam((0.0, 0.0, 1.0), tilt_deg=90.0), (0.0, 0.0, -1.0), atol=1e-9)
    assert np.allclose(body_to_cam((0.1, 0.2, 0.3), signs=(1, -1, -1)), (-0.2, -0.3, 0.1))


def test_fc_calibration_recovers_signs_and_latency():
    from fc_telemetry import GyroIntegrator, fit_calibration
    gi = GyroIntegrator()
    yaw = lambda t: 0.05 * np.sin(t / 300.0)                  # kąt kamery [rad]
    pitch = lambda t: 0.03 * np.sin(t / 170.0 + 1.0)
    for k in range(2400):                                     # FC raportuje pitch i yaw z odwrotnym znakiem
        t = 2.5 * k
        gi.add(t, (0.0, -0.03 / 0.170 * np.cos(t / 170.0 + 1.0), -0.05 / 0.300 * np.cos(t / 300.0)))
    fx, lag = 750.0, 40.0                                     # obraz spóźniony o 40 ms względem żyroskopu
    ts = np.arange(200.0, 5800.0, 33.3)
    shifts = np.array([[np.nan, np.nan]] + [[-fx * (yaw(b - lag) - yaw(a - lag)), fx * (pitch(b - lag) - pitch(a - lag))]
                                            for a, b in zip(ts, ts[1:])])
    res = fit_calibration(ts, shifts, gi, fx, tilt_deg=0.0)
    assert res["signs"][1:] == (-1.0, -1.0) and res["latency_ms"] == 40.0 and res["corr"] > 0.99
    assert abs(res["scale"] - 1.0) < 0.02


def test_moving_camera_needs_compensation_and_gyro_provides_it():
    kw = dict(fps=30, chip_hz=10, seconds=8, ids=[3, 17, 60, 120], seed=5, verbose=False, cam_shake_deg=3.0)
    blind = run(motion_comp=False, **kw)
    gyro = run(motion_comp=False, gyro=True, **kw)
    assert gyro["found"] == {3, 17, 60, 120} and not gyro["wrong"]
    assert gyro["coverage"] > 3 * blind["coverage"]
