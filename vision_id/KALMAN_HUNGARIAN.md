# Optical flow → Kalman filter + Hungarian algorithm

Branch `kalman-and-hungarian`. This note explains what changed in LED track
following (`detector.py`, new `tracking.py`), why it changed, and which
alternatives were considered and rejected.

## Summary

| | before | after |
|---|---|---|
| Where the track moves between frames | Lucas-Kanade optical flow on texture around the LED (optionally plus a least-squares constant-velocity model, `motion_model="cv"`) | Kalman filter prediction, constant-velocity model (`tracking.KalmanCV`) |
| Which blink peak belongs to which track | Greedy: older tracks pick first, each takes the strongest peak within reach | Global minimum-cost assignment, Hungarian algorithm (`tracking.hungarian`) |
| How the track is corrected | Fixed pull of 40% towards the peak (`snap_gain=0.4`), then overwritten by the exact source position | Kalman correction with the peak (coarse), then a second correction with the exact source position (fine) |
| Camera motion | Implicitly contained in the optical flow | Track states are shifted explicitly by the phase-correlation result |

What was **not** changed: blink-energy map, peak detection, spawning new
tracks, ID inheritance, `seek_source` / `core_lock` / `point_lock`, decoder,
deduplication, grouping into drones.

## Why replace optical flow at all

Apart from the direct request, the old approach had concrete weaknesses
already visible in the repo:

1. **Flow needs texture around the LED.** LK tracks corners in a ring
   8–45 px around the LED. On the target scene (LEDs against the sky, short
   exposure ~250 µs) there is almost nothing to track. The `"cv"` motion
   model was added earlier exactly to cover that case.
2. **Cost on the Pi.** Config comment: flow ~16 ms/frame on the Pi. The
   Pi station already ran with `flow=False` (`pi/decode_rec.py`), so on the
   real hardware the flow path was not being used anyway.
3. **Two overlapping motion models.** Flow and the least-squares velocity
   both moved the track, with a `flow_moved` flag deciding which one won.
   A Kalman filter gives one consistent state (position + velocity +
   uncertainty) that both predicts and smooths.
4. **Greedy assignment depends on order.** With two LEDs close to each other
   (several LEDs of one drone, or two drones crossing), the older track
   could take the peak that belonged to the younger one, and the younger
   one then lost its LED or jumped to a reflection. A global assignment
   avoids that.

## Key decisions

### 1. Own Hungarian implementation instead of `scipy.optimize.linear_sum_assignment`

- **Chosen:** ~40 lines in numpy (`tracking.hungarian`, O(n²m) with
  potentials, works on rectangular matrices).
- **Rejected: scipy.** It is not a dependency of the project, and it would
  have to be installed on the Raspberry Pi only for this. The matrices are
  tiny (≤ 24 tracks × ≤ 60 peaks), so a numpy version is fast enough.
- **Rejected: OpenCV.** It has no public Hungarian solver in its Python API.
- **Safeguard:** `test_hungarian_is_optimal` compares the result with brute
  force over all permutations on 200 random matrices of different shapes.

### 2. Own numpy Kalman filter instead of `cv2.KalmanFilter`

- **Chosen:** small class with explicit `predict(dt)` / `correct(z, std)`.
- **Why:** frame timing is not constant (camera jitter, dropped frames,
  30 vs 60 fps), so the transition matrix F and process noise Q must be
  rebuilt every frame from the real `dt`. With `cv2.KalmanFilter` that means
  overwriting its matrices every frame anyway. Different measurements also
  need different noise (decision 5), which is clumsier with the OpenCV object.
  The numpy version matches the style of the existing filter in
  `box_sim.py` (`IdHold`).
- Covariance update uses the Joseph form so P stays symmetric and positive
  definite after many corrections.
- Units are px and px/ms, driven by real timestamps, not frame indices.
  Parameters therefore mean the same thing at 30 and 60 fps.

### 3. Constant-velocity model instead of constant acceleration or IMM

- **Chosen:** state `[x, y, vx, vy]`, random acceleration as process noise
  (`kf_accel = 0.01 px/ms²`).
- **Why:** the prediction only has to bridge short gaps: the LED is off
  for about one chip of the code, i.e. a few frames. Over that horizon
  motion is close to a straight line. The previous `"cv"` model (inspired by
  UVDAR) relied on the same assumption and worked on the Pi recordings.
- **Rejected: constant acceleration.** It needs more measurements to settle
  and overshoots on noisy peak positions.
- **Rejected: IMM** (several models switched automatically). It adds
  complexity that a short prediction horizon does not justify.

### 4. Gating and cost function

- **Cost** = (distance from prediction / gate radius)² − 0.3 × (peak
  energy / strongest peak). Pairs outside the gate get a huge cost, and are
  dropped after the assignment.
- **Why distance first:** the main goal is continuity, so the track should
  keep the same LED. **Why a small energy bonus:** the old greedy code
  always chose the strongest peak, which is what keeps tracks on the LED and
  off its weaker reflections. The bonus preserves part of that preference.
  In the 8-seed simulation, a bonus of 0 or 1.0 was marginally worse
  (13 vs 12 false IDs). That is weak evidence; the value is a setting
  (`assign_energy_w`).
- **Gate radius:** kept the old schedule, `snap_px + LED radius +
  search_grow × time since last seen`, capped at 60 px.
- **Rejected: a Mahalanobis gate that widens with Kalman uncertainty.**
  This was the first implementation (gate +2σ). In simulation it was worse:
  with a wide gate, a track whose LED is off takes a neighbouring LED's
  peak instead of staying unassigned. Results (8 seeds): 2σ → 12 false IDs,
  1σ → 11, 0σ → 10. The option remains as `kf_gate_sigma`, with default 0.

### 5. Two-stage measurement instead of a single one

- **Chosen:** each frame can apply two sequential Kalman corrections:
  1. the assigned blink-energy peak with std `kf_peak_px = 4 px`,
  2. if `seek_source` / `core_lock` / `point_lock` found the exact source,
     that position with std `kf_fine_px = 1 px`.
- **Why:** the energy map has ~1 s of memory and is blurred, so its peak is
  only a rough position (for a moving LED it smears into a trail). The core
  and point detectors find the light source itself, which is precise.
  Sequential corrections with independent measurements are standard Kalman
  practice and weight each source by its accuracy.
- **Rejected: peak only.** It loses the precise source position that the
  decoder window and drone position depend on.
- **Rejected: overwriting the state with the exact position.** That throws
  away the velocity estimate and the smoothing.
- **Tuning:** a looser fine measurement (`kf_fine_px = 4`) doubled false
  IDs (24 vs 12). 0.7 and 1.0 were best; 1.0 had fewer false IDs over
  20 seeds (18 vs 21). A tighter peak std (2.5) or looser one (6–8) was
  worse.

### 6. Coasting and speed limit

- After `kf_coast_ms = 250 ms` without seeing the LED, the velocity is set
  to zero: the track stops and searches in place, as the old `cv_max_gap_ms`
  did. Otherwise a lost track would drift away across the image.
  Shortening it to 80 ms did not help.
- Speed is capped at `max_px_per_ms = 2` (same as before). Lowering it to
  0.5 made things clearly worse (18 vs 12 false IDs): it holds the prediction
  back on fast motion.

### 7. Camera motion compensation moved to the tracks

Optical flow measured the LED's motion in the image, which already
included camera motion. The Kalman filter predicts motion relative to the
scene only, so the phase-correlation shift is now added directly to every
track's state. Without this, a camera pan would look like every LED
jumping.

### 8. Removing the old code instead of keeping it as an option

- **Chosen:** deleted LK flow, the least-squares velocity, the greedy snap
  and their settings (`flow`, `flow_ring`, `flow_points`, `motion_model`,
  `cv_window_ms`, `snap_gain`). One motion path is easier to understand and
  tune than three overlapping ones.
- **Cost of this choice:**
  - Saved configs or `bench.py --config` strings that set these fields will
    now fail.
  - The `pi_niebo` preset and `pi/decode_rec.py` were updated.
  - An A/B comparison with the old tracker on real recordings now needs the
    old commit (`git show main:vision_id/detector.py`), not a flag.

## Evaluation

Only the simulator was available (`simulate.py`: 4 drones × 4 LEDs, random
phase, clock drift, frame jitter, motion, occlusion, static lamp). Each
setting was run at 30 fps / 10 Hz and at 60 fps / 20 Hz, with 20 seeds
each (160 IDs in total).

| | IDs found | false IDs | mean time to first ID |
|---|---|---|---|
| old (optical flow + greedy) | 158 / 160 | 34 | 2.11 s |
| new, first version (gate +2σ, `kf_fine_px=1.5`)* | 63 / 64 | 12 | 2.28 s |
| new, final (`kf_gate_sigma=0`, `kf_fine_px=1.0`) | 157 / 160 | 18 | 2.18 s |

\* measured on 8 seeds only (64 IDs); on the same 8 seeds the old version had
64 / 64 and 9 false IDs. So the untuned Kalman version was slightly worse, and
the gating and measurement-noise decisions above turned it into an
improvement.

## Limitations and what to verify

- **The simulator has no texture around the LEDs** (flat background +
  noise), so optical flow did almost nothing there. The comparison mostly
  measures *greedy snap without prediction* against *Kalman + Hungarian*.
  It does not show what flow contributed on textured real scenes (a hand or
  PCB around the LED). That needs real recordings.
- False-ID counts are small numbers and noisy between seeds. Differences of
  1–3 in an 8-seed run are not meaningful; that is why the final comparison
  used 20 seeds.
- Per-frame time on the Raspberry Pi was not measured. It is expected to be
  lower than with flow (no `goodFeaturesToTrack` / LK, only 4×4 matrix
  operations per track), but this is unverified.
- Next step: on the Pi run `uv run eval_rec.py NAGRANIE --id 5 --preset pi_niebo`
  on the 2026-11-03 recordings, and compare with the README table (especially
  "hand in motion: 88% of frames with ID").

## Files

- `tracking.py`: new; `KalmanCV`, `hungarian`.
- `detector.py`: prediction, assignment, corrections, camera shift; flow
  code and settings removed; new settings `kf_*`, `max_px_per_ms`,
  `assign_energy_w`.
- `pi/decode_rec.py`: `flow=False` removed from the station config.
- `README.md`: file list, Pi copy list (`tracking.py` must be copied too),
  preset description.
- `tests/test_vision_id.py`: Hungarian optimality test, Kalman
  test (tracks through frames where the LED is off).
