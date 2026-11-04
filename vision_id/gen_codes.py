"""
Generuje tablicę kodów diod (codebook.py dla Pythona i firmware/led_beacon/codes.h
dla Arduino). Uruchamiać tylko przy zmianie parametrów - wynik jest w repo.

    uv run gen_codes.py

Kod to cykliczny ciąg CHIPS chipów (1 = dioda świeci), nadawany w kółko:
  * dioda nigdy nie gaśnie na dłużej niż MAX_DARK chip -> tor jej nie gubi
  * świeci DUTY_MIN..DUTY_MAX czasu -> zawsze widać oba poziomy (próg adaptacyjny)
  * dowolne dwa RÓŻNE ID różnią się w >= MIN_DIST chipach przy KAŻDYM
    przesunięciu w czasie -> odbiornik nie musi znać fazy ani mieć
    synchronizacji, a kilka złych chipów nie zamienia jednego ID w inne
Wybór zachłanny w ustalonej kolejności -> deterministyczny.
"""

from pathlib import Path

import numpy as np

CHIPS = 26
N_IDS = 128
MIN_DIST = 4
MAX_DARK = 1
DUTY_MIN, DUTY_MAX = 0.62, 0.80

_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    return _POP[x & 0xFFFF] + _POP[(x >> 16) & 0xFFFF]


def _rotations(x: int) -> list:
    m = (1 << CHIPS) - 1
    return [((x << r) | (x >> (CHIPS - r))) & m for r in range(CHIPS)]


def _ok(x: int) -> bool:
    ones = bin(x).count("1")
    if not DUTY_MIN <= ones / CHIPS <= DUTY_MAX:
        return False
    run = 0
    for i in range(2 * CHIPS):                 # dwa obiegi = runy przez "szew" też
        if (x >> (i % CHIPS)) & 1:
            run = 0
        else:
            run += 1
            if run > MAX_DARK:
                return False
    return True


def generate() -> list:
    cands = {}
    for x in range(1 << CHIPS):
        if not _ok(x):
            continue
        c = min(_rotations(x))                 # jeden reprezentant na klasę przesunięć
        if c not in cands:
            cands[c] = abs(bin(c).count("1") / CHIPS - 0.72)
    order = sorted(cands, key=lambda c: (cands[c], c))
    rots = {c: np.array(_rotations(c), dtype=np.int64) for c in order}
    chosen = []
    for c in order:
        if chosen:
            # kod c w swoim ułożeniu vs KAŻDE przesunięcie każdego wybranego kodu
            other = np.stack([rots[k] for k in chosen])
            if _popcount(np.int64(c) ^ other).min() < MIN_DIST:
                continue
        chosen.append(c)
        if len(chosen) == N_IDS:
            return chosen
    raise SystemExit(f"znaleziono tylko {len(chosen)} kodów - zmień parametry")


def main() -> None:
    codes = generate()
    here = Path(__file__).parent
    rows = "\n".join(f"    0x{c:07X},  # {i:3d}: {c:0{CHIPS}b}" for i, c in enumerate(codes))
    (here / "codebook.py").write_text(
        '"""Wygenerowane przez gen_codes.py - nie edytować ręcznie."""\n\n'
        f"CHIPS = {CHIPS}\nMIN_DIST = {MIN_DIST}\nMAX_DARK = {MAX_DARK}\n\n"
        f"# bit {CHIPS - 1} (najstarszy) = pierwszy nadawany chip\nCODES = [\n{rows}\n]\n")
    hrows = "\n".join(f"  0x{c:07X}UL,  // {i:3d}" for i, c in enumerate(codes))
    (here / "firmware" / "led_beacon" / "codes.h").write_text(
        "// Wygenerowane przez vision_id/gen_codes.py - nie edytować ręcznie.\n"
        "#pragma once\n#include <avr/pgmspace.h>\n\n"
        f"const uint8_t CHIPS = {CHIPS};\nconst uint8_t N_IDS = {N_IDS};\n\n"
        f"// bit {CHIPS - 1} (najstarszy) = pierwszy nadawany chip, 1 = dioda świeci\n"
        f"const uint32_t CODES[N_IDS] PROGMEM = {{\n{hrows}\n}};\n")
    print(f"{len(codes)} kodów po {CHIPS} chipów, min. odległość {MIN_DIST}, max ciemność {MAX_DARK} chip")


if __name__ == "__main__":
    main()
