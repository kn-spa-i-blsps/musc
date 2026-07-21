"""
Protokół optycznej identyfikacji dronów (OOK - On/Off Keying).

Ramka ma 30 bitów i przy 30 bit/s trwa dokładnie 1 sekundę:

    [ 10 bitów preambuły ][ 15 bitów ID ][ 5 bitów sumy kontrolnej ]

Kamera 60 fps daje ~2 próbki (klatki) na bit. Dekoder:
  1. zbiera próbki (timestamp_ms, dioda_świeci) dla jednego śledzonego światła,
  2. kwantyzuje je do bitów (głosowanie większościowe w okienku bita),
     próbując kilku przesunięć fazy, bo nadajnik nie jest zsynchronizowany,
  3. szuka preambuły w strumieniu bitów,
  4. weryfikuje sumę kontrolną,
  5. blokuje ID dopiero po LOCK_DECODES zgodnych ramkach (odporność na
     przypadkowe wystąpienie preambuły w danych).
"""

from collections import deque
from typing import List, Optional, Sequence, Tuple

FPS = 60.0
BIT_RATE = 30.0                     # bitów na sekundę
BIT_DUR_MS = 1000.0 / BIT_RATE      # ~33.33 ms

# Preambuła: naprzemienne 10101010 (łatwa synchronizacja) + "11" jako znacznik końca
PREAMBLE: Sequence[int] = (1, 0, 1, 0, 1, 0, 1, 0, 1, 1)
PREAMBLE_BITS = len(PREAMBLE)
ID_BITS = 15
CHECKSUM_BITS = 5
PACKET_BITS = PREAMBLE_BITS + ID_BITS + CHECKSUM_BITS   # 30
PACKET_DUR_MS = PACKET_BITS * BIT_DUR_MS                # 1000 ms

# Ile przesunięć fazy próbować przy kwantyzacji (krok = BIT_DUR_MS / PHASE_STEPS)
PHASE_STEPS = 8
# Ile ramek z tym samym ID musi się zgodzić, żeby uznać identyfikację za pewną
LOCK_DECODES = 2

_PREAMBLE_LIST = list(PREAMBLE)


def checksum5(drone_id: int) -> int:
    """Suma kontrolna: suma 5-bitowych grup ID modulo 32 (łatwa do policzenia na dronie)."""
    return ((drone_id & 0x1F) + ((drone_id >> 5) & 0x1F) + ((drone_id >> 10) & 0x1F)) & 0x1F


def int_to_bits(value: int, n_bits: int) -> List[int]:
    """Zamienia liczbę na listę bitów, MSB pierwszy."""
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


def bits_to_int(bits: Sequence[int]) -> int:
    """Zamienia listę bitów (MSB pierwszy) na liczbę."""
    value = 0
    for b in bits:
        value = (value << 1) | b
    return value


def encode_packet(drone_id: int) -> List[int]:
    """Buduje pełną 30-bitową ramkę dla danego ID (strona nadajnika / testy)."""
    if not 0 <= drone_id < (1 << ID_BITS):
        raise ValueError(f"ID drona musi być w zakresie 0..{(1 << ID_BITS) - 1}")
    return (_PREAMBLE_LIST
            + int_to_bits(drone_id, ID_BITS)
            + int_to_bits(checksum5(drone_id), CHECKSUM_BITS))


def is_safe_id(drone_id: int) -> bool:
    """
    Sprawdza, czy ID jest bezpieczne do przydzielenia dronowi.

    Przy ciągłym nadawaniu ramki w pętli niektóre ID (np. 21845 =
    101010101010101) tworzą w strumieniu "fałszywą preambułę": przesunięte
    okno 30 bitów też wygląda jak poprawna ramka i dekoduje się jako inne ID.
    Takich ID nie należy przydzielać dronom w roju.
    """
    frame = encode_packet(drone_id)
    stream = frame + frame  # dwie ramki pod rząd = wszystkie przesunięcia cykliczne
    for s in range(1, PACKET_BITS):
        window = stream[s:s + PACKET_BITS]
        if window[:PREAMBLE_BITS] != _PREAMBLE_LIST:
            continue
        fake_id = bits_to_int(window[PREAMBLE_BITS:PREAMBLE_BITS + ID_BITS])
        fake_chk = bits_to_int(window[PREAMBLE_BITS + ID_BITS:])
        if fake_chk == checksum5(fake_id):
            return False
    return True


class BlinkDecoder:
    """Dekoder strumienia mrugnięć jednego śledzonego źródła światła."""

    def __init__(self, buffer_packets: float = 2.5):
        # bufor na ~2.5 ramki próbek (przy 60 fps ~150 próbek)
        maxlen = int(buffer_packets * PACKET_DUR_MS * FPS / 1000.0) + 8
        self.samples: deque = deque(maxlen=maxlen)

        self.drone_id: Optional[int] = None    # zablokowane, potwierdzone ID
        self.confidence: float = 0.0
        self.last_bits: List[int] = []          # ostatnia kwantyzacja (do podglądu)

        self._candidate_id: Optional[int] = None
        self._streak: int = 0
        self._last_packet_ts: float = float("-inf")

    def add_sample(self, ts_ms: float, state: bool) -> Optional[int]:
        """
        Dodaje próbkę z jednej klatki (czy światło było widoczne) i próbuje
        zdekodować ramkę. Zwraca aktualnie potwierdzone ID drona lub None.
        """
        self.samples.append((float(ts_ms), bool(state)))

        result = self._try_decode()
        if result is not None:
            drone_id, packet_ts = result
            # ta sama ramka jest widoczna w buforze przez wiele klatek -
            # liczymy ją tylko raz (nowa ramka = start >= pół pakietu później)
            if packet_ts - self._last_packet_ts > PACKET_DUR_MS * 0.5:
                self._last_packet_ts = packet_ts
                if drone_id == self._candidate_id:
                    self._streak += 1
                else:
                    self._candidate_id = drone_id
                    self._streak = 1
                if self._streak >= LOCK_DECODES:
                    self.drone_id = drone_id
                self.confidence = min(1.0, self._streak / (LOCK_DECODES + 1))
        elif ts_ms - self._last_packet_ts > 3 * PACKET_DUR_MS:
            # dawno nie było poprawnej ramki - pewność powoli spada
            self.confidence *= 0.98
            if self.confidence < 0.05:
                self.drone_id = None
                self._candidate_id = None
                self._streak = 0

        return self.drone_id

    def _try_decode(self) -> Optional[Tuple[int, float]]:
        """Szuka najświeższej poprawnej ramki. Zwraca (id, czas_startu_ramki) lub None."""
        if len(self.samples) < int(PACKET_BITS * FPS / BIT_RATE):
            return None

        t0 = self.samples[0][0]
        for k in range(PHASE_STEPS):
            phase = k * BIT_DUR_MS / PHASE_STEPS
            bits, valid = self._quantize(t0 + phase)
            if len(bits) < PACKET_BITS:
                continue

            # skanujemy od końca, żeby znaleźć najnowszą ramkę
            for start in range(len(bits) - PACKET_BITS, -1, -1):
                if bits[start:start + PREAMBLE_BITS] != _PREAMBLE_LIST:
                    continue
                if not all(valid[start:start + PACKET_BITS]):
                    continue
                id_bits = bits[start + PREAMBLE_BITS:start + PREAMBLE_BITS + ID_BITS]
                chk_bits = bits[start + PREAMBLE_BITS + ID_BITS:start + PACKET_BITS]
                drone_id = bits_to_int(id_bits)
                if bits_to_int(chk_bits) == checksum5(drone_id):
                    self.last_bits = bits
                    return drone_id, t0 + phase + start * BIT_DUR_MS
        return None

    def _quantize(self, t_start: float) -> Tuple[List[int], List[bool]]:
        """
        Kwantyzuje próbki do bitów. Wartość bita bierzemy z próbki najbliższej
        ŚRODKA okienka - próbki trafiające w moment przełączania diody
        (krawędź bita) są niewiarygodne, a przy 2 próbkach na bit zwykłe
        głosowanie większościowe kończy się remisami.
        valid[i] mówi, czy w oknie była jakakolwiek próbka.
        """
        last_ts = self.samples[-1][0]
        n_bits = int((last_ts - t_start) // BIT_DUR_MS)
        if n_bits <= 0:
            return [], []

        half_bit = BIT_DUR_MS * 0.5
        best_dist = [float("inf")] * n_bits
        bits = [0] * n_bits
        valid = [False] * n_bits
        for ts, state in self.samples:
            rel = ts - t_start
            idx = int(rel // BIT_DUR_MS)
            if 0 <= idx < n_bits:
                dist = abs(rel - idx * BIT_DUR_MS - half_bit)
                if dist < best_dist[idx]:
                    best_dist[idx] = dist
                    bits[idx] = 1 if state else 0
                    valid[idx] = True
        return bits, valid


if __name__ == "__main__":
    # Prosty test: symulujemy kamerę 60 fps odbierającą ramki nadawane w pętli
    import random

    def simulate(drone_id: int, n_packets: int = 3, jitter_ms: float = 1.5,
                 error_rate: float = 0.0, t_offset: float = 123.4) -> Optional[int]:
        packet = encode_packet(drone_id)
        decoder = BlinkDecoder()
        frame_dur = 1000.0 / FPS
        n_frames = int(n_packets * PACKET_DUR_MS / frame_dur)
        for f in range(n_frames):
            ts = t_offset + f * frame_dur + random.uniform(-jitter_ms, jitter_ms)
            bit_idx = int(((ts - t_offset) % PACKET_DUR_MS) // BIT_DUR_MS)
            state = bool(packet[min(bit_idx, PACKET_BITS - 1)])
            if random.random() < error_rate:
                state = not state
            decoder.add_sample(ts, state)
        return decoder.drone_id

    random.seed(42)

    unsafe = [i for i in range(1 << ID_BITS) if not is_safe_id(i)]
    print(f"Niebezpieczne ID (falszywa preambula): {len(unsafe)}/{1 << ID_BITS}"
          f" -> przyklady: {unsafe[:8]}")

    ok = 0
    tests = 0
    for did in [0, 1, 1234, 21846, 32767]:
        assert is_safe_id(did), f"ID {did} nie jest bezpieczne - popraw test"
        for offset in [0.0, 57.3, 500.0]:
            tests += 1
            got = simulate(did, t_offset=offset)
            status = "OK" if got == did else f"BLAD (odczytano {got})"
            if got == did:
                ok += 1
            print(f"ID={did:5d} offset={offset:6.1f}ms -> {status}")

    # test odporności: uszkodzona suma kontrolna nie może przejść
    bad = encode_packet(999)
    bad[-1] ^= 1
    dec = BlinkDecoder()
    frame_dur = 1000.0 / FPS
    for f in range(int(2 * PACKET_DUR_MS / frame_dur)):
        ts = f * frame_dur
        bit_idx = int((ts % PACKET_DUR_MS) // BIT_DUR_MS)
        dec.add_sample(ts, bool(bad[min(bit_idx, PACKET_BITS - 1)]))
    assert dec.drone_id is None, "ramka z zla suma kontrolna zostala przyjeta!"
    print(f"\nPoprawne dekodowania: {ok}/{tests}, zla suma kontrolna odrzucona: TAK")
