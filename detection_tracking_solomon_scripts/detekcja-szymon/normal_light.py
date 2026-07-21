from typing import List, Optional

from blink_protocol import BlinkDecoder
from kalman_tracker import KalmanTracker


class Light:
    """
    Jedno śledzone źródło światła.

    Pozycją i prędkością zarządza wyłącznie filtr Kalmana (aktualizowany
    w pętli głównej). Ta klasa dokłada do tego dekoder mrugania, który
    z sekwencji widoczny/niewidoczny odczytuje ID drona.
    """

    def __init__(self, ID: int, TS: int, x: float, y: float):
        self.ID = ID                    # wewnętrzny numer trackera (nie ID drona!)
        self.last_seen_ts = TS
        self.x, self.y = float(x), float(y)
        self.dx, self.dy = 0.0, 0.0
        self.kalman = KalmanTracker(x, y)
        self.decoder = BlinkDecoder()

    @property
    def drone_id(self) -> Optional[int]:
        """Zdekodowane i potwierdzone ID drona albo None."""
        return self.decoder.drone_id

    @property
    def confidence(self) -> float:
        """Pewność identyfikacji (0..1) na podstawie liczby zgodnych ramek."""
        return self.decoder.confidence

    def add_record(self, record: dict) -> None:
        """Rejestruje stan z jednej klatki: {'timestamp': ms, 'state': bool}."""
        if record['state']:
            self.last_seen_ts = record['timestamp']
        self.decoder.add_sample(record['timestamp'], record['state'])

    def get_quantized_bits(self) -> List[int]:
        """Ostatni zdekwantyzowany strumień bitów (do podglądu na ekranie)."""
        return self.decoder.last_bits
