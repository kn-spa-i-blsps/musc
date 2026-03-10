from typing import TypedDict, List
from detekcja.kalman_tracker import KalmanTracker

class Record(TypedDict):
    timestamp: int
    state: bool
    x: int
    y: int

class LightSource:
    def __init__(self, SIZE, FREQUENCY, AUTH, DRIFT_COEF, x, y):
        self.SIZE = SIZE
        self.FREQUENCY = FREQUENCY
        self.AUTH = AUTH
        self.DRIFT_COEF = DRIFT_COEF
        
        self.x, self.y = x, y
        self.dx, self.dy = 0, 0

        self.records = []
        self.match_score = 0.0
        self.confidence = 0.5
        self.last_seen_ts = 0

        self.kf = KalmanTracker(x,y)

    def add_record(self, record):
        MOVEMENT_THRESHOLD = 20 # Piksele: poniżej tej wartości ignorujemy ruch
        
        if record['state']:
            # 1. Obliczamy surowe przesunięcie
            raw_dx = record['x'] - self.x
            raw_dy = record['y'] - self.y
            
            # 2. STREFA NIEOCZUŁOŚCI (Deadzone)
            # Jeśli ruch jest bardzo mały, udajemy że go nie ma (dx=0, dy=0)
            if abs(raw_dx) < MOVEMENT_THRESHOLD: raw_dx = 0
            if abs(raw_dy) < MOVEMENT_THRESHOLD: raw_dy = 0
            
            # 3. Wygładzanie (DRIFT_COEF)
            # Ustaw go w main.py na ok. 0.4 - 0.6 dla responsywności
            self.dx = raw_dx * self.DRIFT_COEF
            self.dy = raw_dy * self.DRIFT_COEF
            
            # Aktualizacja pozycji
            self.x, self.y = record['x'], record['y']
            self.last_seen_ts = record['timestamp']
        else:
            # Gdy światło zgasło, kontynuujemy ruch
            # Dodaj lekkie hamowanie tylko tutaj, by "nie odlatywało" po zniknięciu

            self.x += self.dx
            self.y += self.dy

        self.records.insert(0, record)
        if len(self.records) > self.SIZE:
            self.records.pop()
        
        self.update_analysis()

    def get_quantized_bits(self) -> List[int]:
        if len(self.records) < 5:
            return []

        bit_dur = 1000 / self.FREQUENCY
        end_ts = self.records[0]['timestamp']
        start_ts = self.records[-1]['timestamp']

        # Wyznaczamy granice okienek czasowych
        first_boundary = start_ts + (bit_dur - (start_ts % bit_dur))
        
        # Odrzucamy fragmenty krótsze niż 50% bita na krawędziach
        actual_start = first_boundary if (first_boundary - start_ts) < (bit_dur * 0.5) else (first_boundary - bit_dur)
        
        bits = []
        current_t = actual_start
        
        while current_t + bit_dur <= end_ts:
            window_end = current_t + bit_dur
            # Głosowanie większościowe wewnątrz okienka bita
            states = [r['state'] for r in self.records if current_t <= r['timestamp'] < window_end]
            
            if states:
                bit_val = 1 if sum(states) > len(states) / 2 else 0
                bits.append(bit_val)
            current_t = window_end
            
        return bits

    def update_analysis(self):
        if len(self.records) < 10:
            return

        bit_dur = 1000 / self.FREQUENCY
        total_cycle = bit_dur * len(self.AUTH)
        best_m = 0

        # Skanowanie fazy co 5ms (Cross-correlation)
        for p in range(0, int(total_cycle), 5):
            matches = 0
            for r in self.records:
                t_rel = (r['timestamp'] + p) % total_cycle
                idx = min(int(t_rel / bit_dur), len(self.AUTH) - 1)
                if r['state'] == self.AUTH[idx]:
                    matches += 1
            
            score = matches / len(self.records)
            if score > best_m:
                best_m = score

        self.match_score = best_m
        # Płynna aktualizacja pewności
        self.confidence = self.confidence * 0.8 + (self.match_score * 0.2)