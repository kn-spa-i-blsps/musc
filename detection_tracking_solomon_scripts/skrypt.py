import cv2
import numpy as np
import time

# --- KONFIGURACJA ---
THRESHOLD_VAL = 250   # Podnieś powyżej 240, jeśli masz "duchy" w masce
MAX_DISTANCE = 100     
TARGET_SEQ = "10"
BUFFER_SIZE = 12
SAMPLING_RATE = 0.1
MIN_LS_AREA = 10
NOT_SEEN_TIME = 0.2 # in seconds

class LightSource:
    def __init__(self, pos, id):
        self.id = id
        self.pos = pos
        # Teraz przechowujemy (czas, bit)
        self.bit_history = [] 
        self.last_seen = time.time()
        self.last_sample_time = time.time()
        self.current_state = 1

    def update(self, new_pos):
        self.pos = new_pos
        self.last_seen = time.time()
        self.current_state = 1

    def sample_bit(self):
        now = time.time()
        # Próbkowanie z twardym czasem
        if now - self.last_sample_time >= SAMPLING_RATE:
            # Zapisujemy parę: (moment_czasowy, stan_0_lub_1)
            self.bit_history.append((now, self.current_state))
            
            if len(self.bit_history) > BUFFER_SIZE:
                self.bit_history.pop(0)
            
            self.last_sample_time = now
            self.current_state = 0 

    def get_bits_str(self):
        # Pobieramy tylko bity z par (czas, bit)
        return "".join(map(str, [bit for _, bit in self.bit_history]))

    def get_last_intervals(self):
        # Funkcja pomocnicza: pokazuje odstępy między próbkami w ms
        if len(self.bit_history) < 2: return []
        intervals = []
        for i in range(1, len(self.bit_history)):
            t_diff = self.bit_history[i][0] - self.bit_history[i-1][0]
            intervals.append(int(t_diff * 1000)) # wynik w milisekundach
        return intervals
    def check_frequency_pattern(self, pattern="10011", freq_hz=15):
        if len(self.bit_history) < 10: return False
        
        bit_duration = 1.0 / freq_hz  # dla 15Hz to ~0.066s
        now = time.time()
        detected_pattern = ""
        
        # Sprawdzamy stan źródła w punktach wstecz: 0ms, 66ms, 133ms, 200ms, 266ms
        for i in range(len(pattern)):
            # Celujemy w środek trwania bitu, żeby uniknąć błędów na krawędziach
            target_time = now - (i * bit_duration) - (bit_duration / 2)
            
            # Szukamy w historii bitu, który był najbliżej tego czasu
            closest_bit = None
            min_delta = float('inf')
            
            for ts, bit in reversed(self.bit_history):
                delta = abs(ts - target_time)
                if delta < min_delta:
                    min_delta = delta
                    closest_bit = bit
                if ts < target_time - 0.2: # Nie szukaj za daleko w przeszłości
                    break
            
            if closest_bit is not None:
                detected_pattern = str(closest_bit) + detected_pattern
            else:
                return False

        return detected_pattern == pattern

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 

# 2. Ustaw ręczną wartość ekspozycji
# Uwaga: Wartości mogą być bardzo różne, np. od -1 do -13 (logarytmiczne) 
# albo od 1 do 1000 (milisekundy). Zacznij od małych wartości.
cap.set(cv2.CAP_PROP_EXPOSURE, -8)
sources = []
next_id = 0

# Osobne okno na maskę ułatwia debugowanie
# cv2.namedWindow("Maska", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Przetwarzanie obrazu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD_VAL, 255, cv2.THRESH_BINARY)
    
    # Usuwanie szumów (małych kropek)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1) 
    # Pogrubienie istotnych punktów
    thresh = cv2.dilate(thresh, kernel, iterations=4) 

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_in_frame = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_LS_AREA: # Ignoruj bardzo małe punkty
            # SPRAWDZANIE KOŁOWOŚCI (Filtracja fałszywych świateł)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # Prawdziwe światło ma circularity bliskie 1.0 (np. > 0.5)
            # Podłużne odbicia mają circularity bliskie 0.0
            if circularity > 0.5:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    found_in_frame.append((cX, cY))

    # Logika dopasowania do źródeł
    used_points = set()
    for s in sources:
        matched = False
        for i, pt in enumerate(found_in_frame):
            if i not in used_points:
                dist = np.linalg.norm(np.array(pt) - np.array(s.pos))
                if dist < MAX_DISTANCE:
                    s.update(pt)
                    used_points.add(i)
                    matched = True
                    break
        if not matched:
            s.current_state = 0
        s.sample_bit()

    for i, pt in enumerate(found_in_frame):
        if i not in used_points:
            sources.append(LightSource(pt, next_id))
            next_id += 1

    # Obiekty znikają po 1 sekundzie braku sygnału
    sources = [s for s in sources if time.time() - s.last_seen < NOT_SEEN_TIME]

    # --- WIZUALIZACJA ---
    for s in sources:
        color = (0, 255, 0) if s.current_state == 1 else (0, 0, 255)
        cv2.drawMarker(frame, s.pos, color, cv2.MARKER_CROSS, 20, 2)
        
        bits = s.get_bits_str()
        cv2.putText(frame, f"ID:{s.id} [{bits}]", (s.pos[0]+15, s.pos[1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Wewnątrz pętli: for s in sources:
        if s.check_frequency_pattern(pattern="10011", freq_hz=15):
            cv2.putText(frame, "MATCH 15Hz!", (s.pos[0]-40, s.pos[1]-60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.circle(frame, s.pos, 40, (0, 255, 255), 3)

    # Wyświetlanie maski i obrazu głównego
    cv2.imshow("Maska", thresh)
    cv2.imshow("VLC Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
