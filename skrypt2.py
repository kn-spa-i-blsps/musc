import cv2
import numpy as np
import time

# --- KONFIGURACJA ---
THRESHOLD_VAL = 250   # Czułość maski
MAX_DISTANCE = 200    # Jak daleko punkt może się przesunąć między klatkami
MIN_LS_AREA = 10      # Minimalna wielkość punktu
NOT_SEEN_TIME = 0.5   # Po jakim czasie (s) usuwamy nieaktywne źródło
TARGET_FREQ = 15.0 / 4.0    # Szukana częstotliwość w Hz
TOLERANCE = 1.0       # Tolerancja błędu (+/- 2 Hz)

class LightSource:
    def __init__(self, pos, id):
        self.id = id
        self.pos = pos
        self.last_seen = time.time()
        self.current_state = 0
        self.last_state = 0
        self.on_timestamps = [] 
        self.detected_freq = 0.0

    def update_status(self, is_on, new_pos=None):
        now = time.time()
        if new_pos:
            self.pos = new_pos
            self.last_seen = now
            
        self.current_state = 1 if is_on else 0
        
        # Wykrywanie zbocza narastającego (moment zapalenia diody)
        if self.current_state == 1 and self.last_state == 0:
            self.on_timestamps.append(now)
            
            # Trzymamy ostatnie 10 mignięć
            if len(self.on_timestamps) > 10:
                self.on_timestamps.pop(0)
            
            # Liczymy Hz jeśli mamy min. 3 próbki
            if len(self.on_timestamps) >= 3:
                intervals = np.diff(self.on_timestamps)
                avg_interval = np.mean(intervals)
                if avg_interval > 0:
                    self.detected_freq = 1.0 / avg_interval

        self.last_state = self.current_state

    def is_target_freq(self, target, tolerance):
        return abs(self.detected_freq - target) <= tolerance

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
# cap.set(cv2.CAP_PROP_EXPOSURE, -8)
cap.set(cv2.CAP_PROP_EXPOSURE, -10)

sources = []
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Przetwarzanie obrazu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD_VAL, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1) 
    thresh = cv2.dilate(thresh, kernel, iterations=4) 

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_in_frame = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_LS_AREA:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > 0.5:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    found_in_frame.append((cX, cY))

    # 2. Logika dopasowania (Tracking & Hz)
    used_points_indices = set()
    
    for s in sources:
        matched = False
        best_dist = MAX_DISTANCE
        best_idx = -1
        
        for i, pt in enumerate(found_in_frame):
            if i not in used_points_indices:
                dist = np.linalg.norm(np.array(pt) - np.array(s.pos))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    matched = True
        
        if matched:
            s.update_status(is_on=True, new_pos=found_in_frame[best_idx])
            used_points_indices.add(best_idx)
        else:
            s.update_status(is_on=False)

    # Dodawanie nowych źródeł z punktów, które nie zostały dopasowane
    for i, pt in enumerate(found_in_frame):
        if i not in used_points_indices:
            sources.append(LightSource(pt, next_id))
            next_id += 1

    # Usuwanie starych źródeł
    sources = [s for s in sources if time.time() - s.last_seen < NOT_SEEN_TIME]

    # 3. Wizualizacja
    for s in sources:
        is_match = s.is_target_freq(TARGET_FREQ, TOLERANCE)
        color = (0, 255, 0) if is_match else (0, 0, 255)
        
        cv2.drawMarker(frame, s.pos, color, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"ID:{s.id} {s.detected_freq:.1f} Hz", (s.pos[0]+15, s.pos[1]+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if is_match:
            cv2.putText(frame, "MATCH 15Hz!", (s.pos[0]-40, s.pos[1]-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Maska", thresh)
    cv2.imshow("VLC Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
