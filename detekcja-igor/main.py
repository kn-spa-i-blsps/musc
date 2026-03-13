import cv2
import numpy as np
import time
from light_source import LightSource
from typing import List

# --- PARAMETRY ---
MIN_RADIUS = 3      # Zmniejszone, by łapać mniejsze punkty
RADIUS = 12         # Rozmiar kółka na ekranie
MIN_DIST = 50       # Podstawowy dystans szukania
INTENSITY = 230     
THRESHOLD = 180     
MIN_CONFIDENCE = 0.4
AUTH = [True, False, True, False]
FREQ = 24           # Obniżone do 12Hz dla lepszej stabilności przy 30 FPS
SIZE = 60
DRIFT_COEF = 1    # Zmniejszone, by uniknąć "pływania" wektora
DETECTION_TIMEOUT = 1500 

def main():
    cap = cv2.VideoCapture(0)
    sources: List[LightSource] = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        ts = int((time.time() - start_time) * 1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_points = []
        for c in contours:
            (x, y), r = cv2.minEnclosingCircle(c)
            # Zwiększony zakres dla smugi przy szybkim ruchu
            if MIN_RADIUS < r < RADIUS * 2.5:
                if gray[int(y), int(x)] > INTENSITY:
                    detected_points.append({'x': int(x), 'y': int(y)})

        # --- LOGIKA TRACKINGU ---
        used_points = set()
        for s in sources:
            # 1. DYNAMICZNY DYSTANS: jeśli obiekt leci szybko, szukamy szerzej
            velocity = (s.dx**2 + s.dy**2)**0.5
            dynamic_min_dist = max(MIN_DIST, velocity * 2.5)
            
            ax, ay = s.x, s.y
            bx, by = s.x + s.dx, s.y + s.dy
            
            found_match = None
            current_min_dist = dynamic_min_dist
            match_idx = -1
            
            for i, p in enumerate(detected_points):
                if i in used_points: continue
                px, py = p['x'], p['y']
                
                # Odległość punktu od ścieżki AB (Kapsuła)
                dx, dy = bx - ax, by - ay
                mag_sq = dx*dx + dy*dy
                
                if mag_sq == 0:
                    dist = ((px - ax)**2 + (py - ay)**2)**0.5
                else:
                    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / mag_sq))
                    closest_x = ax + t * dx
                    closest_y = ay + t * dy
                    dist = ((px - closest_x)**2 + (py - closest_y)**2)**0.5

                if dist < current_min_dist:
                    current_min_dist = dist
                    found_match = p
                    match_idx = i

            if found_match:
                s.add_record({'timestamp': ts, 'state': True, 'x': found_match['x'], 'y': found_match['y']})
                s.last_seen_ts = ts
                used_points.add(match_idx)
            else:
                s.add_record({'timestamp': ts, 'state': False, 'x': int(s.x), 'y': int(s.y)})

        # --- DODAWANIE NOWYCH ---
        for i, p in enumerate(detected_points):
            if i not in used_points:
                new_s = LightSource(SIZE, FREQ, AUTH, DRIFT_COEF, p['x'], p['y'])
                new_s.last_seen_ts = ts
                sources.append(new_s)

        # --- 4. USUWANIE STARYCH (Dynamiczny Timeout) ---
        # Jeśli s.match_score jest niskie, usuwamy po 300ms (duchy nie zostają)
        sources = [s for s in sources if (ts - s.last_seen_ts) < (DETECTION_TIMEOUT if s.match_score > 0.4 else 300)]

        # --- 5. RYSOWANIE ---
        # --- 5. RYSOWANIE I DEBUG ---
        # Pomocniczo rysujemy wszystkie wykryte kontury (na biało)
        cv2.drawContours(frame, contours, -1, (200, 200, 200), 1)

        for s in sources:
            if s.confidence < 0.1: continue # Pokazujemy nawet te z małą pewnością do testów
            
            is_auth = s.match_score > 0.75 and s.confidence > MIN_CONFIDENCE
            color = (0, 255, 0) if is_auth else (0, 0, 255)
            
            cx, cy = int(s.x), int(s.y)
            bx, by = int(s.x + s.dx), int(s.y + s.dy) # Przewidywany koniec
            
            # --- WIZUALIZACJA OBSZARU SZUKANIA (KAPSUŁA) ---
            # Obliczamy dynamiczny dystans tak samo jak w logice trackingu
            velocity = (s.dx**2 + s.dy**2)**0.5
            d_dist = int(max(MIN_DIST, velocity * 2.5))
            
            # Rysujemy "korytarz" szukania na żółto (półprzezroczysty efekt linią)
            cv2.line(frame, (cx, cy), (bx, by), (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), d_dist, (0, 150, 150), 1) # Okrąg wokół starej pozycji
            cv2.circle(frame, (bx, by), d_dist, (0, 255, 255), 1) # Okrąg wokół nowej pozycji
            
            # Główne kółko źródła
            cv2.circle(frame, (cx, cy), RADIUS, color, 2)
            
            # Statystyki
            q_bits = s.get_quantized_bits()
            q_str = "".join(map(str, q_bits[-8:])) # Ostatnie 8 bitów
            cv2.putText(frame, f"ID:{id(s)%100} BITS:{q_str}", (cx - 40, cy - 60), 1, 0.7, (0, 255, 255), 1)
            cv2.putText(frame, f"M:{s.match_score:.2f} V:{velocity:.1f}", (cx - 40, cy - 40), 1, 0.7, color, 1)

        cv2.imshow("Optical Auth Receiver", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()