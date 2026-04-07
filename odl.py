import cv2
import numpy as np

# ==========================================================
# KONFIGURACJA KAMERY (SKALIBROWANA DLA REDMI 13 - ZDJĘCIE 4:3)
# ==========================================================
import math

Szerokosc = 3000
Wysokosc = 4000

CX = Szerokosc / 2.0  
CY = Wysokosc / 2.0   

FX = 2757.0 
FY = 2757.0

MACIERZ_K = np.array([
    [FX,  0, CX],
    [ 0, FY, CY],
    [ 0,  0,  1]
], dtype=np.float32)

DYSTORSJA = np.array([0.0, 0.0, 0, 0, 0], dtype=np.float32)

def oblicz_pozycje_drona(punkty_2d, bok_a):
    """
    punkty_2d: lista krotek (x, y) w pikselach
    bok_a: rzeczywisty bok kwadratu w metrach (np. 0.21)
    Zakładana kolejność punktów: LD, LG, PG, PD
    Zwraca odległość od obiektywu do ŚRODKA kwadratu.
    """
    n = len(punkty_2d)
    punkty_2d = np.array(punkty_2d, dtype=np.float32)

    # Wyliczamy połowę boku, aby wyśrodkować układ współrzędnych
    polowa = bok_a / 2.0

    # --- PRZYPADEK 4 PUNKTY (Pełny kwadrat) ---
    if n == 4:
        # Kolejność: LD, LG, PG, PD
        # Środek kwadratu to teraz (0, 0, 0)
        punkty_3d = np.array([
            [-polowa,  polowa, 0],   # LD (Lewy Dół)
            [-polowa, -polowa, 0],   # LG (Lewy Góra)
            [ polowa, -polowa, 0],   # PG (Prawy Góra)
            [ polowa,  polowa, 0]    # PD (Prawy Dół)
        ], dtype=np.float32)
        
        sukces, rvec, tvec = cv2.solvePnP(punkty_3d, punkty_2d, MACIERZ_K, DYSTORSJA)
        if sukces:
            # np.linalg.norm(tvec) liczy teraz dystans do punktu (0,0,0), czyli do środka
            return np.linalg.norm(tvec), "PnP (4 punkty)"

    # --- PRZYPADEK 3 PUNKTY (Trzy rogi) ---
    elif n == 3:
        # Zakładamy podanie pierwszych 3 punktów z sekwencji: LD, LG, PG
        punkty_3d = np.array([
            [-polowa,  polowa, 0],   # LD
            [-polowa, -polowa, 0],   # LG
            [ polowa, -polowa, 0]    # PG
        ], dtype=np.float32)
        
        sukces, rvecs, tvecs = cv2.solveP3P(punkty_3d, punkty_2d, MACIERZ_K, DYSTORSJA, flags=cv2.SOLVEPNP_P3P)
        
        if sukces:
            poprawne_odleglosci = []
            
            for tvec in tvecs:
                # Sprawdzamy, czy rozwiązanie jest przed kamerą (Z > 0)
                if tvec[2][0] > 0:
                    poprawne_odleglosci.append(np.linalg.norm(tvec))
            
            if poprawne_odleglosci:
                opcje_tekst = " lub ".join([f"{d:.2f}m" for d in poprawne_odleglosci])
                info = f"P3P (Możliwe opcje: {opcje_tekst})"
                return poprawne_odleglosci[0], info

    # --- PRZYPADEK 2 PUNKTY (Środek krawędzi) ---
    elif n == 2:
        p1, p2 = punkty_2d[0], punkty_2d[1]
        
        # Długość krawędzi na zdjęciu w pikselach
        p = np.linalg.norm(p1 - p2)
        f_avg = (FX + FY) / 2.0
        
        # Proporcja daje nam "głębokość" (oś Z), czyli odległość płaszczyzny krawędzi od kamery
        Z = (bok_a * f_avg) / p
        
        # Znajdujemy środek krawędzi na płaskim zdjęciu (piksele)
        u_c = (p1[0] + p2[0]) / 2.0
        v_c = (p1[1] + p2[1]) / 2.0
        
        # Przeliczamy środek w pikselach na rzeczywiste współrzędne X i Y w metrach, 
        # korzystając z wyliczonej głębokości Z i optyki kamery (CX, CY, FX, FY)
        X = (u_c - CX) * Z / FX
        Y = (v_c - CY) * Z / FY
        
        # Liczymy rzeczywistą odległość w linii prostej od obiektywu (0,0,0) do środka krawędzi
        odleglosc = np.sqrt(X**2 + Y**2 + Z**2)
        
        return odleglosc, "Proporcja (2 punkty - Środek krawędzi)"

    return None, "Błąd: Za mało punktów lub nieudane obliczenia"

# ==========================================================
# TESTY - MOŻESZ TU WPISAĆ SWOJE DANE ZE ZDJĘĆ
# ==========================================================
if __name__ == "__main__":
    BOK_KWADRATU = 0.21 # 21 cm


# punkty w odległości 1m 0,8m 1,2 m 1,4 m

    test_4pkt = [(665, 2745), (983, 2309), (1483, 2471), (1209, 2967)]
    d, info = oblicz_pozycje_drona(test_4pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")

    test_4pkt = [(923, 2361), (1505, 1901), (1975, 2477), (1377, 2971)]
    d, info = oblicz_pozycje_drona(test_4pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")

    test_4pkt = [(1295, 1892), (1722, 1835), (1853, 2130), (1373, 2195)]
    d, info = oblicz_pozycje_drona(test_4pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")

    test_4pkt = [(1609, 2060), (1580, 1817), (1964, 1836), (2029, 2076)]
    d, info = oblicz_pozycje_drona(test_4pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")



    test_3pkt = [(665, 2745), (983, 2309), (1483, 2471)]
    d, info = oblicz_pozycje_drona(test_3pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")
    
    test_3pkt = [(923, 2361), (1505, 1901), (1975, 2477)]
    d, info = oblicz_pozycje_drona(test_3pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")

    test_3pkt = [(1295, 1892), (1722, 1835), (1853, 2130)]
    d, info = oblicz_pozycje_drona(test_3pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")

    test_3pkt = [(1609, 2060), (1580, 1817), (1964, 1836)]
    d, info = oblicz_pozycje_drona(test_3pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")


    # Przykład 2: Widzimy tylko górną krawędź (2 punkty) - Środek krawędzi
    test_2pkt = [(923, 2361), (1377, 2971)]
    d, info = oblicz_pozycje_drona(test_2pkt, BOK_KWADRATU)
    print(f"{info:<30} | {d:.2f} m")