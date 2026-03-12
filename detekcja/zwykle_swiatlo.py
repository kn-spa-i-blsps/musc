from kalman_tracker import KalmanTracker


class Light:
    def __init__(self, ID, TS, x, y):
        self.ID = ID # trzyma swoje id
        self.last_seen_ts = TS #uplyw czau od kiedy był ostatnio widziany
        self.x, self.y = x, y #wsporzedne
        self.dx, self.dy = 0, 0 # predkosc (chyba niepotrzebne)
        self.kalman = KalmanTracker(x, y) #predyckja gdzie będzie