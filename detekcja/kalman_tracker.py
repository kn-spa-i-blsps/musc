import cv2
import numpy as np

class KalmanTracker:
    def __init__(self, x, y):
        self.kf = cv2.KalmanFilter(4, 2, 0)

        # Macierz przejścia: [x, y, dx, dy] -> [x+dx, y+dy, dx, dy]
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Macierz pomiaru – mierzymy tylko x i y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # Szum procesu – im mniejszy, tym gładszy tor ruchu
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        # Szum pomiaru – im większy, tym mniej ufamy detekcji
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Stan początkowy
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    def predict(self):
        predicted = self.kf.predict()
        return predicted[0, 0], predicted[1, 0]

    def update(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)

    def get_state(self):
        state = self.kf.statePost
        return state[0, 0], state[1, 0], state[2, 0], state[3, 0]