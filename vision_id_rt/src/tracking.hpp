#pragma once
#include <utility>
#include <vector>

namespace ledrx {

// Constant-velocity Kalman filter. State [x, y, vx, vy] in px and px/ms; process noise is a random
// acceleration with std `accel` px/ms^2.
class KalmanCV {
public:
    KalmanCV(double x, double y, double pos_std = 3.0, double vel_std = 0.3, double accel = 0.01);

    void predict(double dt);
    void correct(double zx, double zy, double meas_std);
    void shift(double dx, double dy);
    void stop();
    void clamp_speed(double vmax);
    double pos_std() const;

    double x[4];
    double P[4][4];
    double accel;
};

// Minimum-cost assignment for an n x m matrix (row-major). Returns min(n, m) (row, col) pairs sorted by row.
std::vector<std::pair<int, int>> hungarian(const std::vector<double>& cost, int n, int m);

}  // namespace ledrx
