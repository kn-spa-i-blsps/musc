#include "tracking.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ledrx {

KalmanCV::KalmanCV(double px, double py, double pos_std, double vel_std, double accel_std)
    : x{px, py, 0.0, 0.0}, P{}, accel(accel_std) {
    P[0][0] = P[1][1] = pos_std * pos_std;
    P[2][2] = P[3][3] = vel_std * vel_std;
}

void KalmanCV::predict(double dt) {
    if (dt <= 0) return;
    double F[4][4] = {{1, 0, dt, 0}, {0, 1, 0, dt}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    double G[4][2] = {{dt * dt / 2, 0}, {0, dt * dt / 2}, {dt, 0}, {0, dt}};
    double nx[4] = {};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) nx[i] += F[i][k] * x[k];
    double FP[4][4] = {};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) FP[i][j] += F[i][k] * P[k][j];
    double q = accel * accel;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += FP[i][k] * F[j][k];
            P[i][j] = s + q * (G[i][0] * G[j][0] + G[i][1] * G[j][1]);
        }
    std::copy(nx, nx + 4, x);
}

void KalmanCV::correct(double zx, double zy, double meas_std) {
    double r = meas_std * meas_std;
    double s00 = P[0][0] + r, s01 = P[0][1], s10 = P[1][0], s11 = P[1][1] + r;
    double det = s00 * s11 - s01 * s10;
    double i00 = s11 / det, i01 = -s01 / det, i10 = -s10 / det, i11 = s00 / det;
    double K[4][2];
    for (int i = 0; i < 4; ++i) {
        K[i][0] = P[i][0] * i00 + P[i][1] * i10;
        K[i][1] = P[i][0] * i01 + P[i][1] * i11;
    }
    double y0 = zx - x[0], y1 = zy - x[1];
    for (int i = 0; i < 4; ++i) x[i] += K[i][0] * y0 + K[i][1] * y1;

    // Joseph form keeps P symmetric and positive definite.
    double A[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) A[i][j] = (i == j) - (j < 2 ? K[i][j] : 0.0);
    double AP[4][4] = {};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) AP[i][j] += A[i][k] * P[k][j];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += AP[i][k] * A[j][k];
            P[i][j] = s + r * (K[i][0] * K[j][0] + K[i][1] * K[j][1]);
        }
}

void KalmanCV::shift(double dx, double dy) {
    x[0] += dx;
    x[1] += dy;
}

void KalmanCV::stop() { x[2] = x[3] = 0.0; }

void KalmanCV::clamp_speed(double vmax) {
    double sp = std::hypot(x[2], x[3]);
    if (sp > vmax) {
        x[2] *= vmax / sp;
        x[3] *= vmax / sp;
    }
}

double KalmanCV::pos_std() const { return std::sqrt(std::max(P[0][0], P[1][1])); }

// Shortest augmenting path with potentials, O(n^2 m). Same steps and tie-breaking as tracking.py on the
// kalman-and-hungarian branch, so both pick the same assignment when costs tie.
std::vector<std::pair<int, int>> hungarian(const std::vector<double>& cost_in, int n, int m) {
    if (n == 0 || m == 0) return {};
    const bool transposed = n > m;
    auto cost = [&](int i, int j) { return transposed ? cost_in[static_cast<size_t>(j) * m + i] : cost_in[static_cast<size_t>(i) * m + j]; };
    if (transposed) std::swap(n, m);
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0), minv(m + 1);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    std::vector<char> used(m + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), inf);
        std::fill(used.begin(), used.end(), 0);
        while (true) {
            used[j0] = 1;
            int i0 = p[j0];
            int j1 = -1;
            double delta = inf;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                double cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (j1 < 0 || minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if (p[j0] == 0) break;
        }
        while (j0) {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }
    std::vector<std::pair<int, int>> pairs;
    for (int j = 1; j <= m; ++j)
        if (p[j]) pairs.emplace_back(transposed ? j - 1 : p[j] - 1, transposed ? p[j] - 1 : j - 1);
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

}  // namespace ledrx
