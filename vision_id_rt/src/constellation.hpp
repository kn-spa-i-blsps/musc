// Groups tracks that blink with the same code, frame by frame, before any ID is decoded.
// Same algorithm as constellation.py on the kalman-and-hungarian branch.
#pragma once
#include <cstdint>
#include <map>
#include <vector>

namespace ledrx {

class BlinkGrouper {
public:
    BlinkGrouper(double code_ms, double chip_ms, double min_events = 2.0, double radius_px = 150.0,
                 double block_level = 0.5, double near_weight = 0.5);

    // states[i]: +1 lit, -1 dark, 0 unknown. Returns track id -> group id.
    std::map<int, int> update(double ts, const std::vector<int>& tids, const std::vector<double>& xy,
                              const std::vector<int>& states);
    // External evidence that two tracks are different transmitters.
    void block(int tid_a, int tid_b);

    double radius_px;

private:
    void reindex(const std::vector<int>& tids);
    std::map<int, int> cluster(const std::vector<int>& tids, const std::vector<char>& edge,
                               const std::vector<char>& blocked);
    double& at(std::vector<double>& m, int i, int j) { return m[static_cast<size_t>(i) * n_ + j]; }

    double code_ms_, chip_ms_, min_events_, block_level_, near_weight_;
    int n_ = 0;
    std::vector<int> tids_;
    std::vector<int8_t> last_, ev_prev_;
    std::vector<int> run_;
    std::vector<double> sync_, strikes_, ok_ms_;
    std::map<int, int> gid_;
    int next_gid_ = 1;
    bool have_ts_ = false;
    double last_ts_ = 0.0, frame_dt_ = 33.3;
};

}  // namespace ledrx
