#pragma once
#include <deque>
#include <limits>
#include <utility>
#include <vector>

namespace ledrx {

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
constexpr int NO_ID = -1;

class TemplateDecoder {
public:
    TemplateDecoder(double chip_hz, double min_contrast);

    int push(double ts_ms, double value);

    double chip_ms, frame_ms;
    int last_id = NO_ID;
    int confirmed_id = NO_ID;
    double confirmed_ts = NEG_INF;
    double last_ok_ts = NEG_INF;
    double last_score = 0.0, last_margin = 0.0, coverage = 0.0;
    int frames_ok = 0, frames_bad = 0;

private:
    bool id_scores(double now, float* per_id);
    bool normalize();
    bool looks_like_beacon() const;
    int evaluate(double now);

    double min_contrast_;
    std::deque<std::pair<double, float>> samples_;
    double cand_since_ = NEG_INF;
    int n_since_eval_ = 0;
    std::vector<double> t_;
    std::vector<float> v_, s_, sorted_;
};

}  // namespace ledrx
