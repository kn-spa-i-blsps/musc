// Finds blinking LEDs, tracks them, decodes their IDs and groups LEDs with the same ID into drones.
// Coordinates are in input image pixels.
#pragma once
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <vector>

#include "decoder.hpp"

namespace ledrx {

struct Config {
    double chip_hz = 20.0;
    float energy_threshold = 15.0f;
    float spawn_threshold = 25.0f;
    float energy_decay = 0.95f;
    int peak_nms_px = 11;
    double snap_px = 20.0;
    double snap_gain = 0.4;
    double min_track_sep_px = 30.0;
    double lost_ms = 1000.0;
    double search_grow = 0.08;
    int source_search_px = 35;
    double sync_corr = 0.85;
    double source_margin = 40.0;
    int sample_pad = 3;
    int sample_max = 12;
    double min_contrast = 6.0;
    size_t max_tracks = 24;
    double hfov_deg = 65.0;
    double group_radius_px = 150.0;
    double core_drop = 15.0;
    double cv_window_ms = 300.0;
    double cv_max_gap_ms = 250.0;
    double cv_max_px_per_ms = 2.0;
    double point_min = 20.0;
    double point_rel = 0.5;

    double chip_ms() const { return 1000.0 / chip_hz; }
};

struct Peak {
    double x, y, radius, energy;
};

struct Track {
    Track(int tid, double x, double y, double ts, const Config& cfg, double radius)
        : tid(tid), x(x), y(y), created_ts(ts), last_seen_ts(ts),
          decoder(cfg.chip_hz, cfg.min_contrast), radius(radius) {}

    int tid;
    double x, y;
    double created_ts, last_seen_ts;
    TemplateDecoder decoder;
    double radius;
    std::deque<double> levels;
    std::deque<uint8_t> states;
    bool has_lohi = false;
    double lo = 0.0, hi = 0.0, contrast = 0.0;
    int roi[4] = {0, 0, 0, 0};
    int drone_id = NO_ID;
    bool id_confirmed = false;
    double inherited_ts = NAN;
    double on_peak = 0.0;
    int synced_with = 0;
    double core = 0.0;
    struct Obs {
        double ts, x, y;
    };
    std::deque<Obs> obs;
    double vx = 0.0, vy = 0.0;
    double point_peak = 0.0;

    bool inherited() const { return !std::isnan(inherited_ts); }
};

struct Drone {
    int drone_id;
    double x, y;
    double azimuth_deg, elevation_deg;
    int n_leds;
    double spread_px;
    std::vector<int> track_ids;
    bool confirmed;
};

class Receiver {
public:
    explicit Receiver(Config cfg = {});

    // bgr: 8-bit BGR, rows `stride` bytes apart.
    std::vector<Drone> process(const uint8_t* bgr, int width, int height, size_t stride, double ts);

    const Config& config() const { return cfg_; }
    const std::vector<std::unique_ptr<Track>>& tracks() const { return tracks_; }
    const std::vector<Peak>& last_peaks() const { return peaks_; }
    const std::vector<uint8_t>& feature() const { return feat_; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    void update_feature_and_energy(const uint8_t* bgr, size_t stride);
    float exposure_gain(const uint8_t* bgr, size_t stride) const;
    void find_peaks();
    void predict(double ts, double dt);
    void snap_and_spawn(double ts);
    void inherit_id(Track& t, double ts);
    void seek_source(Track& t);
    void lock_point(Track& t, double ts);
    void update_velocity(Track& t, double ts);
    double sample(Track& t);
    void update_id(Track& t, double ts);
    void adopt_synced_ids(double ts);
    bool still_blinking(const Track& t, double ts) const;
    void drop_lost(double ts);
    void dedupe();
    std::vector<Drone> group(double ts);

    float e(int x, int y) const { return energy_[static_cast<size_t>(y) * w_ + x]; }
    uint8_t f(int x, int y) const { return feat_[static_cast<size_t>(y) * w_ + x]; }

    Config cfg_;
    int w_ = 0, h_ = 0;
    std::vector<uint8_t> feat_, prev_feat_, diff_;
    static constexpr int HBLUR_ROWS = 8;
    std::vector<uint16_t> hblur_;
    std::vector<float> fdiff_, fhblur_;
    std::vector<float> energy_;
    bool have_feat_ = false, have_energy_ = false;
    std::vector<uint32_t> candidates_;
    std::vector<Peak> peaks_;

    std::vector<std::unique_ptr<Track>> tracks_;
    int next_tid_ = 1;
    double last_ts_ = NAN;
    double frame_dt_ = 33.3;
    struct Lost {
        double ts, x, y;
        int id;
    };
    std::vector<Lost> lost_;
    std::map<int, std::pair<double, double>> src_core_;
};

}  // namespace ledrx
