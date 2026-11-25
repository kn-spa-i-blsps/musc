// Finds blinking LEDs, tracks them (Kalman filter + Hungarian assignment), groups tracks blinking the
// same code into constellations, decodes IDs and reports drones with their bearing.
// Coordinates are in input image pixels.
#pragma once
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <vector>

#include "constellation.hpp"
#include "decoder.hpp"
#include "motion.hpp"
#include "tracking.hpp"

namespace ledrx {

struct Config {
    double chip_hz = 20.0;
    float energy_threshold = 15.0f;
    float spawn_threshold = 25.0f;
    float energy_decay = 0.95f;
    int peak_nms_px = 11;
    double snap_px = 30.0;  // 20 in the Python preset; 30 kept the LED with light and camera moving (2026-11-25 clips)
    double min_track_sep_px = 30.0;
    double lost_ms = 1000.0;
    double search_grow = 0.08;
    int source_search_px = 35;
    double source_margin = 40.0;
    int sample_pad = 3;
    int sample_max = 12;
    double min_contrast = 6.0;
    size_t max_tracks = 24;
    double hfov_deg = 65.0;
    double group_radius_px = 150.0;
    double core_drop = 15.0;
    double point_min = 20.0;
    double point_rel = 0.5;
    // Kalman filter (constant velocity) and assignment
    double kf_accel = 0.01;
    double kf_peak_px = 4.0;
    double kf_fine_px = 1.0;
    double kf_coast_ms = 250.0;
    double max_px_per_ms = 2.0;
    double assign_energy_w = 0.3;
    double kf_gate_sigma = 0.0;
    // Constellations: tracks blinking the same code share an ID
    bool constellations = true;
    double cs_min_events = 2.0;
    double cs_near_weight = 0.5;
    double cs_margin = 0.3;
    int cs_confirm_votes = 2;
    bool sync_adopt = true;  // only without constellations
    double sync_corr = 0.85;
    // Camera motion from the image when no gyro rotation is given
    bool motion_comp = false;

    double chip_ms() const { return 1000.0 / chip_hz; }
};

struct Peak {
    double x, y, radius, energy;
};

struct Track {
    Track(int tid, double x, double y, double ts, const Config& cfg, double radius)
        : tid(tid), x(x), y(y), created_ts(ts), last_seen_ts(ts), decoder(cfg.chip_hz, cfg.min_contrast),
          radius(radius), kf(x, y, cfg.kf_peak_px, 0.3, cfg.kf_accel) {}

    int tid;
    double x, y;
    double created_ts, last_seen_ts;
    TemplateDecoder decoder;
    double radius;
    KalmanCV kf;
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
    double point_peak = 0.0;
    int gid = 0;
    int cs_state = 0;

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

struct Constellation {
    int gid;
    std::vector<int> track_ids;
    double x, y;
    double since_ts;
    int drone_id;
    bool confirmed;
};

class Receiver {
public:
    explicit Receiver(Config cfg = {});

    // bgr: 8-bit BGR, rows `stride` bytes apart. cam_rot: camera rotation since the previous frame
    // (rotation vector in camera axes, rad, e.g. from the flight controller gyro), or null.
    std::vector<Drone> process(const uint8_t* bgr, int width, int height, size_t stride, double ts,
                               const double* cam_rot = nullptr);

    const Config& config() const { return cfg_; }
    const std::vector<std::unique_ptr<Track>>& tracks() const { return tracks_; }
    const std::vector<Peak>& last_peaks() const { return peaks_; }
    const std::vector<Constellation>& constellations() const { return constellations_; }
    const std::vector<uint8_t>& feature() const { return feat_; }
    double last_motion_x() const { return motion_[0]; }
    double last_motion_y() const { return motion_[1]; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    void compensate_rotation(const double rot[3]);
    void compensate_shift(const uint8_t* bgr, size_t stride);
    void warp_history(const Homography& H);
    void update_feature_and_energy(const uint8_t* bgr, size_t stride);
    float exposure_gain(const uint8_t* bgr, size_t stride) const;
    void find_peaks();
    void predict(double ts, double dt);
    void correct(Track& t, double zx, double zy, double meas_std);
    double reach(const Track& t, double ts) const;
    void assign_and_spawn(double ts);
    void inherit_id(Track& t, double ts);
    void seek_source(Track& t);
    void lock_point(Track& t, double ts);
    double sample(Track& t);
    void update_id(Track& t, double ts);
    void constellate(double ts);
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
    std::vector<float> prev_warped_;  // previous feature after camera motion compensation
    bool have_warped_ = false;
    static constexpr int HBLUR_ROWS = 8;
    std::vector<uint16_t> hblur_;
    std::vector<float> fdiff_, fhblur_;
    std::vector<float> energy_, energy_tmp_, scratch_;
    bool have_feat_ = false, have_energy_ = false;
    std::vector<uint32_t> candidates_;
    std::vector<Peak> peaks_;
    PhaseCorrelator phase_;
    double motion_[2] = {0, 0};

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
    BlinkGrouper grouper_;
    std::vector<Constellation> constellations_;
    std::map<int, double> group_since_;
};

}  // namespace ledrx
