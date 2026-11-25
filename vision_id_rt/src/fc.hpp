// Flight controller telemetry (Betaflight / INAV) over MSP v1: gyro rates for camera rotation, attitude.
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ledrx {

constexpr int MSP_FC_VARIANT = 2;
constexpr int MSP_RAW_IMU = 102;
constexpr int MSP_ATTITUDE = 108;

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;  // row-major

std::vector<uint8_t> msp_frame(int cmd, const std::vector<uint8_t>& payload = {});

struct MspReply {
    int cmd;  // -1: error reply or bad checksum
    std::vector<uint8_t> payload;
    size_t used;
};
// First complete MSP v1 reply in `buf`, or nothing if the frame is still incomplete.
std::optional<MspReply> msp_parse(const std::vector<uint8_t>& buf);

// FC (roll, pitch, yaw) -> camera axes (x right, y down, z forward); camera tilted up by tilt_deg.
Vec3 body_to_cam(const Vec3& rpy, double tilt_deg, const Vec3& signs = {1, 1, 1});
// The same mapping as a matrix: camera = M * (roll, pitch, yaw).
Mat3 mount_matrix(double tilt_deg, const Vec3& signs = {1, 1, 1});
Vec3 mat_apply(const Mat3& m, const Vec3& v);

double monotonic_ms();

class GyroIntegrator {
public:
    explicit GyroIntegrator(size_t maxlen = 4000) : maxlen_(maxlen) {}
    void add(double t_ms, const Vec3& rate_rad_s);
    // Rotation [rad] between t0 and t1 (trapezoid rule); nothing if the samples do not cover the interval.
    std::optional<Vec3> rotation(double t0_ms, double t1_ms, double max_gap_ms = 60.0) const;

private:
    size_t maxlen_;
    mutable std::mutex m_;
    std::deque<std::array<double, 4>> buf_;
};

class FcTelemetry {
public:
    FcTelemetry(const std::string& port, int baud = 115200, Mat3 mount = mount_matrix(0.0), double latency_ms = 0.0,
                double gyro_scale = 1.0, int attitude_every = 10);
    ~FcTelemetry();
    FcTelemetry(const FcTelemetry&) = delete;
    FcTelemetry& operator=(const FcTelemetry&) = delete;

    // Times are monotonic_ms(), the clock of the camera's SensorTimestamp.
    std::optional<Vec3> rotation_body(double t0_ms, double t1_ms) const;
    std::optional<Vec3> rotation(double t0_ms, double t1_ms) const;
    std::optional<Vec3> attitude() const;
    std::array<int, 3> acc_raw() const;
    double rate_hz() const;
    void close();

    const GyroIntegrator& gyro() const { return gyro_; }
    std::string variant;
    Mat3 mount;
    double latency_ms;
    double gyro_scale;  // raw MSP gyro units -> deg/s (some firmware sends raw sensor units)
    std::atomic<long> samples{0}, errors{0};

private:
    std::optional<std::vector<uint8_t>> request(int cmd, double timeout_s = 0.1);
    void run();

    int fd_ = -1;
    int attitude_every_;
    std::vector<uint8_t> buf_;
    GyroIntegrator gyro_;
    mutable std::mutex m_;
    std::optional<Vec3> attitude_;
    std::array<int, 3> acc_{0, 0, 0};
    double started_ms_ = 0.0;
    std::atomic<bool> running_{true};
    std::thread thread_;
};

struct Calibration {
    double corr, latency_ms, scale;
    Mat3 mount;
    int n;
};

// Fits gyro to image motion. shifts[i] = image shift (dx, dy) from frame i-1 to i on a static scene
// (NaN = unknown). Tries the latencies and all 24 right-angle mountings of the FC relative to the camera.
std::optional<Calibration> fit_calibration(const std::vector<double>& frame_ts,
                                           const std::vector<std::array<double, 2>>& shifts,
                                           const GyroIntegrator& gyro, double fx);

}  // namespace ledrx
