#include "fc.hpp"

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ledrx {

std::vector<uint8_t> msp_frame(int cmd, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> out = {'$', 'M', '<', static_cast<uint8_t>(payload.size()), static_cast<uint8_t>(cmd)};
    out.insert(out.end(), payload.begin(), payload.end());
    uint8_t chk = 0;
    for (size_t i = 3; i < out.size(); ++i) chk ^= out[i];
    out.push_back(chk);
    return out;
}

std::optional<MspReply> msp_parse(const std::vector<uint8_t>& buf) {
    size_t i = 0;
    while (i + 1 < buf.size() && !(buf[i] == '$' && buf[i + 1] == 'M')) ++i;
    if (i + 1 >= buf.size() || buf.size() < i + 6) return std::nullopt;
    uint8_t size = buf[i + 3], cmd = buf[i + 4];
    size_t end = i + 5 + size + 1;
    if (buf.size() < end) return std::nullopt;
    uint8_t chk = size ^ cmd;
    for (size_t k = i + 5; k < i + 5 + size; ++k) chk ^= buf[k];
    if (buf[i + 2] != '>' || chk != buf[end - 1]) return MspReply{-1, {}, end};
    return MspReply{cmd, std::vector<uint8_t>(buf.begin() + i + 5, buf.begin() + i + 5 + size), end};
}

Vec3 body_to_cam(const Vec3& rpy, double tilt_deg, const Vec3& signs) {
    double p = rpy[0] * signs[0], q = rpy[1] * signs[1], r = rpy[2] * signs[2];
    double c = std::cos(tilt_deg * M_PI / 180), s = std::sin(tilt_deg * M_PI / 180);
    return {q, r * c + p * s, p * c - r * s};
}

Mat3 mount_matrix(double tilt_deg, const Vec3& sg) {
    double c = std::cos(tilt_deg * M_PI / 180), s = std::sin(tilt_deg * M_PI / 180);
    return {0, sg[1], 0, s * sg[0], 0, c * sg[2], c * sg[0], 0, -s * sg[2]};
}

Vec3 mat_apply(const Mat3& m, const Vec3& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

double monotonic_ms() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

void GyroIntegrator::add(double t_ms, const Vec3& w) {
    std::lock_guard<std::mutex> lock(m_);
    buf_.push_back({t_ms, w[0], w[1], w[2]});
    if (buf_.size() > maxlen_) buf_.pop_front();
}

std::optional<Vec3> GyroIntegrator::rotation(double t0, double t1, double max_gap) const {
    if (t1 <= t0) return std::nullopt;
    std::vector<std::array<double, 4>> a;
    {
        std::lock_guard<std::mutex> lock(m_);
        a.assign(buf_.begin(), buf_.end());
    }
    if (a.size() < 2 || a.front()[0] > t0 + max_gap || a.back()[0] < t1 - max_gap) return std::nullopt;
    auto by_time = [](const std::array<double, 4>& s, double t) { return s[0] < t; };
    size_t lo = std::lower_bound(a.begin(), a.end(), t0, by_time) - a.begin();
    size_t hi = std::lower_bound(a.begin(), a.end(), t1, by_time) - a.begin();
    lo = lo > 0 ? lo - 1 : 0;
    hi = std::min(hi + 1, a.size());
    std::vector<std::array<double, 4>> s(a.begin() + lo, a.begin() + hi);
    if (s.size() < 2) return std::nullopt;
    for (size_t i = 1; i < s.size(); ++i)
        if (s[i][0] - s[i - 1][0] > max_gap) return std::nullopt;

    std::vector<double> grid = {t0, t1};
    for (const auto& x : s) grid.push_back(std::clamp(x[0], t0, t1));
    std::sort(grid.begin(), grid.end());
    grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
    auto interp = [&](double t, int k) {
        if (t <= s.front()[0]) return s.front()[k];
        if (t >= s.back()[0]) return s.back()[k];
        size_t j = std::upper_bound(s.begin(), s.end(), t, [](double v, const auto& e) { return v < e[0]; }) - s.begin();
        double f = (t - s[j - 1][0]) / (s[j][0] - s[j - 1][0]);
        return s[j - 1][k] + f * (s[j][k] - s[j - 1][k]);
    };
    Vec3 out{};
    for (int k = 0; k < 3; ++k)
        for (size_t i = 1; i < grid.size(); ++i)
            out[k] += 0.5 * (interp(grid[i], k + 1) + interp(grid[i - 1], k + 1)) * (grid[i] - grid[i - 1]) / 1000.0;
    return out;
}

namespace {

speed_t baud_constant(int baud) {
    switch (baud) {
        case 9600: return B9600;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
#ifdef B460800
        case 460800: return B460800;
#endif
#ifdef B921600
        case 921600: return B921600;
#endif
        default: throw std::invalid_argument("unsupported baud rate " + std::to_string(baud));
    }
}

}  // namespace

FcTelemetry::FcTelemetry(const std::string& port, int baud, Mat3 m, double latency, double scale, int attitude_every)
    : mount(m), latency_ms(latency), gyro_scale(scale), attitude_every_(attitude_every) {
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) throw std::runtime_error("cannot open " + port + ": " + std::strerror(errno));
    termios tio{};
    tcgetattr(fd_, &tio);
    cfmakeraw(&tio);
    cfsetispeed(&tio, baud_constant(baud));
    cfsetospeed(&tio, baud_constant(baud));
    tio.c_cflag |= CLOCAL | CREAD;
    tcsetattr(fd_, TCSANOW, &tio);
    usleep(100000);
    tcflush(fd_, TCIFLUSH);
    auto got = request(MSP_FC_VARIANT);
    if (!got) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("FC on " + port +
                                 " does not answer MSP (wrong port? MSP disabled on this UART? Configurator open?)");
    }
    variant.assign(got->begin(), got->end());
    started_ms_ = monotonic_ms();
    thread_ = std::thread(&FcTelemetry::run, this);
}

FcTelemetry::~FcTelemetry() { close(); }

void FcTelemetry::close() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

std::optional<std::vector<uint8_t>> FcTelemetry::request(int cmd, double timeout_s) {
    auto frame = msp_frame(cmd);
    if (::write(fd_, frame.data(), frame.size()) != static_cast<ssize_t>(frame.size())) return std::nullopt;
    double end = monotonic_ms() + timeout_s * 1000;
    uint8_t chunk[256];
    while (monotonic_ms() < end) {
        pollfd p{fd_, POLLIN, 0};
        if (poll(&p, 1, 5) > 0) {
            ssize_t n = ::read(fd_, chunk, sizeof chunk);
            if (n > 0) buf_.insert(buf_.end(), chunk, chunk + n);
        }
        while (true) {
            auto reply = msp_parse(buf_);
            if (!reply) {
                if (buf_.size() > 512) buf_.erase(buf_.begin(), buf_.end() - 64);
                break;
            }
            buf_.erase(buf_.begin(), buf_.begin() + reply->used);
            if (reply->cmd == cmd) return reply->payload;
        }
    }
    return std::nullopt;
}

void FcTelemetry::run() {
    long k = 0;
    auto i16 = [](const std::vector<uint8_t>& b, size_t i) { return static_cast<int16_t>(b[i] | b[i + 1] << 8); };
    while (running_) {
        double t_send = monotonic_ms();
        auto raw = request(MSP_RAW_IMU);
        if (!raw || raw->size() < 12) {
            ++errors;
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            acc_ = {i16(*raw, 0), i16(*raw, 2), i16(*raw, 4)};
        }
        // Sample time: middle between request and reply.
        Vec3 w = {i16(*raw, 6) * M_PI / 180, i16(*raw, 8) * M_PI / 180, i16(*raw, 10) * M_PI / 180};
        gyro_.add(0.5 * (t_send + monotonic_ms()), w);
        ++samples;
        if (++k % attitude_every_ == 0) {
            auto att = request(MSP_ATTITUDE);
            if (att && att->size() >= 6) {
                std::lock_guard<std::mutex> lock(m_);
                attitude_ = Vec3{i16(*att, 0) / 10.0, i16(*att, 2) / 10.0, static_cast<double>(i16(*att, 4))};
            }
        }
    }
}

std::optional<Vec3> FcTelemetry::rotation_body(double t0, double t1) const {
    auto rot = gyro_.rotation(t0 - latency_ms, t1 - latency_ms);
    if (rot)
        for (double& v : *rot) v *= gyro_scale;
    return rot;
}

std::optional<Vec3> FcTelemetry::rotation(double t0, double t1) const {
    auto rot = rotation_body(t0, t1);
    if (!rot) return std::nullopt;
    return mat_apply(mount, *rot);
}

std::optional<Vec3> FcTelemetry::attitude() const {
    std::lock_guard<std::mutex> lock(m_);
    return attitude_;
}

std::array<int, 3> FcTelemetry::acc_raw() const {
    std::lock_guard<std::mutex> lock(m_);
    return acc_;
}

double FcTelemetry::rate_hz() const { return samples / std::max((monotonic_ms() - started_ms_) / 1000, 1e-3); }

std::optional<Calibration> fit_calibration(const std::vector<double>& frame_ts,
                                           const std::vector<std::array<double, 2>>& shifts,
                                           const GyroIntegrator& gyro, double fx) {
    // Right-angle mountings: camera x and y are each +-one FC axis, z = x cross y.
    std::vector<Mat3> mounts;
    for (int ax = 0; ax < 3; ++ax)
        for (int ay = 0; ay < 3; ++ay) {
            if (ay == ax) continue;
            for (double sx : {1.0, -1.0})
                for (double sy : {1.0, -1.0}) {
                    double X[3] = {}, Y[3] = {};
                    X[ax] = sx;
                    Y[ay] = sy;
                    mounts.push_back({X[0], X[1], X[2], Y[0], Y[1], Y[2], X[1] * Y[2] - X[2] * Y[1],
                                      X[2] * Y[0] - X[0] * Y[2], X[0] * Y[1] - X[1] * Y[0]});
                }
        }
    std::optional<Calibration> best;
    for (int lag = -120; lag <= 120; lag += 10) {
        std::vector<Vec3> rots;
        std::vector<std::array<double, 2>> meas;
        for (size_t i = 1; i < frame_ts.size(); ++i) {
            auto rot = gyro.rotation(frame_ts[i - 1] - lag, frame_ts[i] - lag);
            if (rot && std::isfinite(shifts[i][0]) && std::isfinite(shifts[i][1])) {
                rots.push_back(*rot);
                meas.push_back(shifts[i]);
            }
        }
        if (rots.size() < 30) continue;
        for (const Mat3& m : mounts) {
            double num = 0, pp = 0, mm = 0;
            for (size_t i = 0; i < rots.size(); ++i) {
                Vec3 cam = mat_apply(m, rots[i]);
                double px = -fx * cam[1], py = fx * cam[0];  // camera turning right moves the image left
                num += px * meas[i][0] + py * meas[i][1];
                pp += px * px + py * py;
                mm += meas[i][0] * meas[i][0] + meas[i][1] * meas[i][1];
            }
            double den = std::sqrt(pp * mm);
            double corr = den > 0 ? num / den : 0.0;
            if (!best || corr > best->corr)
                best = Calibration{corr, static_cast<double>(lag), pp > 0 ? num / pp : 0.0, m,
                                   static_cast<int>(rots.size())};
        }
    }
    return best;
}

}  // namespace ledrx
