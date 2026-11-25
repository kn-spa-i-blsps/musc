// Decode LED IDs from a station recording (raw.avi + ts.npy + info.json).
//
//   ledrx_rec REC_DIR [--chip-hz 20] [--pace] [--motion-comp] [--dump frames.jsonl] [--out result.json]
//
// --pace feeds frames at their recorded timestamps, like a live camera, so the reported CPU
// load is what real-time decoding costs.
#include <time.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <thread>

#include "receiver.hpp"
#include "recording.hpp"

using namespace ledrx;
using Clock = std::chrono::steady_clock;

namespace {

struct Frame {
    std::vector<uint8_t> bgr;
    int width = 0, height = 0;
    size_t index = 0;
};

class FrameQueue {
public:
    explicit FrameQueue(size_t capacity) : capacity_(capacity) {}

    void push(Frame f) {
        std::unique_lock<std::mutex> lock(m_);
        not_full_.wait(lock, [&] { return q_.size() < capacity_; });
        q_.push_back(std::move(f));
        not_empty_.notify_one();
    }

    bool pop(Frame& f) {
        std::unique_lock<std::mutex> lock(m_);
        not_empty_.wait(lock, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        f = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(m_);
        closed_ = true;
        not_empty_.notify_all();
    }

private:
    size_t capacity_;
    std::deque<Frame> q_;
    bool closed_ = false;
    std::mutex m_;
    std::condition_variable not_full_, not_empty_;
};

double thread_cpu_s() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

int transmitter_id(const std::string& dir) {
    std::ifstream in(dir + "/info.json");
    std::stringstream ss;
    ss << in.rdbuf();
    std::smatch m;
    std::string s = ss.str();
    if (std::regex_search(s, m, std::regex(R"(ID=(\d+))"))) return std::stoi(m[1]);
    return NO_ID;
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q / 100 * v.size()))];
}

int usage() {
    std::cerr << "usage: ledrx_rec REC_DIR [--chip-hz HZ] [--pace] [--motion-comp] [--dump FRAMES.jsonl] [--out RESULT.json]\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string dir = argv[1], dump_path, out_path = std::string(argv[1]) + "/result_rt.json";
    Config cfg;
    bool pace = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--chip-hz" && i + 1 < argc) cfg.chip_hz = std::stod(argv[++i]);
        else if (a == "--pace") pace = true;
        else if (a == "--motion-comp") cfg.motion_comp = true;
        else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else return usage();
    }

    MjpgAvi avi(dir + "/raw.avi");
    std::vector<double> ts = load_npy_f64(dir + "/ts.npy");
    size_t n = std::min(avi.size(), ts.size());
    int true_id = transmitter_id(dir);

    FrameQueue queue(pace ? 8 : 4);
    double decode_cpu = 0;
    std::thread reader([&] {
        double c0 = thread_cpu_s();
        JpegDecoder jpeg;
        for (size_t i = 0; i < n; ++i) {
            Frame f;
            f.index = i;
            if (!jpeg.decode(avi.data(i), avi.bytes(i), f.bgr, f.width, f.height)) break;
            queue.push(std::move(f));
        }
        decode_cpu = thread_cpu_s() - c0;
        queue.close();
    });

    std::ofstream dump;
    if (!dump_path.empty()) dump.open(dump_path);
    dump.precision(12);
    Receiver rx(cfg);
    std::vector<double> proc_ms;
    std::map<int, int> wrong;
    std::map<int, std::pair<int, int>> per_sec;
    double first = -1;
    int hit = 0, after = 0, late = 0;

    auto t0 = Clock::now();
    double c0 = thread_cpu_s();
    Frame f;
    while (queue.pop(f)) {
        double t = ts[f.index] - ts[0];
        if (pace) {
            auto due = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(t));
            if (Clock::now() > due + std::chrono::milliseconds(5)) ++late;
            std::this_thread::sleep_until(due);
        }
        auto p0 = Clock::now();
        std::vector<Drone> drones = rx.process(f.bgr.data(), f.width, f.height, f.width * 3, t);
        proc_ms.push_back(seconds_since(p0) * 1000);

        std::set<int> ids;
        for (const Drone& d : drones) ids.insert(d.drone_id);
        bool good = true_id != NO_ID ? ids.count(true_id) > 0 : !ids.empty();
        auto& sec = per_sec[static_cast<int>(t / 1000)];
        ++sec.first;
        sec.second += good;
        if (first < 0 && good) first = t;
        if (first >= 0) {
            ++after;
            hit += good;
        }
        for (int id : ids)
            if (true_id != NO_ID && id != true_id) ++wrong[id];
        if (dump) {
            dump << "{\"i\":" << f.index << ",\"ts\":" << t << ",\"tracks\":" << rx.tracks().size() << ",\"drones\":[";
            for (size_t k = 0; k < drones.size(); ++k)
                dump << (k ? "," : "") << "[" << drones[k].drone_id << "," << drones[k].x << "," << drones[k].y << ","
                     << drones[k].confirmed << "]";
            dump << "],\"tr\":[";
            for (size_t k = 0; k < rx.tracks().size(); ++k) {
                const Track& t = *rx.tracks()[k];
                dump << (k ? "," : "") << "[" << t.tid << "," << t.x << "," << t.y << "," << t.drone_id << "]";
            }
            dump << "]}\n";
        }
    }
    double wall = seconds_since(t0), detect_cpu = thread_cpu_s() - c0;
    reader.join();

    size_t frames = proc_ms.size();
    double mean_ms = 0;
    for (double v : proc_ms) mean_ms += v;
    mean_ms /= std::max<size_t>(frames, 1);

    std::ofstream out(out_path);
    out << "{\n \"transmitter_id\": " << (true_id == NO_ID ? std::string("null") : std::to_string(true_id))
        << ",\n \"frames\": " << frames << ",\n \"first_id_s\": "
        << (first < 0 ? std::string("null") : std::to_string(first / 1000)) << ",\n \"frames_with_id_after_first\": ["
        << hit << ", " << after << "],\n \"percent\": "
        << (after ? std::to_string(100.0 * hit / after) : std::string("null")) << ",\n \"false_ids\": {";
    bool comma = false;
    for (auto& [id, cnt] : wrong) out << (comma ? ", " : "") << "\"" << id << "\": " << cnt, comma = true;
    out << "},\n \"detect_ms_mean\": " << mean_ms << ",\n \"detect_ms_p99\": " << pct(proc_ms, 99)
        << ",\n \"seconds\": {";
    comma = false;
    for (auto& [s, v] : per_sec)
        out << (comma ? ", " : "") << "\"" << s << "\": {\"frames\": " << v.first << ", \"with_id\": " << v.second << "}",
            comma = true;
    out << "}\n}\n";

    double video_s = frames > 1 ? (ts[frames - 1] - ts[0]) / 1000 : 0;
    std::printf("%s: %zu frames, %.1f s of video\n", dir.c_str(), frames, video_s);
    std::printf("detect: mean %.2f ms, p50 %.2f, p99 %.2f, max %.2f ms/frame\n", mean_ms, pct(proc_ms, 50),
                pct(proc_ms, 99), pct(proc_ms, 100));
    std::printf("jpeg decode: %.2f ms/frame\n", decode_cpu * 1000 / std::max<size_t>(frames, 1));
    std::printf("wall %.2f s -> %.1f fps%s\n", wall, frames / wall, pace ? " (paced)" : "");
    std::printf("cpu: detect %.1f%% + decode %.1f%% of one core\n", 100 * detect_cpu / wall, 100 * decode_cpu / wall);
    if (pace) std::printf("frames late >5 ms: %d\n", late);
    std::printf("first ID %s, then %d/%d frames with ID %d, false IDs: %zu kinds\n",
                first < 0 ? "never" : (std::to_string(first / 1000) + " s").c_str(), hit, after, true_id, wrong.size());
    return 0;
}
