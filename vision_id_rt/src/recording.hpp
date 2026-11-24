#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ledrx {

// MJPG frames of an AVI file (as written by OpenCV), read into memory.
class MjpgAvi {
public:
    explicit MjpgAvi(const std::string& path);
    size_t size() const { return frames_.size(); }
    const uint8_t* data(size_t i) const { return file_.data() + frames_[i].first; }
    size_t bytes(size_t i) const { return frames_[i].second; }

private:
    void walk(size_t begin, size_t end);
    std::vector<uint8_t> file_;
    std::vector<std::pair<size_t, size_t>> frames_;
};

class JpegDecoder {
public:
    JpegDecoder();
    ~JpegDecoder();
    JpegDecoder(const JpegDecoder&) = delete;
    JpegDecoder& operator=(const JpegDecoder&) = delete;

    // Decodes to tightly packed BGR, resizing `out` as needed.
    bool decode(const uint8_t* jpg, size_t n, std::vector<uint8_t>& out, int& width, int& height);

private:
    void* handle_;
};

std::vector<double> load_npy_f64(const std::string& path);

}  // namespace ledrx
