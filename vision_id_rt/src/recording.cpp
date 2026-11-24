#include "recording.hpp"

#include <turbojpeg.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace ledrx {
namespace {

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

uint32_t le32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | static_cast<uint32_t>(p[3]) << 24; }

}  // namespace

MjpgAvi::MjpgAvi(const std::string& path) : file_(read_file(path)) {
    walk(0, file_.size());
    if (frames_.empty()) throw std::runtime_error("no video frames in " + path);
}

void MjpgAvi::walk(size_t pos, size_t end) {
    while (pos + 8 <= end) {
        const uint8_t* id = file_.data() + pos;
        size_t size = le32(id + 4);
        size_t body = pos + 8;
        if (body + size > end) size = end - body;
        if (!std::memcmp(id, "RIFF", 4) || !std::memcmp(id, "LIST", 4)) {
            walk(body + 4, body + size);
        } else if (id[2] == 'd' && (id[3] == 'c' || id[3] == 'b') && size > 0) {
            frames_.emplace_back(body, size);
        }
        pos = body + size + (size & 1);
    }
}

JpegDecoder::JpegDecoder() : handle_(tjInitDecompress()) {
    if (!handle_) throw std::runtime_error("tjInitDecompress failed");
}

JpegDecoder::~JpegDecoder() { tjDestroy(handle_); }

bool JpegDecoder::decode(const uint8_t* jpg, size_t n, std::vector<uint8_t>& out, int& width, int& height) {
    int subsamp, colorspace;
    auto* src = const_cast<unsigned char*>(jpg);
    if (tjDecompressHeader3(handle_, src, n, &width, &height, &subsamp, &colorspace)) return false;
    out.resize(static_cast<size_t>(width) * height * 3);
    return tjDecompress2(handle_, src, n, out.data(), width, 0, height, TJPF_BGR, 0) == 0;
}

std::vector<double> load_npy_f64(const std::string& path) {
    std::vector<uint8_t> raw = read_file(path);
    if (raw.size() < 10 || std::memcmp(raw.data(), "\x93NUMPY", 6)) throw std::runtime_error("not a .npy file: " + path);
    int major = raw[6];
    size_t header_len = major == 1 ? (raw[8] | raw[9] << 8) : le32(raw.data() + 8);
    size_t offset = (major == 1 ? 10 : 12) + header_len;
    std::string header(raw.begin() + (major == 1 ? 10 : 12), raw.begin() + offset);
    if (header.find("'<f8'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("expected a C-ordered float64 array in " + path);
    size_t n = (raw.size() - offset) / sizeof(double);
    std::vector<double> out(n);
    std::memcpy(out.data(), raw.data() + offset, n * sizeof(double));
    return out;
}

}  // namespace ledrx
