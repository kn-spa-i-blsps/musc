#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fc.hpp"
#include "motion.hpp"
#include "receiver.hpp"

namespace py = pybind11;
using namespace ledrx;

namespace {

py::object id_or_none(int id) { return id == NO_ID ? py::none() : py::cast(id); }

struct Image {
    const uint8_t* data;
    int width, height;
    size_t stride;
};

Image image(py::array& frame) {
    py::buffer_info b = frame.request();
    if (b.ndim != 3 || b.shape[2] != 3 || b.itemsize != 1 || b.strides[2] != 1 || b.strides[1] != 3)
        throw std::invalid_argument("expected an HxWx3 uint8 BGR image with packed pixels");
    return {static_cast<const uint8_t*>(b.ptr), static_cast<int>(b.shape[1]), static_cast<int>(b.shape[0]),
            static_cast<size_t>(b.strides[0])};
}

std::vector<Drone> process(Receiver& rx, py::array frame, double ts, std::optional<Vec3> cam_rot) {
    Image im = image(frame);
    py::gil_scoped_release release;
    return rx.process(im.data, im.width, im.height, im.stride, ts, cam_rot ? cam_rot->data() : nullptr);
}

py::list tracks(const Receiver& rx) {
    py::list out;
    for (const auto& t : rx.tracks()) {
        py::dict d;
        d["tid"] = t->tid;
        d["x"] = t->x;
        d["y"] = t->y;
        d["radius"] = t->radius;
        d["drone_id"] = id_or_none(t->drone_id);
        d["id_confirmed"] = t->id_confirmed;
        d["contrast"] = t->contrast;
        d["level"] = t->levels.empty() ? 0.0 : t->levels.back();
        d["lo"] = t->lo;
        d["hi"] = t->hi;
        d["gid"] = t->gid;
        d["states"] = std::vector<uint8_t>(t->states.begin(), t->states.end());
        out.append(d);
    }
    return out;
}

py::list constellations(const Receiver& rx) {
    py::list out;
    for (const auto& c : rx.constellations()) {
        py::dict d;
        d["gid"] = c.gid;
        d["track_ids"] = c.track_ids;
        d["x"] = c.x;
        d["y"] = c.y;
        d["since_ts"] = c.since_ts;
        d["drone_id"] = id_or_none(c.drone_id);
        d["confirmed"] = c.confirmed;
        out.append(d);
    }
    return out;
}

std::optional<Calibration> calibrate(const std::vector<double>& ts, const std::vector<std::array<double, 2>>& shifts,
                                     const FcTelemetry& fc, double fx) {
    py::gil_scoped_release release;
    return fit_calibration(ts, shifts, fc.gyro(), fx);
}

}  // namespace

PYBIND11_MODULE(ledrx, m) {
    m.doc() = "Real-time LED ID receiver with flight controller gyro support";

    py::class_<Drone>(m, "Drone")
        .def_readonly("drone_id", &Drone::drone_id)
        .def_readonly("x", &Drone::x)
        .def_readonly("y", &Drone::y)
        .def_readonly("azimuth_deg", &Drone::azimuth_deg)
        .def_readonly("elevation_deg", &Drone::elevation_deg)
        .def_readonly("n_leds", &Drone::n_leds)
        .def_readonly("spread_px", &Drone::spread_px)
        .def_readonly("track_ids", &Drone::track_ids)
        .def_readonly("confirmed", &Drone::confirmed)
        .def("__repr__", [](const Drone& d) {
            return "<Drone " + std::to_string(d.drone_id) + (d.confirmed ? "" : "?") + " at (" +
                   std::to_string(d.x) + ", " + std::to_string(d.y) + ")>";
        });

    py::class_<Receiver>(m, "Receiver")
        .def(py::init([](double chip_hz, bool motion_comp, bool constellations, double hfov_deg) {
                 Config cfg;
                 cfg.chip_hz = chip_hz;
                 cfg.motion_comp = motion_comp;
                 cfg.constellations = constellations;
                 cfg.hfov_deg = hfov_deg;
                 return Receiver(cfg);
             }),
             py::arg("chip_hz") = 20.0, py::arg("motion_comp") = false, py::arg("constellations") = true,
             py::arg("hfov_deg") = 65.0)
        .def("process", &process, py::arg("frame"), py::arg("ts_ms"), py::arg("cam_rot") = py::none(),
             "Process one BGR frame (HxWx3 uint8, rows may be padded). cam_rot: camera rotation since the "
             "previous frame (rotation vector in camera axes, rad), e.g. FcTelemetry.rotation().")
        .def("tracks", &tracks)
        .def("constellations", &constellations)
        .def_property_readonly("peaks",
                               [](const Receiver& rx) {
                                   std::vector<std::tuple<double, double, double>> out;
                                   for (const Peak& p : rx.last_peaks()) out.emplace_back(p.x, p.y, p.radius);
                                   return out;
                               })
        .def_property_readonly("last_motion", [](const Receiver& rx) {
            return std::make_pair(rx.last_motion_x(), rx.last_motion_y());
        });

    py::class_<PhaseCorrelator>(m, "PhaseCorrelator")
        .def(py::init<>())
        .def("update", [](PhaseCorrelator& pc, py::array frame) -> std::optional<std::tuple<double, double, double>> {
            Image im = image(frame);
            double dx, dy, resp;
            if (!pc.update(im.data, im.width, im.height, im.stride, dx, dy, resp)) return std::nullopt;
            return std::make_tuple(dx, dy, resp);
        }, "Image shift (dx, dy, response) since the previous frame, None for the first frame.");

    py::class_<Calibration>(m, "Calibration")
        .def_readonly("corr", &Calibration::corr)
        .def_readonly("latency_ms", &Calibration::latency_ms)
        .def_readonly("scale", &Calibration::scale)
        .def_readonly("mount", &Calibration::mount)
        .def_readonly("n", &Calibration::n);

    py::class_<FcTelemetry>(m, "FcTelemetry")
        .def(py::init([](const std::string& port, int baud, Mat3 mount, double latency_ms) {
                 py::gil_scoped_release release;
                 return std::make_unique<FcTelemetry>(port, baud, mount, latency_ms);
             }),
             py::arg("port"), py::arg("baud") = 115200, py::arg("mount") = mount_matrix(0.0),
             py::arg("latency_ms") = 0.0)
        .def("rotation", &FcTelemetry::rotation, py::arg("t0_ms"), py::arg("t1_ms"),
             "Camera rotation between two monotonic_ms() times, or None without gyro data.")
        .def("rotation_body", &FcTelemetry::rotation_body, py::arg("t0_ms"), py::arg("t1_ms"))
        .def_property_readonly("attitude", &FcTelemetry::attitude)
        .def_property_readonly("acc_raw", &FcTelemetry::acc_raw)
        .def_property_readonly("rate_hz", &FcTelemetry::rate_hz)
        .def_property_readonly("samples", [](const FcTelemetry& f) { return f.samples.load(); })
        .def_property_readonly("errors", [](const FcTelemetry& f) { return f.errors.load(); })
        .def_readonly("variant", &FcTelemetry::variant)
        .def_readwrite("mount", &FcTelemetry::mount)
        .def_readwrite("latency_ms", &FcTelemetry::latency_ms)
        .def("close", &FcTelemetry::close, py::call_guard<py::gil_scoped_release>());

    m.def("monotonic_ms", &monotonic_ms, "CLOCK_MONOTONIC in ms (the clock of picamera2 SensorTimestamp).");
    m.def("mount_matrix", &mount_matrix, py::arg("tilt_deg") = 0.0, py::arg("signs") = Vec3{1, 1, 1},
          "FC on the drone, camera looking forward and tilted up by tilt_deg.");
    m.def("fit_calibration", &calibrate, py::arg("frame_ts"), py::arg("shifts"), py::arg("fc"), py::arg("fx"));
}
