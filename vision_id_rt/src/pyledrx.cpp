#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "receiver.hpp"

namespace py = pybind11;
using namespace ledrx;

namespace {

py::object id_or_none(int id) { return id == NO_ID ? py::none() : py::cast(id); }

std::vector<Drone> process(Receiver& rx, py::array frame, double ts) {
    py::buffer_info b = frame.request();
    if (b.ndim != 3 || b.shape[2] != 3 || b.itemsize != 1 || b.strides[2] != 1 || b.strides[1] != 3)
        throw std::invalid_argument("expected an HxWx3 uint8 BGR image with packed pixels");
    const auto* data = static_cast<const uint8_t*>(b.ptr);
    py::gil_scoped_release release;
    return rx.process(data, static_cast<int>(b.shape[1]), static_cast<int>(b.shape[0]),
                      static_cast<size_t>(b.strides[0]), ts);
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
        d["states"] = std::vector<uint8_t>(t->states.begin(), t->states.end());
        out.append(d);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(ledrx, m) {
    m.doc() = "Real-time LED ID receiver";

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
        .def(py::init([](double chip_hz) {
                 Config cfg;
                 cfg.chip_hz = chip_hz;
                 return Receiver(cfg);
             }),
             py::arg("chip_hz") = 20.0)
        .def("process", &process, py::arg("frame"), py::arg("ts_ms"),
             "Process one BGR frame (HxWx3 uint8, rows may be padded). Returns confirmed and tentative drones.")
        .def("tracks", &tracks)
        .def_property_readonly("peaks", [](const Receiver& rx) {
            std::vector<std::tuple<double, double, double>> out;
            for (const Peak& p : rx.last_peaks()) out.emplace_back(p.x, p.y, p.radius);
            return out;
        });
}
