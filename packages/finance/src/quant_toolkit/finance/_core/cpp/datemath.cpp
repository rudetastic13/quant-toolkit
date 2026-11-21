// pybind11 bindings for civil.hpp — the finance._core.datemath extension.
//
// Epoch contract: every integer day count crossing this boundary is days since
// 1970-01-01 (numpy datetime64[D] epoch), matching Date.toordinal(). Callers
// converting from datetime.date must subtract 719_163 (its 0001-01-01 ordinal
// of 1970-01-01); Date and datetime64[D] values convert with no offset.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "civil.hpp"

namespace py = pybind11;

namespace {

py::array_t<std::int64_t> py_generate_schedule(std::int64_t start, std::int64_t end,
                                               int freq_type, int freq_value,
                                               std::int64_t first_regular, std::int64_t last_regular,
                                               int roll, int direction) {
    const std::size_t cap = fin::core::schedule_capacity(start, end, freq_type, freq_value);
    py::array_t<std::int64_t> out(static_cast<py::ssize_t>(cap));
    std::int64_t* data = out.mutable_data();
    std::size_t count;
    {
        py::gil_scoped_release release;
        count = fin::core::generate_schedule(start, end, freq_type, freq_value,
                                             first_regular, last_regular,
                                             roll, direction, data, cap);
    }
    out.resize({static_cast<py::ssize_t>(count)});
    return out;
}

}  // namespace

PYBIND11_MODULE(datemath, m) {
    m.doc() =
        "Low-level date math and schedule generation (C++).\n\n"
        "Epoch contract: all day counts are days since 1970-01-01 (numpy\n"
        "datetime64[D] epoch), matching Date.toordinal(). datetime.date\n"
        "ordinals are 0001-01-01 based and must be shifted by 719_163 before\n"
        "crossing this boundary.";

    m.def("days_from_ymd", &fin::core::days_from_ymd, py::arg("y"), py::arg("m"), py::arg("d"),
          "Day count since 1970-01-01 for a civil (year, month, day).");
    m.def(
        "ymd_from_days",
        [](std::int64_t days) {
            const fin::core::YMD ymd = fin::core::ymd_from_days(days);
            return py::make_tuple(ymd.y, ymd.m, ymd.d);
        },
        py::arg("days"), "Civil (year, month, day) for a day count since 1970-01-01.");
    m.def("is_leap_year", &fin::core::is_leap, py::arg("y"));
    m.def("days_in_month", &fin::core::days_in_month, py::arg("y"), py::arg("m"));
    m.def("generate_schedule", &py_generate_schedule,
          py::arg("start"), py::arg("end"), py::arg("freq_type"), py::arg("freq_value"),
          py::arg("first_regular"), py::arg("last_regular"), py::arg("roll"), py::arg("direction"),
          "Ascending schedule of day counts (int64, 1970 epoch). Mirrors\n"
          "finance.dates.schedules.vectorized.generate_schedule semantics.");
}
