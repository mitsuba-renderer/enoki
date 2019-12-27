#include "common.h"
#include <pybind11/functional.h>

void bind_dynamic_1d(py::module& m) {
    auto mask_class = bind<MaskX>(m, "MaskX");
    auto uint32_class = bind<UInt32X>(m, "UInt32X");
    auto uint64_class = bind<UInt64X>(m, "UInt64X");
    auto int32_class = bind<Int32X>(m, "Int32X");
    auto int64_class = bind<Int64X>(m, "Int64X");
    auto float_class = bind<FloatX>(m, "FloatX");

    float_class
        .def(py::init<const Int32X &>())
        .def(py::init<const Int64X &>())
        .def(py::init<const UInt32X &>())
        .def(py::init<const UInt64X &>());

    int32_class
        .def(py::init<const FloatX &>())
        .def(py::init<const Int64X &>())
        .def(py::init<const UInt32X &>())
        .def(py::init<const UInt64X &>());

    int64_class
        .def(py::init<const FloatX &>())
        .def(py::init<const Int32X &>())
        .def(py::init<const UInt32X &>())
        .def(py::init<const UInt64X &>());

    uint32_class
        .def(py::init<const FloatX &>())
        .def(py::init<const Int32X &>())
        .def(py::init<const Int64X &>())
        .def(py::init<const UInt64X &>());

    uint64_class
        .def(py::init<const FloatX &>())
        .def(py::init<const Int32X &>())
        .def(py::init<const Int64X &>())
        .def(py::init<const UInt32X &>());

    bind<Vector0mX>(m, "Vector0mX");
    bind<Vector0fX>(m, "Vector0fX");
    bind<Vector1mX>(m, "Vector1mX");
    bind<Vector1fX>(m, "Vector1fX");

    m.def(
        "binary_search",
        [](uint32_t start,
           uint32_t end,
           const std::function<MaskX (const UInt32X &, const MaskX &)> &pred,
           const MaskX &mask) {
            return enoki::binary_search(start, end, pred, mask);
        },
        "start"_a, "end"_a, "pred"_a, "mask"_a = true);

    m.def("meshgrid", [](const FloatX &x, const FloatX &y) {
        auto result = meshgrid(x, y);
        return std::make_pair(std::move(result.x()), std::move(result.y()));
    });
}
