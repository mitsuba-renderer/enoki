#include "common.h"
#include <pybind11/functional.h>

void bind_cuda_1d(py::module& m) {
    auto mask_class = bind<MaskC>(m, "MaskC");
    auto uint32_class = bind<UInt32C>(m, "UInt32C");
    auto uint64_class = bind<UInt64C>(m, "UInt64C");
    auto int32_class = bind<Int32C>(m, "Int32C");
    auto int64_class = bind<Int64C>(m, "Int64C");
    auto float_class = bind<FloatC>(m, "FloatC");

    float_class
        .def(py::init<const Int32C &>())
        .def(py::init<const Int64C &>())
        .def(py::init<const UInt32C &>())
        .def(py::init<const UInt64C &>());

    int32_class
        .def(py::init<const FloatC &>())
        .def(py::init<const Int64C &>())
        .def(py::init<const UInt32C &>())
        .def(py::init<const UInt64C &>());

    int64_class
        .def(py::init<const FloatC &>())
        .def(py::init<const Int32C &>())
        .def(py::init<const UInt32C &>())
        .def(py::init<const UInt64C &>());

    uint32_class
        .def(py::init<const FloatC &>())
        .def(py::init<const Int32C &>())
        .def(py::init<const Int64C &>())
        .def(py::init<const UInt64C &>());

    uint64_class
        .def(py::init<const FloatC &>())
        .def(py::init<const Int32C &>())
        .def(py::init<const Int64C &>())
        .def(py::init<const UInt32C &>());

    bind<Vector0mC>(m, "Vector0mC");
    bind<Vector0fC>(m, "Vector0fC");
    bind<Vector1mC>(m, "Vector1mC");
    bind<Vector1fC>(m, "Vector1fC");

    m.def(
        "binary_search",
        [](uint32_t start,
           uint32_t end,
           const std::function<MaskC(UInt32C, MaskC)> &pred,
           MaskC mask) {
            return enoki::binary_search(start, end, pred, mask);
        },
        "start"_a, "end"_a, "pred"_a, "mask"_a = true);

    m.def("meshgrid", [](const FloatC &x, const FloatC &y) {
        auto result = meshgrid(x, y);
        return std::make_pair(std::move(result.x()), std::move(result.y()));
    });
}
