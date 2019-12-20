#include "common.h"

void bind_cuda_1d(py::module& m) {
    auto mask_class = bind<MaskC>(m, "MaskC");
    auto float_class = bind<FloatC>(m, "FloatC");
    auto int32_class = bind<Int32C>(m, "Int32C");
    auto int64_class = bind<Int64C>(m, "Int64C");
    auto uint32_class = bind<UInt32C>(m, "UInt32C");
    auto uint64_class = bind<UInt64C>(m, "UInt64C");

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
}
