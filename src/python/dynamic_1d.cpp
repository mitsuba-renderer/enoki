#include "common.h"

void bind_dynamic_1d(py::module& m) {
    auto mask_class = bind<MaskX>(m, "MaskX");
    auto float_class = bind<FloatX>(m, "FloatX");
    auto int32_class = bind<Int32X>(m, "Int32X");
    auto int64_class = bind<Int64X>(m, "Int64X");
    auto uint32_class = bind<UInt32X>(m, "UInt32X");
    auto uint64_class = bind<UInt64X>(m, "UInt64X");

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
}
