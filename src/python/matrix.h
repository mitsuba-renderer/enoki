#pragma once
#include "common.h"
#include <enoki/matrix.h>

template <typename Matrix>
py::class_<Matrix> bind_matrix(py::module &m, py::module &s, const char *name) {
    using Vector = typename Matrix::Column;
    using Value  = typename Matrix::Entry;
    using Array  = Array<Vector, Matrix::Size>;
    using Scalar = scalar_t<Matrix>;
    static constexpr bool IsDynamic = is_dynamic_v<Value>;

    auto cls = py::class_<Matrix>(s, name)
        .def(py::init<>())
        .def(py::init<const Value &>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self - py::self)
        .def(py::self + py::self)
        .def(py::self * Value())
        .def(-py::self)
        .def("__mul__", [](const Matrix &a, const Matrix &b) {
            return Matrix(Array(a) * Array(b));
        })
        .def("__repr__", [](const Matrix &a) -> std::string {
            if (*implicit_conversion)
                return "";
            std::ostringstream oss;
            oss << a;
            return oss.str();
        })
        .def("__getitem__", [](const Matrix &a, std::pair<size_t, size_t> index) {
            if (index.first >= Matrix::Size || index.second >= Matrix::Size)
                throw py::index_error();
            return a.coeff(index.second, index.first);
        })
        .def("__setitem__", [](Matrix &a, std::pair<size_t, size_t> index, const Value &value) {
            if (index.first >= Matrix::Size || index.second >= Matrix::Size)
                throw py::index_error();
            a.coeff(index.second, index.first) = value;
        })
        .def("__setitem__", [](Matrix &a, const mask_t<Value> &m, const Matrix &b) {
            a[m] = b;
        })
        .def_static("identity", [](size_t size) { return identity<Matrix>(size); }, "size"_a = 1)
        .def_static("zero", [](size_t size) { return zero<Matrix>(size); }, "size"_a = 1);

    cls.def(py::init([](const py::ndarray &obj) -> Matrix {
               using T = expr_t<decltype(detach(std::declval<Matrix &>()))>;
               return numpy_to_enoki<T>(obj);
           }))
      .def("numpy", [](const Matrix &a, bool eval) { return enoki_to_numpy(detach(a), eval); },
           "eval"_a = true)
      .def_property_readonly("__array_interface__", [](const py::object &o) {
          py::object np_array = o.attr("numpy")();
          py::dict result;
          result["data"]    = np_array;
          result["shape"]   = np_array.attr("shape");
          result["version"] = 3;
          char typestr[4]   = { '<', 0, '0' + sizeof(Scalar), 0 };
          if (std::is_floating_point_v<Scalar>)
              typestr[1] = 'f';
          else if (std::is_same_v<Scalar, bool>)
              typestr[1] = 'b';
          else if (std::is_unsigned_v<Scalar>)
              typestr[1] = 'u';
          else
              typestr[1] = 'i';
          result["typestr"] = typestr;
          return result;
      });

    cls.def(py::init([](const py::torch_tensor &obj) -> Matrix {
        using T = expr_t<decltype(detach(std::declval<Matrix&>()))>;
        return torch_to_enoki<T>(obj);
    }))
    .def("torch", [](const Matrix &a, bool eval) {
        return enoki_to_torch(detach(a), eval); },
        "eval"_a = true
    );

    m.def("shape", [](const Matrix &a) { return shape(a); });
    m.def("slices", [](const Matrix &a) { return slices(a); });
    m.def("set_slices", [](Matrix &a, size_t size) { set_slices(a, size); }, "array"_a, "slices"_a);

    cls.def("__matmul__",
            [](const Matrix &a, const Matrix &b) { return a * b; });

    if constexpr (array_depth_v<Value> == (IsDynamic ? 1 : 0))
        cls.def("__matmul__",
                [](const Matrix &a, const Vector &b) { return a * b; });

    if constexpr (Matrix::Size == 2) {
        cls.def(py::init<const Value &, const Value &,
                         const Value &, const Value &>());
        if constexpr (array_depth_v<Value> == (IsDynamic ? 1 : 0))
            cls.def(py::init<const Vector &, const Vector &>());
    } else if constexpr (Matrix::Size == 3) {
        cls.def(py::init<const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &>());
        if constexpr (array_depth_v<Value> == (IsDynamic ? 1 : 0))
            cls.def(py::init<const Vector &, const Vector &, const Vector &>());
    } else if constexpr (Matrix::Size == 4) {
        using Vector3 = enoki::Array<Value, 3>;

        cls.def(py::init<const Value &, const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &, const Value &>());

        if constexpr (array_depth_v<Value> == (IsDynamic ? 1 : 0)) {
            cls.def(py::init<const Vector &, const Vector &, const Vector &, const Vector &>())
               .def_static("translate", [](const Vector3 &v) { return translate<Matrix>(v); })
               .def_static("scale", [](const Vector3 &v) { return scale<Matrix>(v); })
               .def_static("rotate", [](const Vector3 &axis, const Value &angle) {
                       return rotate<Matrix>(axis, angle);
                   },
                   "axis"_a, "angle"_a
               )
               .def_static("look_at", [](const Vector3 &origin,
                                         const Vector3 &target,
                                         const Vector3 &up) {
                       return look_at<Matrix>(origin, target, up);
                   },
                   "origin"_a, "target"_a, "up"_a
               );
        }
    }

    auto transpose_m = [](const Matrix &m) { return transpose(m); };
    cls.def_property_readonly("T", transpose_m);

    m.def("transpose", transpose_m);
    m.def("det", [](const Matrix &m) { return det(m); });
    m.def("inverse", [](const Matrix &m) { return inverse(m); });
    m.def("inverse_transpose", [](const Matrix &m) { return inverse_transpose(m); });

    if constexpr (is_diff_array_v<Matrix>) {
        using Detached = expr_t<decltype(detach(std::declval<Matrix&>()))>;

        m.def("detach", [](const Matrix &a) -> Detached { return detach(a); });
        m.def("requires_gradient",
              [](const Matrix &a) { return requires_gradient(a); },
              "array"_a);

        m.def("set_requires_gradient",
              [](Matrix &a, bool value) { set_requires_gradient(a, value); },
              "array"_a, "value"_a = true);

        m.def("gradient", [](Matrix &a) { return eval(gradient(a)); });
        m.def("set_gradient",
              [](Matrix &a, const Detached &g, bool b) { set_gradient(a, g, b); },
              "array"_a, "gradient"_a, "backward"_a = true);

        m.def("graphviz", [](const Matrix &a) { return graphviz(a); });

        m.def("set_label", [](const Matrix &a, const char *label) {
            set_label(a, label);
        });
    }

    return cls;
}
