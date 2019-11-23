#include <enoki/cuda.h>
#include <enoki/autodiff.h>
#include <enoki/matrix.h>
#include <enoki/transform.h>
#include <enoki/special.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sstream>

using namespace enoki;
namespace py = pybind11;
using namespace py::literals;

using Float     = float;
using FloatC    = CUDAArray<Float>;
using UInt32C   = CUDAArray<uint32_t>;
using UInt64C   = CUDAArray<uint64_t>;
using BoolC     = CUDAArray<bool>;

using FloatD    = DiffArray<FloatC>;
using UInt32D   = DiffArray<UInt32C>;
using UInt64D   = DiffArray<UInt64C>;
using BoolD     = DiffArray<BoolC>;

using Vector2fC = Array<FloatC, 2>;
using Vector2fD = Array<FloatD, 2>;
using Vector2uC = Array<UInt32C, 2>;
using Vector2uD = Array<UInt32D, 2>;
using Vector2bC = mask_t<Vector2fC>;
using Vector2bD = mask_t<Vector2fD>;

using Vector3fC = Array<FloatC, 3>;
using Vector3fD = Array<FloatD, 3>;
using Vector3uC = Array<UInt32C, 3>;
using Vector3uD = Array<UInt32D, 3>;
using Vector3bC = mask_t<Vector3fC>;
using Vector3bD = mask_t<Vector3fD>;

using Vector4fC = Array<FloatC, 4>;
using Vector4fD = Array<FloatD, 4>;
using Vector4uC = Array<UInt32C, 4>;
using Vector4uD = Array<UInt32D, 4>;
using Vector4bC = mask_t<Vector4fC>;
using Vector4bD = mask_t<Vector4fD>;

using Matrix4fC = Matrix<FloatC, 4>;
using Matrix4fD = Matrix<FloatD, 4>;

struct CUDAManagedBuffer {
    CUDAManagedBuffer(size_t size) {
        ptr = cuda_managed_malloc(size);
    }

    ~CUDAManagedBuffer() {
        cuda_free(ptr);
    }

    void *ptr = nullptr;
};


namespace enoki {
extern ENOKI_IMPORT uint32_t cuda_var_copy_to_device(EnokiType type,
                                                     size_t size, const void *value);
};

template <typename Array> py::object enoki_to_torch(const Array &array, bool eval);
template <typename Array> py::object enoki_to_numpy(const Array &array, bool eval);
template <typename Array> Array torch_to_enoki(py::object src);
template <typename Array> Array numpy_to_enoki(py::array src);

extern bool disable_print_flag;

/// Customized version of pybind11::implicitly_convertible() which disables
/// __repr__ during implicit casts (this can be triggered at implicit cast
/// failures and causes a totally unnecessary/undesired cuda_eval() invocation)
template <typename InputType, typename OutputType> void implicitly_convertible() {
    struct set_flag {
        bool &flag, backup;
        set_flag(bool &flag) : flag(flag), backup(flag) { flag = true; }
        ~set_flag() { flag = backup; }
    };

    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject * {
        static bool currently_used = false;
        if (currently_used) // implicit conversions are non-reentrant
            return nullptr;
        set_flag flag_helper(currently_used);
        set_flag flag_helper_2(disable_print_flag);
        if (!py::detail::make_caster<InputType>().load(obj, false))
            return nullptr;
        py::tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call((PyObject *) type, args.ptr(), nullptr);
        if (result == nullptr)
            PyErr_Clear();
        return result;
    };

    if (auto tinfo = py::detail::get_type_info(typeid(OutputType)))
        tinfo->implicit_conversions.push_back(implicit_caster);
    else
        py::pybind11_fail("implicitly_convertible: Unable to find type " + pybind11::type_id<OutputType>());
}

template <typename Array>
py::class_<Array> bind(py::module &m, const char *name) {
    using Scalar = scalar_t<Array>;
    using Value  = value_t<Array>;

    static constexpr bool IsMask  = std::is_same_v<Scalar, bool>;
    static constexpr bool IsFloat = std::is_floating_point_v<Scalar>;

    py::class_<Array> cl(m, name);

    cl.def(py::init<>())
      .def(py::init<const Array &>())
      .def(py::init<const Value &>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__repr__", [](const Array &a) -> std::string {
          if (disable_print_flag)
              return "";
          std::ostringstream oss;
          oss << a;
          return oss.str();
      })
      .def_static("zero", [](size_t size) { return zero<Array>(size); },
                  "size"_a = 1)
      .def_static("empty", [](size_t size) { return empty<Array>(size); },
                  "size"_a = 1);

    if constexpr (array_depth_v<Array> == 1 && is_cuda_array_v<Array>) {
        cl.def(
            py::init([](Scalar scalar, bool literal) -> Array {
                if (literal) {
                    return Array(scalar);
                } else {
                    if constexpr (is_diff_array_v<Array>)
                        return Array::UnderlyingType::copy(&scalar, 1);
                    else
                        return Array::copy(&scalar, 1);
                }
            }),
            "scalar"_a, "literal"_a = true
        );
    }

    cl.def(py::init([](const py::object &obj) -> Array {
          const char *tp_name = ((PyTypeObject *) obj.get_type().ptr())->tp_name;
          if (strstr(tp_name, "Tensor") != nullptr) {
              using T = expr_t<decltype(detach(std::declval<Array&>()))>;
              return torch_to_enoki<T>(obj);
          }

          if (strstr(tp_name, "numpy.ndarray") != nullptr) {
              using T = expr_t<decltype(detach(std::declval<Array&>()))>;
              return numpy_to_enoki<T>(obj);
          }

          if constexpr (!IsMask && array_depth_v<Array> == 1) {
              if (strstr(tp_name, "enoki.") == nullptr && py::isinstance<py::sequence>(obj)) {
                  try {
                      auto a = py::cast<std::vector<Value>>(obj);
                      if constexpr (!is_diff_array_v<Array>) {
                          uint32_t index = cuda_var_copy_to_device(Array::Type, a.size(), a.data());
                          return Array::from_index_(index);
                      } else {
                          uint32_t index = cuda_var_copy_to_device(Array::UnderlyingType::Type, a.size(), a.data());
                          return Array::UnderlyingType::from_index_(index);
                      }
                  } catch (...) { }
              }
          }

          throw py::reference_cast_error();
      }))
      .def("torch", [](const Array &a, bool eval) {
          return enoki_to_torch(detach(a), eval); },
          "eval"_a = true
      )
      .def("numpy", [](const Array &a, bool eval) {
          return enoki_to_numpy(detach(a), eval); },
          "eval"_a = true
      )
      .def_property_readonly("__array_interface__", [](const py::object &o) {
          py::object np_array = o.attr("numpy")();
          py::dict result;
          result["data"] = np_array;
          result["shape"] = np_array.attr("shape");
          result["version"] = 3;
          char typestr[4] = { '<', 0, '0' + sizeof(Scalar), 0 };
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

    if constexpr (!IsMask) {
        cl.def(py::self + py::self)
          .def(Value() + py::self)
          .def(py::self - py::self)
          .def(Value() - py::self)
          .def(py::self / py::self)
          .def(Value() / py::self)
          .def(py::self * py::self)
          .def(Value() * py::self)
          .def(py::self < py::self)
          .def(Value() < py::self)
          .def(py::self > py::self)
          .def(Value() > py::self)
          .def(py::self >= py::self)
          .def(Value() >= py::self)
          .def(py::self <= py::self)
          .def(Value() <= py::self)
          .def(-py::self);

        if constexpr (std::is_integral_v<Scalar>) {
            cl.def(py::self % py::self);
            cl.def("__truediv__", [](const Array &a1, Scalar a2) {
                return (a2 == 1) ? a1 : (a1 / a2);
            });
            cl.def("__mod__", [](const Array &a1, Scalar a2) {
                return (a2 == 1) ? 1 : (a1 % a2);
            });
        }
    } else {
        cl.def(py::self | py::self)
          .def(py::self & py::self)
          .def(py::self ^ py::self)
          .def(!py::self)
          .def(~py::self);
    }

    if constexpr (!IsMask && array_depth_v<Array> == 1) {
        if (IsFloat)
            cl.def_static("linspace",
                  [](Scalar min, Scalar max, size_t size) {
                      return linspace<Array>(min, max, size);
                  },
                  "min"_a, "max"_a, "size"_a);

        cl.def_static("arange",
              [](size_t size) { return arange<Array>(size); }, "size"_a);
    }

    cl.def_static("full",
                  [](const Scalar &value, size_t size) {
                      Array result(value);
                      set_slices(result, size);
                      return result;
                  }, "value"_a, "size"_a);

    cl.def("__getitem__", [](const Array &a, size_t index) -> Value {
        if (index >= a.size())
            throw py::index_error();
        return a.coeff(index);
    });

    cl.def("__len__", [](const Array &a) { return a.size(); });
    cl.def("resize", [](Array &a, size_t size) { enoki::set_slices(a, size); });
    m.def("slices", [](const Array &a) { return slices(a); });

    if constexpr (array_depth_v<Array> > 1) {
        cl.def("__setitem__", [](Array &a, size_t index, const Value &b) {
            if (index >= Array::Size)
                throw py::index_error();
            a.coeff(index) = b;
        });

        if constexpr (array_size_v<Array> == 2)
            cl.def(py::init<Value, Value>());
        else if constexpr (array_size_v<Array> == 3)
            cl.def(py::init<Value, Value, Value>());
        else if constexpr (array_size_v<Array> == 4)
            cl.def(py::init<Value, Value, Value, Value>());

        cl.def(py::init([](const std::array<Value, Array::Size> &a) {
            Array result;
            for (size_t i = 0; i<Array::Size; ++i)
                result.coeff(i) = a[i];
            return result;
        }));

        if constexpr (array_size_v<Array> >= 1)
            cl.def_property("x", [](const Array &a) { return a.x(); },
                                 [](Array &a, const Value &v) { a.x() = v; });
        if constexpr (array_size_v<Array> >= 2)
            cl.def_property("y", [](const Array &a) { return a.y(); },
                                 [](Array &a, const Value &v) { a.y() = v; });
        if constexpr (array_size_v<Array> >= 3)
            cl.def_property("z", [](const Array &a) { return a.z(); },
                                 [](Array &a, const Value &v) { a.z() = v; });
        if constexpr (array_size_v<Array> >= 4)
            cl.def_property("w", [](const Array &a) { return a.w(); },
                                 [](Array &a, const Value &v) { a.w() = v; });

        if constexpr (!IsMask) {
            m.def("dot", [](const Array &a, const Array &b) { return enoki::dot(a, b); });
            m.def("abs_dot", [](const Array &a, const Array &b) { return enoki::abs_dot(a, b); });
            m.def("normalize", [](const Array &a) { return enoki::normalize(a); });
            m.def("squared_norm", [](const Array &a) { return enoki::squared_norm(a); });
            m.def("norm", [](const Array &a) { return enoki::norm(a); });

            if constexpr (array_size_v<Array> == 3)
                m.def("cross", [](const Array &a, const Array &b) { return enoki::cross(a, b); });
        }
    } else {
        cl.def_property_readonly("index", [](const Array &a) { return a.index_(); });
        cl.def_property_readonly("data", [](const Array &a) { return (uintptr_t) a.data(); });
    }

    if constexpr (!is_diff_array_v<Array>) {
        m.def("compress", [](const Array &array, const mask_t<Array> &mask) {
            return compress(array, mask);
        });
    }

    using Index = uint32_array_t<
        std::conditional_t<array_depth_v<Array> == 1, Array, value_t<Array>>>;
    m.def("gather",
          [](const Array &source, const Index &index, mask_t<Index> &mask) {
              return gather<Array>(source, index, mask);
          },
          "source"_a, "index"_a, "mask"_a = true);

    m.def("scatter",
          [](Array &target, const Array &source,
             const Index &index,
             mask_t<Index> &mask) { scatter(target, source, index, mask); },
          "target"_a, "source"_a, "index"_a, "mask"_a = true);

    m.def("scatter_add",
          [](Array &target, const Array &source,
             const Index &index,
             mask_t<Index> &mask) { scatter_add(target, source, index, mask); },
          "target"_a, "source"_a, "index"_a, "mask"_a = true);

    if constexpr (IsFloat) {
        m.def("abs", [](const Array &a) { return enoki::abs(a); });
        m.def("sqr", [](const Array &a) { return enoki::sqr(a); });
        m.def("sqrt", [](const Array &a) { return enoki::sqrt(a); });
        m.def("cbrt", [](const Array &a) { return enoki::cbrt(a); });
        m.def("rcp", [](const Array &a) { return enoki::rcp(a); });
        m.def("rsqrt", [](const Array &a) { return enoki::rsqrt(a); });

        m.def("ceil", [](const Array &a) { return enoki::ceil(a); });
        m.def("floor", [](const Array &a) { return enoki::floor(a); });
        m.def("round", [](const Array &a) { return enoki::round(a); });
        m.def("trunc", [](const Array &a) { return enoki::trunc(a); });

        m.def("sin", [](const Array &a) { return enoki::sin(a); });
        m.def("cos", [](const Array &a) { return enoki::cos(a); });
        m.def("sincos", [](const Array &a) { return enoki::sincos(a); });
        m.def("tan", [](const Array &a) { return enoki::tan(a); });
        m.def("sec", [](const Array &a) { return enoki::sec(a); });
        m.def("csc", [](const Array &a) { return enoki::csc(a); });
        m.def("cot", [](const Array &a) { return enoki::cot(a); });
        m.def("asin", [](const Array &a) { return enoki::asin(a); });
        m.def("acos", [](const Array &a) { return enoki::acos(a); });
        m.def("atan", [](const Array &a) { return enoki::atan(a); });
        m.def("atan2", [](const Array &a, const Array &b) {
            return enoki::atan2(a, b);
        });

        m.def("sinh", [](const Array &a) { return enoki::sinh(a); });
        m.def("cosh", [](const Array &a) { return enoki::cosh(a); });
        m.def("sincosh", [](const Array &a) { return enoki::sincosh(a); });
        m.def("tanh", [](const Array &a) { return enoki::tanh(a); });
        m.def("sech", [](const Array &a) { return enoki::sech(a); });
        m.def("csch", [](const Array &a) { return enoki::csch(a); });
        m.def("coth", [](const Array &a) { return enoki::coth(a); });
        m.def("asinh", [](const Array &a) { return enoki::asinh(a); });
        m.def("acosh", [](const Array &a) { return enoki::acosh(a); });
        m.def("atanh", [](const Array &a) { return enoki::atanh(a); });

        m.def("log", [](const Array &a) { return enoki::log(a); });
        m.def("exp", [](const Array &a) { return enoki::exp(a); });
        m.def("erfinv", [](const Array &a) { return enoki::erfinv(a); });
        m.def("erf", [](const Array &a) { return enoki::erf(a); });
        m.def("pow", [](const Array &a, const Array &b) {
            return enoki::pow(a, b);
        });

        m.def("lerp", [](const Array &a, const Array &b, const Array &t) {
            return enoki::lerp(a, b, t);
        });
        m.def("clamp", [](const Array &value, const Array &min, const Array &max) {
            return enoki::clamp(value, min, max);
        });

        m.def("isfinite", [](const Array &a) { return enoki::isfinite(a); });
        m.def("isnan", [](const Array &a) { return enoki::isnan(a); });
        m.def("isinf", [](const Array &a) { return enoki::isinf(a); });
    } else if constexpr (!IsMask) {
        m.def("popcnt", [](const Array &a) { return enoki::popcnt(a); });
        m.def("lzcnt", [](const Array &a) { return enoki::lzcnt(a); });
        m.def("tzcnt", [](const Array &a) { return enoki::tzcnt(a); });
        m.def("mulhi", [](const Array &a, const Array &b) { return enoki::mulhi(a, b); });
    }

    if constexpr (!IsMask) {
        m.def("max", [](const Array &a, const Array &b) { return enoki::max(a, b); });
        m.def("min", [](const Array &a, const Array &b) { return enoki::min(a, b); });

        m.def("hsum", [](const Array &a) { return enoki::hsum(a); });
        m.def("hprod", [](const Array &a) { return enoki::hprod(a); });
        m.def("hmin", [](const Array &a) { return enoki::hmin(a); });
        m.def("hmax", [](const Array &a) { return enoki::hmax(a); });
        m.def("hmean", [](const Array &a) { return enoki::hmean(a); });

        m.def("fmadd", [](const Array &a, const Array &b, const Array &c) {
            return enoki::fmadd(a, b, c);
        });
        m.def("fmsub", [](const Array &a, const Array &b, const Array &c) {
            return enoki::fmsub(a, b, c);
        });
        m.def("fnmadd", [](const Array &a, const Array &b, const Array &c) {
            return enoki::fnmadd(a, b, c);
        });
        m.def("fnmsub", [](const Array &a, const Array &b, const Array &c) {
            return enoki::fnmsub(a, b, c);
        });
    } else {
        m.def("any", [](const Array &a) { return enoki::any(a); });
        m.def("none", [](const Array &a) { return enoki::none(a); });
        m.def("all", [](const Array &a) { return enoki::all(a); });
        m.def("count", [](const Array &a) { return enoki::count(a); });

        if constexpr (array_depth_v<Array> > 1) {
            m.def("any_nested", [](const Array &a) { return enoki::any_nested(a); });
            m.def("none_nested", [](const Array &a) { return enoki::none_nested(a); });
            m.def("all_nested", [](const Array &a) { return enoki::all_nested(a); });
        }
    }

    m.def("eq", [](const Array &a, const Array &b) { return eq(a, b); });
    m.def("neq", [](const Array &a, const Array &b) { return neq(a, b); });

    m.def("select", [](const mask_t<Array> &a, const Array &b, const Array &c) {
        return enoki::select(a, b, c);
    });

    if constexpr (is_diff_array_v<Array>) {
        m.def("detach", [](const Array &a) { return eval(detach(a)); });
        m.def("reattach", [](Array &a, Array &b) { reattach(a, b); });
    }

    if constexpr (IsFloat && is_diff_array_v<Array>) {
        using Detached = decltype(eval(detach(std::declval<Array&>())));

        m.def("requires_gradient",
              [](const Array &a) { return requires_gradient(a); },
              "array"_a);

        m.def("set_requires_gradient",
              [](Array &a, bool value) { set_requires_gradient(a, value); },
              "array"_a, "value"_a = true);

        m.def("gradient", [](Array &a) -> Detached { return gradient(a); });
        m.def("gradient_index", [](Array &a) { return gradient_index(a); });

        m.def("set_gradient",
              [](Array &a, const Detached &g, bool backward) { set_gradient(a, g, backward); },
              "array"_a, "gradient"_a, "backward"_a = true);

        m.def("graphviz", [](const Array &a) { return graphviz(a); });

        if constexpr (array_depth_v<Array> == 1) {
            m.def("backward",
                  [](Array &a, bool free_graph) { backward(a, free_graph); },
                  "array"_a, "free_graph"_a = true);
            m.def("forward",
                  [](Array &a, bool free_graph) { return forward(a, free_graph); },
                  "array"_a, "free_graph"_a = true);
        }
    }

    if constexpr (is_diff_array_v<Array>) {
        using BaseType = expr_t<decltype(detach(std::declval<Array&>()))>;
        implicitly_convertible<BaseType, Array>();
    }

    m.def("set_label", [](const Array &a, const char *label) {
        set_label(a, label);
    });

    implicitly_convertible<Value, Array>();

    if constexpr (IsFloat)
        implicitly_convertible<int, Array>();

    if constexpr (!std::is_same_v<Value, Scalar>)
        implicitly_convertible<Scalar, Array>();

    return cl;
}

template <typename Matrix>
py::class_<Matrix> bind_matrix(py::module &m, const char *name) {
    using Vector = typename Matrix::Column;
    using Value  = typename Matrix::Entry;
    using Vector3 = Array<Value, 3>;

    auto cls = py::class_<Matrix>(m, name)
        .def(py::init<>())
        .def(py::init<const Value &>())
        .def(py::init<const Vector &, const Vector &, const Vector &, const Vector &>())
        .def(py::init<const Value &, const Value &, const Value &, const Value &,
                      const Value &, const Value &, const Value &, const Value &,
                      const Value &, const Value &, const Value &, const Value &,
                      const Value &, const Value &, const Value &, const Value &>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self - py::self)
        .def(py::self + py::self)
        .def(py::self * py::self)
        .def(py::self * Vector())
        .def(py::self * Value())
        .def(-py::self)
        .def("__repr__", [](const Matrix &a) -> std::string {
            if (disable_print_flag)
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
        .def_static("identity", [](size_t size) { return identity<Matrix>(size); }, "size"_a = 1)
        .def_static("zero", [](size_t size) { return zero<Matrix>(size); }, "size"_a = 1)
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

    m.def("transpose", [](const Matrix &m) { return transpose(m); });
    m.def("det", [](const Matrix &m) { return det(m); });
    m.def("inverse", [](const Matrix &m) { return inverse(m); });
    m.def("inverse_transpose", [](const Matrix &m) { return inverse_transpose(m); });
    return cls;
}

template <typename Scalar> py::object torch_dtype() {
    py::object torch = py::module::import("torch");
    const char *name = nullptr;

    if (std::is_same_v<Scalar, enoki::half>) {
        name = "float16";
    } else if (std::is_same_v<Scalar, float>) {
        name = "float32";
    } else if (std::is_same_v<Scalar, double>) {
        name = "float64";
    } else if (std::is_integral_v<Scalar>) {
        if (sizeof(Scalar) == 1)
            name = std::is_signed_v<Scalar> ? "int8" : "uint8";
        else if (sizeof(Scalar) == 2)
            name = "int16";
        else if (sizeof(Scalar) == 4)
            name = "int32";
        else if (sizeof(Scalar) == 8)
            name = "int64";
    }

    if (name == nullptr)
        throw std::runtime_error("pytorch_dtype(): Unsupported type");

    return torch.attr(name);
}

template <size_t Index, size_t Dim, typename Source, typename Target>
static void copy_array_gather(size_t offset,
                              const std::array<size_t, Dim> &shape,
                              const std::array<size_t, Dim> &strides,
                              const Source &source, Target &target) {
    using namespace enoki;
    if constexpr (Index == Dim - 1) {
        using UInt32 = uint32_array_t<Source>;
        UInt32 index = fmadd(arange<UInt32>((uint32_t) shape[Index]),
                             (uint32_t) strides[Index], (uint32_t) offset);
        target = gather<Target>(source, index);
    } else {
        const size_t step = strides[Index];
        for (size_t i = 0; i < shape[Index]; ++i) {
            copy_array_gather<Index + 1, Dim>(offset, shape, strides, source,
                                              target.coeff(i));
            offset += step;
        }
    }
}

template <size_t Index, size_t Dim, typename Source, typename Target>
static void copy_array_scatter(size_t offset,
                               const std::array<size_t, Dim> &shape,
                               const std::array<size_t, Dim> &strides,
                               const Source &source, Target &target) {
    using namespace enoki;
    if constexpr (Index == Dim - 1) {
        using UInt32 = uint32_array_t<Source>;
        UInt32 index = fmadd(arange<UInt32>((uint32_t) shape[Index]),
                             (uint32_t) strides[Index], (uint32_t) offset);
        scatter(target, source, index);
    } else {
        const size_t step = strides[Index];
        for (size_t i = 0; i < shape[Index]; ++i) {
            copy_array_scatter<Index + 1, Dim>(offset, shape, strides,
                                               source.coeff(i), target);
            offset += step;
        }
    }
}

template <typename Array>
ENOKI_NOINLINE py::object enoki_to_torch(const Array &src, bool eval) {
    constexpr size_t Depth = array_depth_v<Array>;
    using Scalar = scalar_t<Array>;

    std::array<size_t, Depth> shape = enoki::shape(src),
                              shape_rev = shape,
                              strides;
    std::reverse(shape_rev.begin(), shape_rev.end());

    py::object torch = py::module::import("torch");
    py::object dtype_obj = torch_dtype<Scalar>();

    py::object result = torch.attr("empty")(
        py::cast(shape_rev),
        "dtype"_a = dtype_obj,
        "device"_a = "cuda");

    size_t size = 1;
    for (size_t i : shape)
        size *= i;

    strides = py::cast<std::array<size_t, Depth>>(result.attr("stride")());
    std::reverse(strides.begin(), strides.end());
    if (size > 0) {
        CUDAArray<Scalar> target = CUDAArray<Scalar>::map(
            (Scalar *) py::cast<uintptr_t>(result.attr("data_ptr")()), size);
        copy_array_scatter<0>(0, shape, strides, src, target);
        if (eval) {
            cuda_eval();
            cuda_sync();
        }
    }
    return result;
}

template <typename Array>
ENOKI_NOINLINE Array torch_to_enoki(py::object src) {
    constexpr size_t Depth = array_depth_v<Array>;
    using Scalar = scalar_t<Array>;

    py::tuple shape_obj = src.attr("shape");
    py::object dtype_obj = src.attr("dtype");
    py::object target_dtype = torch_dtype<Scalar>();
    if (((std::string) ((py::str) src.attr("device"))).find("cuda") == std::string::npos)
        throw std::runtime_error("Attempted to cast a Torch CPU tensor to a Enoki GPU array!");

    if (shape_obj.size() != Depth || !dtype_obj.is(target_dtype))
        throw py::reference_cast_error();

    auto shape = py::cast<std::array<size_t, Depth>>(shape_obj);
    auto strides = py::cast<std::array<size_t, Depth>>(src.attr("stride")());
    std::reverse(shape.begin(), shape.end());
    std::reverse(strides.begin(), strides.end());

    size_t size = 1;
    for (size_t i : shape)
        size *= i;

    Array result;

    if (size > 0) {
        CUDAArray<Scalar> source = CUDAArray<Scalar>::map(
            (Scalar *) py::cast<uintptr_t>(src.attr("data_ptr")()), size);

        copy_array_gather<0>(0, shape, strides, source, result);
    }
    return result;
}

template <typename Array>
ENOKI_NOINLINE py::object enoki_to_numpy(const Array &src, bool eval) {
    constexpr size_t Depth = array_depth_v<Array>;
    using Scalar = scalar_t<Array>;

    std::array<size_t, Depth> shape = enoki::shape(src),
                              shape_rev = shape, strides;
    std::reverse(shape_rev.begin(), shape_rev.end());

    size_t size = 1, stride = sizeof(Scalar);
    for (ssize_t i = (ssize_t) Depth - 1; i >= 0; --i) {
        size *= shape_rev[i];
        strides[i] = stride;
        stride *= shape_rev[i];
    }

    CUDAManagedBuffer *buf = new CUDAManagedBuffer(stride);
    py::object buf_py = py::cast(buf, py::return_value_policy::take_ownership);

    py::array result(py::dtype::of<Scalar>(), shape_rev, strides, buf->ptr, buf_py);
    for (ssize_t i = (ssize_t) Depth - 1; i >= 0; --i)
        strides[i] /= sizeof(Scalar);
    std::reverse(strides.begin(), strides.end());

    if (size > 0) {
        CUDAArray<Scalar> target = CUDAArray<Scalar>::map(buf->ptr, stride);
        copy_array_scatter<0>(0, shape, strides, src, target);
        if (eval) {
            cuda_eval();
            cuda_sync();
        }
    }

    return result;
}

template <typename Array>
ENOKI_NOINLINE Array numpy_to_enoki(py::array src) {
    constexpr size_t Depth = array_depth_v<Array>;
    using Scalar = scalar_t<Array>;

    py::tuple shape_obj = src.attr("shape");
    py::object dtype_obj = src.attr("dtype");
    py::object target_dtype = py::dtype::of<Scalar>();

    if (shape_obj.size() != Depth)
        throw py::reference_cast_error();

    if (!dtype_obj.is(target_dtype))
        src = py::array_t<Scalar, py::array::forcecast>::ensure(src);

    auto shape = py::cast<std::array<size_t, Depth>>(shape_obj);
    auto strides = py::cast<std::array<size_t, Depth>>(src.attr("strides"));
    std::reverse(shape.begin(), shape.end());
    std::reverse(strides.begin(), strides.end());

    size_t size = 1;
    for (size_t i : shape)
        size *= i;

    for (ssize_t i = 0; i < Depth; ++i)
        strides[i] /= sizeof(Scalar);

    Array result;

    if (size > 0) {
        CUDAArray<Scalar> source = CUDAArray<Scalar>::from_index_(
            cuda_var_copy_to_device(enoki_type_v<Scalar>, size, src.data()));

        copy_array_gather<0>(0, shape, strides, source, result);
    }

    return result;
}
