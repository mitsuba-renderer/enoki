#include <enoki/cuda.h>
#include <enoki/dynamic.h>
#include <enoki/autodiff.h>
#include <enoki/complex.h>
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

using Float  = float;
using Int32  = int32_t;
using UInt32 = uint32_t;
using Int64  = int64_t;
using UInt64 = uint64_t;

constexpr size_t PacketSize = array_default_size<Float>;

using FloatX    = DynamicArray<Packet<Float, PacketSize>>;
using Int32X    = DynamicArray<Packet<Int32, PacketSize>>;
using Int64X    = DynamicArray<Packet<Int64, PacketSize>>;
using UInt32X   = DynamicArray<Packet<UInt32, PacketSize>>;
using UInt64X   = DynamicArray<Packet<UInt64, PacketSize>>;
using MaskX     = mask_t<FloatX>;

using FloatC    = CUDAArray<Float>;
using Int32C    = CUDAArray<Int32>;
using Int64C    = CUDAArray<Int64>;
using UInt32C   = CUDAArray<UInt32>;
using UInt64C   = CUDAArray<UInt64>;
using MaskC     = mask_t<FloatC>;

using FloatD    = DiffArray<FloatC>;
using Int32D    = DiffArray<Int32C>;
using Int64D    = DiffArray<Int64C>;
using UInt32D   = DiffArray<UInt32C>;
using UInt64D   = DiffArray<UInt64C>;
using MaskD     = mask_t<FloatD>;

using Vector2f  = Array<Float , 2>;
using Vector2fX = Array<FloatX, 2>;
using Vector2fC = Array<FloatC, 2>;
using Vector2fD = Array<FloatD, 2>;

using Vector2i  = Array<Int32 , 2>;
using Vector2iX = Array<Int32X, 2>;
using Vector2iC = Array<Int32C, 2>;
using Vector2iD = Array<Int32D, 2>;

using Vector2u  = Array<UInt32 , 2>;
using Vector2uX = Array<UInt32X, 2>;
using Vector2uC = Array<UInt32C, 2>;
using Vector2uD = Array<UInt32D, 2>;

using Vector2m  = mask_t<Vector2f >;
using Vector2mX = mask_t<Vector2fX>;
using Vector2mC = mask_t<Vector2fC>;
using Vector2mD = mask_t<Vector2fD>;

using Vector3f  = Array<Float , 3>;
using Vector3fX = Array<FloatX, 3>;
using Vector3fC = Array<FloatC, 3>;
using Vector3fD = Array<FloatD, 3>;

using Vector3i  = Array<Int32 , 3>;
using Vector3iX = Array<Int32X, 3>;
using Vector3iC = Array<Int32C, 3>;
using Vector3iD = Array<Int32D, 3>;

using Vector3u  = Array<UInt32 , 3>;
using Vector3uX = Array<UInt32X, 3>;
using Vector3uC = Array<UInt32C, 3>;
using Vector3uD = Array<UInt32D, 3>;

using Vector3m  = mask_t<Vector3f >;
using Vector3mX = mask_t<Vector3fX>;
using Vector3mC = mask_t<Vector3fC>;
using Vector3mD = mask_t<Vector3fD>;

using Vector4f  = Array<Float , 4>;
using Vector4fX = Array<FloatX, 4>;
using Vector4fC = Array<FloatC, 4>;
using Vector4fD = Array<FloatD, 4>;

using Vector4i  = Array<Int32 , 4>;
using Vector4iX = Array<Int32X, 4>;
using Vector4iC = Array<Int32C, 4>;
using Vector4iD = Array<Int32D, 4>;

using Vector4u  = Array<UInt32 , 4>;
using Vector4uX = Array<UInt32X, 4>;
using Vector4uC = Array<UInt32C, 4>;
using Vector4uD = Array<UInt32D, 4>;

using Vector4m  = mask_t<Vector4f >;
using Vector4mX = mask_t<Vector4fX>;
using Vector4mC = mask_t<Vector4fC>;
using Vector4mD = mask_t<Vector4fD>;

using Complex2f  = Complex<Float >;
using Complex2fX = Complex<FloatX>;
using Complex2fC = Complex<FloatC>;
using Complex2fD = Complex<FloatD>;

using Complex24f  = Complex<Vector4f >;
using Complex24fX = Complex<Vector4fX>;
using Complex24fC = Complex<Vector4fC>;
using Complex24fD = Complex<Vector4fD>;

using Matrix2f  = Matrix<Float , 2>;
using Matrix2fX = Matrix<FloatX, 2>;
using Matrix2fC = Matrix<FloatC, 2>;
using Matrix2fD = Matrix<FloatD, 2>;

using Matrix3f  = Matrix<Float , 3>;
using Matrix3fX = Matrix<FloatX, 3>;
using Matrix3fC = Matrix<FloatC, 3>;
using Matrix3fD = Matrix<FloatD, 3>;

using Matrix4f  = Matrix<Float , 4>;
using Matrix4fX = Matrix<FloatX, 4>;
using Matrix4fC = Matrix<FloatC, 4>;
using Matrix4fD = Matrix<FloatD, 4>;

using Matrix44f  = Matrix<Vector4f , 4>;
using Matrix44fX = Matrix<Vector4fX, 4>;
using Matrix44fC = Matrix<Vector4fC, 4>;
using Matrix44fD = Matrix<Vector4fD, 4>;

struct Buffer {
    Buffer(size_t size, bool cuda_managed)
        : cuda_managed(cuda_managed) {
#if defined(ENOKI_CUDA)
        if (cuda_managed) {
            ptr = cuda_managed_malloc(size);
            return;
        }
#endif
        ptr = std::aligned_alloc(64, size);
    }

    ~Buffer() {
#if defined(ENOKI_CUDA)
        if (cuda_managed) {
            cuda_free(ptr);
            return;
        }
#endif
        std::free(ptr);
    }

    void *ptr = nullptr;
    bool cuda_managed = false;
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
        py::pybind11_fail("implicitly_convertible: Unable to find type " +
                          pybind11::type_id<OutputType>());
}

template <typename Array>
py::class_<Array> bind(py::module &m, const char *name) {
    using Scalar = std::conditional_t<
        !is_mask_v<Array>, scalar_t<Array>, bool>;
    using Value  = std::conditional_t<
        is_mask_v<Array> && array_depth_v<Array> == 1,
        bool, value_t<Array>>;

    static constexpr bool IsMask    = is_mask_v<Array>;
    static constexpr bool IsFloat   = is_float_v<Scalar>;
    static constexpr bool IsCUDA    = is_cuda_array_v<Array>;
    static constexpr bool IsDiff    = is_diff_array_v<Array>;
    static constexpr bool IsDynamic = is_dynamic_v<Array>;

    py::class_<Array> cl(m, name);

    cl.def(py::init<>())
      .def(py::init<const Array &>())
      .def(py::init<const Value &>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("managed", &Array::managed)
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
                if (literal)
                    return Array(scalar);
                else
                    return Array::copy(&scalar, 1);
            }),
            "scalar"_a, "literal"_a = true
        );
    }

    if constexpr (IsCUDA) {
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
                if (strstr(tp_name, "enoki.") == nullptr &&
                    py::isinstance<py::sequence>(obj)) {
                    try {
                        std::vector<Value> result = py::cast<std::vector<Value>>(obj);
                        return Array::copy(result.data(), result.size());
                    } catch (...) { }
                }
            }

            throw py::reference_cast_error();
        }))
        .def("torch", [](const Array &a, bool eval) {
            return enoki_to_torch(detach(a), eval); },
            "eval"_a = true
        );
    } else {
        cl.def(py::init([](const py::object &obj) -> Array {
            const char *tp_name = ((PyTypeObject *) obj.get_type().ptr())->tp_name;
            if constexpr (IsDynamic && !IsMask) {
                if (strstr(tp_name, "numpy.ndarray") != nullptr) {
                    using T = expr_t<decltype(detach(std::declval<Array&>()))>;
                    return numpy_to_enoki<T>(obj);
                }

                if constexpr (array_depth_v<Array> == 1) {
                    if (strstr(tp_name, "enoki.") == nullptr &&
                        py::isinstance<py::sequence>(obj)) {
                        try {
                            std::vector<Value> result = py::cast<std::vector<Value>>(obj);
                            return Array::copy(result.data(), result.size());
                        } catch (...) { }
                    }
                }
            }

            throw py::reference_cast_error();
        }));
    }

    if constexpr (IsCUDA || !IsMask) {
        cl.def("numpy", [](const Array &a, bool eval) {
                return enoki_to_numpy(detach(a), eval); },
                "eval"_a = true)
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
    }

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

    if constexpr (IsDynamic) {
        cl.def_static("full",
                      [](Scalar value, size_t size) {
                          Array result(value);
                          set_slices(result, size);
                          return result;
                      }, "value"_a, "size"_a);
    }

    cl.def("__getitem__", [](const Array &a, size_t index) -> Value {
        if (index >= a.size())
            throw py::index_error();
        return a.coeff(index);
    });

    cl.def("__len__", [](const Array &a) { return a.size(); });
    cl.def("resize", [](Array &a, size_t size) { enoki::set_slices(a, size); });
    m.def("slices", [](const Array &a) { return slices(a); });

    if constexpr (!IsDynamic || array_depth_v<Array> > 1) {
        cl.def("__setitem__", [](Array &a, size_t index, const Value &b) {
            if (index >= Array::Size)
                throw py::index_error();
            a.coeff(index) = b;
        });

        if constexpr (!IsMask || array_depth_v<Array> > 1) {
            if constexpr (array_size_v<Array> == 2)
                cl.def(py::init<Value, Value>());
            else if constexpr (array_size_v<Array> == 3)
                cl.def(py::init<Value, Value, Value>());
            else if constexpr (array_size_v<Array> == 4)
                cl.def(py::init<Value, Value, Value, Value>());
        }

        cl.def(py::init([](const std::array<Value, Array::Size> &a) {
            Array result;
            for (size_t i = 0; i < Array::Size; ++i)
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
    } else {
        cl.def_property_readonly("data", [](const Array &a) { return (uintptr_t) a.data(); });

        if constexpr (IsCUDA || IsDiff)
            cl.def_property_readonly("index", [](const Array &a) { return a.index_(); });
    }

    if constexpr (!IsMask) {
        cl.def("__matmul__", [](const Array &a, const Array &b) { return enoki::dot(a, b); });
        m.def("dot", [](const Array &a, const Array &b) { return enoki::dot(a, b); });
        m.def("abs_dot", [](const Array &a, const Array &b) { return enoki::abs_dot(a, b); });
        m.def("normalize", [](const Array &a) { return enoki::normalize(a); });
        m.def("squared_norm", [](const Array &a) { return enoki::squared_norm(a); });
        m.def("norm", [](const Array &a) { return enoki::norm(a); });

        if constexpr (array_size_v<Array> == 3)
            m.def("cross", [](const Array &a, const Array &b) { return enoki::cross(a, b); });
    }

    if constexpr (IsDynamic && !IsDiff) {
        m.def("compress", [](const Array &array, const mask_t<Array> &mask) {
            return compress(array, mask);
        });
    }

    using Index = array_t<uint32_array_t<
        std::conditional_t<array_depth_v<Array> == 1, array_t<Array>, value_t<array_t<Array>>>>>;

    // Scatter/gather currently not supported for dynamic CPU arrays containing masks
    if constexpr (IsDynamic && (!IsMask || IsCUDA)) {
        m.def("gather",
              [](const Array &source, const Index &index, const mask_t<Index> &mask) {
                  return gather<Array>(source, index, mask);
              },
              "source"_a, "index"_a, "mask"_a = true);

        m.def("scatter",
              [](Array &target, const Array &source,
                 const Index &index,
                 const mask_t<Index> &mask) { scatter(target, source, index, mask); },
              "target"_a, "source"_a, "index"_a, "mask"_a = true);

        m.def("scatter_add",
            [](Array &target, const Array &source,
               const Index &index,
               const mask_t<Index> &mask) { scatter_add(target, source, index, mask); },
            "target"_a, "source"_a, "index"_a, "mask"_a = true);
    }

    if constexpr (IsFloat) {
        m.def("abs",        [](const Array &a) { return enoki::abs(a); });
        m.def("sqr",        [](const Array &a) { return enoki::sqr(a); });
        m.def("sqrt",       [](const Array &a) { return enoki::sqrt(a); });
        m.def("safe_sqrt",  [](const Array &a) { return enoki::safe_sqrt(a); });
        m.def("cbrt",       [](const Array &a) { return enoki::cbrt(a); });
        m.def("rcp",        [](const Array &a) { return enoki::rcp(a); });
        m.def("rsqrt",      [](const Array &a) { return enoki::rsqrt(a); });
        m.def("safe_rsqrt", [](const Array &a) { return enoki::safe_rsqrt(a); });

        m.def("ceil",       [](const Array &a) { return enoki::ceil(a); });
        m.def("floor",      [](const Array &a) { return enoki::floor(a); });
        m.def("round",      [](const Array &a) { return enoki::round(a); });
        m.def("trunc",      [](const Array &a) { return enoki::trunc(a); });

        m.def("sin",        [](const Array &a) { return enoki::sin(a); });
        m.def("cos",        [](const Array &a) { return enoki::cos(a); });
        m.def("sincos",     [](const Array &a) { return enoki::sincos(a); });
        m.def("tan",        [](const Array &a) { return enoki::tan(a); });
        m.def("sec",        [](const Array &a) { return enoki::sec(a); });
        m.def("csc",        [](const Array &a) { return enoki::csc(a); });
        m.def("cot",        [](const Array &a) { return enoki::cot(a); });
        m.def("asin",       [](const Array &a) { return enoki::asin(a); });
        m.def("safe_asin",  [](const Array &a) { return enoki::safe_asin(a); });
        m.def("acos",       [](const Array &a) { return enoki::acos(a); });
        m.def("safe_acos",  [](const Array &a) { return enoki::safe_acos(a); });
        m.def("atan",       [](const Array &a) { return enoki::atan(a); });
        m.def("atan2",      [](const Array &a, const Array &b) {
            return enoki::atan2(a, b);
        });

        m.def("sinh",    [](const Array &a) { return enoki::sinh(a); });
        m.def("cosh",    [](const Array &a) { return enoki::cosh(a); });
        m.def("sincosh", [](const Array &a) { return enoki::sincosh(a); });
        m.def("tanh",    [](const Array &a) { return enoki::tanh(a); });
        m.def("sech",    [](const Array &a) { return enoki::sech(a); });
        m.def("csch",    [](const Array &a) { return enoki::csch(a); });
        m.def("coth",    [](const Array &a) { return enoki::coth(a); });
        m.def("asinh",   [](const Array &a) { return enoki::asinh(a); });
        m.def("acosh",   [](const Array &a) { return enoki::acosh(a); });
        m.def("atanh",   [](const Array &a) { return enoki::atanh(a); });

        m.def("log",    [](const Array &a) { return enoki::log(a); });
        m.def("exp",    [](const Array &a) { return enoki::exp(a); });
        m.def("erfinv", [](const Array &a) { return enoki::erfinv(a); });
        m.def("erf",    [](const Array &a) { return enoki::erf(a); });
        m.def("pow",    [](const Array &a, const Array &b) {
            return enoki::pow(a, b);
        });

        m.def("pow",    [](const Array &a, int b) {
            return enoki::pow(a, b);
        });

        m.def("pow",    [](const Array &a, Scalar b) {
            return enoki::pow(a, b);
        });

        cl.def("__pow__", [](const Array &a, const Array &b) {
            return enoki::pow(a, b);
        });

        cl.def("__pow__", [](const Array &a, int b) {
            return enoki::pow(a, b);
        });

        cl.def("__pow__", [](const Array &a, Scalar b) {
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
        if constexpr (IsCUDA) {
            m.def("popcnt", [](const Array &a) { return enoki::popcnt(a); });
            m.def("lzcnt", [](const Array &a) { return enoki::lzcnt(a); });
            m.def("tzcnt", [](const Array &a) { return enoki::tzcnt(a); });
        }
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

        if constexpr (array_depth_v<Array> > 1) {
            m.def("hsum_nested", [](const Array &a) { return enoki::hsum_nested(a); });
            m.def("hprod_nested", [](const Array &a) { return enoki::hprod_nested(a); });
            m.def("hmin_nested", [](const Array &a) { return enoki::hmin_nested(a); });
            m.def("hmax_nested", [](const Array &a) { return enoki::hmax_nested(a); });
            m.def("hmean_nested", [](const Array &a) { return enoki::hmean_nested(a); });
        }

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

    if constexpr (IsDiff) {
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

    if constexpr (IsDiff) {
        using BaseType = expr_t<decltype(detach(std::declval<Array&>()))>;
        implicitly_convertible<BaseType, Array>();
    }

    if constexpr (is_cuda_array_v<Array>) {
        m.def("set_label", [](const Array &a, const char *label) {
            set_label(a, label);
        });
    }

    implicitly_convertible<Value, Array>();

    if constexpr (IsFloat)
        implicitly_convertible<int, Array>();

    if constexpr (!std::is_same_v<Value, Scalar>)
        implicitly_convertible<Scalar, Array>();

    if constexpr (!IsDynamic || array_depth_v<Array> > 1)
        implicitly_convertible<py::list, Array>();

    return cl;
}

template <typename Matrix>
py::class_<Matrix> bind_matrix(py::module &m, const char *name) {
    using Vector = typename Matrix::Column;
    using Value  = typename Matrix::Entry;
    using Array  = Array<Vector, Matrix::Size>;

    auto cls = py::class_<Matrix>(m, name)
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
        .def("__matmul__", [](const Matrix &a, const Matrix &b) {
            return a * b;
        })
        .def("__matmul__", [](const Matrix &a, const Vector &b) {
            return a * b;
        })
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
        .def_static("zero", [](size_t size) { return zero<Matrix>(size); }, "size"_a = 1);

    if constexpr (Matrix::Size == 2) {
        cls.def(py::init<const Vector &, const Vector &>())
           .def(py::init<const Value &, const Value &,
                         const Value &, const Value &>());
    } else if constexpr (Matrix::Size == 3) {
        cls.def(py::init<const Vector &, const Vector &, const Vector &>())
           .def(py::init<const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &>());
    } else if constexpr (Matrix::Size == 4) {
        using Vector3 = enoki::Array<Value, 3>;

        cls.def(py::init<const Vector &, const Vector &, const Vector &, const Vector &>())
           .def(py::init<const Value &, const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &, const Value &,
                         const Value &, const Value &, const Value &, const Value &>())
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

    m.def("transpose", [](const Matrix &m) { return transpose(m); });
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

template <typename Complex>
py::class_<Complex> bind_complex(py::module &m, const char *name) {
    using Value = value_t<Complex>;

    auto cls = py::class_<Complex>(m, name)
        .def(py::init<>())
        .def(py::init<const Value &>())
        .def(py::init<const Value &, const Value &>(), "real"_a, "imag"_a)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self - py::self)
        .def(py::self + py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def("__repr__", [](const Complex &a) -> std::string {
            if (disable_print_flag)
                return "";
            std::ostringstream oss;
            oss << a;
            return oss.str();
        })
        .def("__getitem__", [](const Complex &a, size_t index) {
            if (index >= 2)
                throw py::index_error();
            return a.coeff(index);
        })
        .def("__setitem__", [](Complex &a, size_t index, const Value &value) {
            if (index >= 2)
                throw py::index_error();
            a.coeff(index) = value;
        })
        .def_static("identity", [](size_t size) { return identity<Complex>(size); }, "size"_a = 1)
        .def_static("zero", [](size_t size) { return zero<Complex>(size); }, "size"_a = 1);

    m.def("real", [](const Complex &z) { return real(z); });
    m.def("imag", [](const Complex &z) { return imag(z); });
    m.def("norm", [](const Complex &z) { return norm(z); });
    m.def("squared_norm", [](const Complex &z) { return squared_norm(z); });
    m.def("rcp", [](const Complex &z) { return rcp(z); });
    m.def("conj", [](const Complex &z) { return conj(z); });
    m.def("exp", [](const Complex &z) { return exp(z); });
    m.def("log", [](const Complex &z) { return log(z); });
    m.def("pow", [](const Complex &z1, const Complex &z2) { return pow(z1, z2); });
    m.def("sqrt", [](const Complex &z) { return sqrt(z); });
    m.def("sin", [](const Complex &z) { return sin(z); });
    m.def("cos", [](const Complex &z) { return cos(z); });
    m.def("sincos", [](const Complex &z) { return sincos(z); });
    m.def("tan", [](const Complex &z) { return tan(z); });
    m.def("asin", [](const Complex &z) { return asin(z); });
    m.def("acos", [](const Complex &z) { return acos(z); });
    m.def("atan", [](const Complex &z) { return atan(z); });
    m.def("sinh", [](const Complex &z) { return sinh(z); });
    m.def("cosh", [](const Complex &z) { return cosh(z); });
    m.def("sincosh", [](const Complex &z) { return sincosh(z); });
    m.def("tanh", [](const Complex &z) { return tanh(z); });
    m.def("asinh", [](const Complex &z) { return asinh(z); });
    m.def("acosh", [](const Complex &z) { return acosh(z); });
    m.def("atanh", [](const Complex &z) { return atanh(z); });

    if constexpr (is_diff_array_v<Complex>) {
        using Detached = expr_t<decltype(detach(std::declval<Complex&>()))>;

        m.def("detach", [](const Complex &a) -> Detached { return detach(a); });
        m.def("requires_gradient",
              [](const Complex &a) { return requires_gradient(a); },
              "array"_a);

        m.def("set_requires_gradient",
              [](Complex &a, bool value) { set_requires_gradient(a, value); },
              "array"_a, "value"_a = true);

        m.def("gradient", [](Complex &a) { return eval(gradient(a)); });
        m.def("set_gradient",
              [](Complex &a, const Detached &g, bool b) { set_gradient(a, g, b); },
              "array"_a, "gradient"_a, "backward"_a = true);

        m.def("graphviz", [](const Complex &a) { return graphviz(a); });

        m.def("set_label", [](const Complex &a, const char *label) {
            set_label(a, label);
        });
    }

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

template <bool Scatter, size_t Index, size_t Dim, typename Source, typename Target>
static void copy_array(size_t offset,
                       const std::array<size_t, Dim> &shape,
                       const std::array<size_t, Dim> &strides,
                       const Source &source, Target &target) {
    using namespace enoki;

    size_t cur_shape = shape[Index],
           cur_stride = strides[Index];

    if constexpr (Index == Dim - 1) {
        if constexpr (is_cuda_array_v<Source>) {
            using UInt32 = uint32_array_t<Source>;
            UInt32 index = fmadd(arange<UInt32>((uint32_t) cur_shape),
                                 (uint32_t) cur_stride, (uint32_t) offset);
            if constexpr (Scatter)
                scatter(target, source, index);
            else
                target = gather<Target>(source, index);
        } else {
            for (size_t i = 0; i < cur_shape; ++i) {
                if constexpr (Scatter)
                    target[offset + cur_stride*i] = source[i];
                else
                    target[i] = source[offset + cur_stride*i];
            }
        }
    } else {
        for (size_t i = 0; i < cur_shape; ++i) {
            if constexpr (Scatter)
                copy_array<Scatter, Index + 1, Dim>(offset, shape, strides,
                                                    source.coeff(i), target);
            else
                copy_array<Scatter, Index + 1, Dim>(offset, shape, strides,
                                                    source, target.coeff(i));
            offset += cur_stride;
        }
    }
}

#if defined(ENOKI_CUDA)
template <typename Array>
ENOKI_NOINLINE py::object enoki_to_torch(const Array &src, bool eval) {
    constexpr size_t Depth = array_depth_v<Array>;
    using Scalar = scalar_t<Array>;

    if (enoki::ragged(src))
        throw std::runtime_error("Enoki array is ragged -- cannot convert to PyTorch format!");

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
        copy_array</* Scatter = */ true, 0>(0, shape, strides, src, target);

#if defined(ENOKI_CUDA)
        if (eval) {
            cuda_eval();
            cuda_sync();
        }
#endif
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

        copy_array</* Scatter = */ false, 0>(0, shape, strides, source, result);
    }
    return result;
}
#endif

template <typename Array>
ENOKI_NOINLINE py::object enoki_to_numpy(const Array &src, bool eval) {
    constexpr size_t Depth = array_depth_v<Array>;
    using Scalar = scalar_t<Array>;

    if (enoki::ragged(src))
        throw std::runtime_error("Enoki array is ragged -- cannot convert to NumPy format!");

    std::array<size_t, Depth> shape = enoki::shape(src),
                              shape_rev = shape, strides;
    std::reverse(shape_rev.begin(), shape_rev.end());

    size_t size = 1, stride = sizeof(Scalar);
    for (ssize_t i = (ssize_t) Depth - 1; i >= 0; --i) {
        size *= shape_rev[i];
        strides[i] = stride;
        stride *= shape_rev[i];
    }

    Buffer *buf = new Buffer(stride, is_cuda_array_v<Array>);
    py::object buf_py = py::cast(buf, py::return_value_policy::take_ownership);

    py::array result(py::dtype::of<Scalar>(), shape_rev, strides, buf->ptr, buf_py);
    for (ssize_t i = (ssize_t) Depth - 1; i >= 0; --i)
        strides[i] /= sizeof(Scalar);
    std::reverse(strides.begin(), strides.end());

    if (size > 0) {
        using T = std::conditional_t<
            is_cuda_array_v<Array>,
            CUDAArray<Scalar>,
            DynamicArray<Packet<Scalar, PacketSize>>
        >;
        T target = T::map(buf->ptr, stride);
        copy_array</* Scatter = */ true, 0>(0, shape, strides, src, target);

#if defined(ENOKI_CUDA)
        if (is_cuda_array_v<Array> && eval) {
            cuda_eval();
            cuda_sync();
        }
#endif
    }

    return result;
}

template <typename Array>
ENOKI_NOINLINE Array numpy_to_enoki(py::array src) {
    constexpr size_t Depth = array_depth_v<Array>;
    using SizeArray = std::array<size_t, Depth>;
    using Scalar = scalar_t<Array>;

    py::tuple shape_obj = src.attr("shape");
    py::object dtype_obj = src.attr("dtype");
    py::object target_dtype = py::dtype::of<Scalar>();

    if (shape_obj.size() != Depth)
        throw py::reference_cast_error();

    if (!dtype_obj.is(target_dtype))
        src = py::array_t<Scalar, py::array::forcecast>::ensure(src);

    SizeArray shape   = py::cast<SizeArray>(shape_obj),
              strides = py::cast<SizeArray>(src.attr("strides"));

    std::reverse(shape.begin(), shape.end());
    std::reverse(strides.begin(), strides.end());

    size_t size = 1;
    for (size_t i : shape)
        size *= i;

    for (ssize_t i = 0; i < Depth; ++i)
        strides[i] /= sizeof(Scalar);

    Array result;

    if constexpr (!is_cuda_array_v<Array>)
        set_shape(result, shape);

    if (size > 0) {
        using T = std::conditional_t<
            is_cuda_array_v<Array>,
            CUDAArray<Scalar>,
            DynamicArray<Packet<Scalar, PacketSize>>
        >;
        const T source = T::copy(src.data(), size);

        copy_array</* Scatter = */ false, 0>(0, shape, strides, source, result);
    }

    return result;
}
