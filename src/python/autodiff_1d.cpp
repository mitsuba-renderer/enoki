#include "common.h"

void bind_autodiff_1d(py::module& m) {
    struct Scope {
        Scope(const std::string &name) : name(name) { }

        void enter() { FloatD::push_prefix_(name.c_str()); }
        void exit(py::handle, py::handle, py::handle) { FloatD::pop_prefix_(); }

        std::string name;
    };

    bind<mask_t<FloatD>>(m, "BoolD")
        .def(py::init<const BoolC &>());

    auto fd = bind<FloatD>(m, "FloatD")
        .def(py::init<const FloatC &>())
        .def(py::init<const UInt64D &>())
        .def(py::init<const UInt32D &>())
        .def("set_graph_simplification", [](bool value) { FloatD::set_graph_simplification_(value); })
        .def("whos", []() { py::print(FloatD::whos_()); })
        .def_static("set_log_level",
             [](int log_level) { FloatD::set_log_level_(log_level); },
             "Sets the current log level (0 == none, 1 == minimal, 2 == moderate, 3 == high, 4 == everything)")
        .def_static("log_level", []() { return FloatD::log_level_(); })
        .def_static("simplify_graph", []() { FloatD::simplify_graph_(); })
        .def_static("backward",
                    [](bool free_graph) { backward<FloatD>(free_graph); },
                    "free_graph"_a = true)
        .def_static("forward",
                    [](bool free_graph) { forward<FloatD>(free_graph); },
                    "free_graph"_a = true);

    py::class_<Scope>(fd, "Scope")
        .def(py::init<const std::string &>())
        .def("__enter__", &Scope::enter)
        .def("__exit__", &Scope::exit);

    bind<UInt32D>(m, "UInt32D")
        .def(py::init<const UInt32C &>())
        .def(py::init<const UInt64D &>())
        .def(py::init<const FloatD &>());

    bind<UInt64D>(m, "UInt64D")
        .def(py::init<const UInt64C &>())
        .def(py::init<const UInt32D &>())
        .def(py::init<const FloatD &>());
}
