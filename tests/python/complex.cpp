#include <enoki/python.h>
#include <enoki/complex.h>
#include <complex>

using namespace enoki;

Complex<float> f1() {
    auto result = Complex<float>(1, 2);
    std::cout << result << std::endl;
    return result;
}

Complex<Packet<float>> f2() {
    auto result = Complex<Packet<float>>(1, 2);
    std::cout << result << std::endl;
    return result;
}

void f3(Complex<float> value) {
    std::cout << value << std::endl;
}

void f4(Complex<Packet<float>> value) {
    std::cout << value << std::endl;
}

PYBIND11_MODULE(test, m) {
    m.def("f1", &f1);
    m.def("f2", &f2);
    m.def("f3", &f3);
    m.def("f4", &f4);
}

