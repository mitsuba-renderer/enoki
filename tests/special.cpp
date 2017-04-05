/*
    tests/conv.cpp -- tests special functions

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <enoki/special.h>

ENOKI_TEST_FLOAT(test01_i0e)  {
    using Scalar = scalar_t<T>;

    double results[] = {
        1.000000000,  0.4657596076, 0.3085083226, 0.2430003542,
        0.2070019212, 0.1835408126, 0.1666574326, 0.1537377447,
        0.1434317819, 0.1349595246, 0.1278333372, 0.1217301682,
        0.1164262212, 0.1117608338, 0.1076152517, 0.1038995314
    };

    for (int i = 0; i < 16; ++i)
        assert(hmax(abs(i0e(T(Scalar(i))) - T(Scalar(results[i])))) < 1e-6);
}
