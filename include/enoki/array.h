/*
    enoki/array.h -- Main header file for the Enoki array class and
    various template specializations

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4146) // warning C4146: unary minus operator applied to unsigned type, result still unsigned
#  pragma warning(disable: 4554) // warning C4554: '>>': check operator precedence for possible error; use parentheses to clarify precedence
#  pragma warning(disable: 4127) // warning C4127: conditional expression is constant
#  pragma warning(disable: 4310) // warning C4310: cast truncates constant value
#endif

#include "array_generic.h"

#include "half.h"

#include "array_idiv.h"

#include "array_recursive.h"

#if !defined(ENOKI_DISABLE_VECTORIZATION)
#if defined(__SSE4_2__)
#  include "array_sse42.h"
#endif

#if defined(__AVX__)
#  include "array_avx.h"
#endif

#if defined(__AVX2__)
#  include "array_avx2.h"
#endif

#if defined(__AVX512F__)
#  include "array_avx512.h"
#endif
#endif

#include "array_round.h"

#include "array_misc.h"

#include "array_dynamic.h"

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
