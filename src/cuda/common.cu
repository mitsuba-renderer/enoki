/*
    src/cuda/common.cpp -- CUDA backend (wrapper routines)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyrighe (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <cuda.h>
#include <stdio.h>
#include "common.cuh"

NAMESPACE_BEGIN(enoki)

ENOKI_EXPORT void* cuda_malloc(size_t size) {
    void *result = nullptr;
    cuda_check(cudaMalloc(&result, size));
    return result;
}

ENOKI_EXPORT void* cuda_malloc_zero(size_t size) {
    void *result = nullptr;
    cuda_check(cudaMalloc(&result, size));
    cuda_check(cudaMemsetAsync(result, 0, size));
    return result;
}

ENOKI_EXPORT void* cuda_managed_malloc(size_t size) {
    void *result = nullptr;
    cuda_check(cudaMallocManaged(&result, size));
    return result;
}

ENOKI_EXPORT void cuda_free(void *ptr) {
    cuda_check(cudaFree(ptr));
}

ENOKI_EXPORT void cuda_memcpy_to_device(void *dst, const void *src, size_t size) {
    cuda_check(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

ENOKI_EXPORT void cuda_memcpy_from_device(void *dst, const void *src, size_t size) {
    cuda_check(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

NAMESPACE_END(enoki)
