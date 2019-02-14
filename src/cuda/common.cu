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
#include <iostream>
#include "common.cuh"

NAMESPACE_BEGIN(enoki)

std::string mem_string(size_t size) {
    const char *orders[] = {
        "B", "KiB", "MiB", "GiB",
        "TiB", "PiB", "EiB"
    };
    float value = (float) size;

    int i = 0;
    for (i = 0; i < 6 && value > 1024.f; ++i)
        value /= 1024.f;

    char buf[32];
    snprintf(buf, 32, "%.5g %s", value, orders[i]);

    return buf;
}

std::string time_string(size_t value_) {
    struct Order { float factor; const char* suffix; };
    const Order orders[] = { { 0, "us" },   { 1000, "ms" },
                             { 1000, "s" }, { 60, "m" },
                             { 60, "h" },   { 24, "d" },
                             { 7, "w" },    { (float) 52.1429, "y" } };

    int i = 0;
    float value = (float) value_;
    for (i = 0; i < 7 && value > orders[i+1].factor; ++i)
        value /= orders[i+1].factor;

    char buf[32];
    snprintf(buf, 32, "%.5g %s", value, orders[i].suffix);

    return buf;
}

ENOKI_EXPORT void* cuda_malloc(size_t size) {
    void *result = nullptr;
    cudaError_t ret = cudaMalloc(&result, size);
    if (ret != cudaSuccess) {
        cuda_sync();
        ret = cudaMalloc(&result, size);
    }
    cuda_check(ret);
    return result;
}

ENOKI_EXPORT void* cuda_malloc_zero(size_t size) {
    void *result = nullptr;
    cuda_check(cudaMalloc(&result, size));
    cuda_check(cudaMemsetAsync(result, 0, size));
    return result;
}

template <typename T> __global__ void fill(size_t n, T value, T *out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        out[i] = value;
}

ENOKI_EXPORT void* cuda_malloc_fill(size_t size, uint8_t value) {
    uint8_t *result = nullptr;
    cuda_check(cudaMalloc(&result, size * sizeof(uint8_t)));
    cuda_check(cudaMemsetAsync(result, value, size));
    return result;
}

ENOKI_EXPORT void* cuda_malloc_fill(size_t size, uint16_t value) {
    uint16_t *result = nullptr;
    cuda_check(cudaMalloc(&result, size * sizeof(uint16_t)));
    fill<<<256, 256>>>(size, value, result);
    return result;
}

ENOKI_EXPORT void* cuda_malloc_fill(size_t size, uint32_t value) {
    uint32_t *result = nullptr;
    cuda_check(cudaMalloc(&result, size * sizeof(uint32_t)));
    fill<<<256, 256>>>(size, value, result);
    return result;
}

ENOKI_EXPORT void* cuda_malloc_fill(size_t size, uint64_t value) {
    uint64_t *result = nullptr;
    cuda_check(cudaMalloc(&result, size * sizeof(uint64_t)));
    fill<<<256, 256>>>(size, value, result);
    return result;
}

ENOKI_EXPORT void* cuda_managed_malloc(size_t size) {
    void *result = nullptr;
    cuda_check(cudaMallocManaged(&result, size));
    return result;
}

ENOKI_EXPORT void cuda_memcpy_to_device(void *dst, const void *src, size_t size) {
    cuda_check(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

ENOKI_EXPORT void cuda_memcpy_from_device(void *dst, const void *src, size_t size) {
    cuda_check(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

NAMESPACE_END(enoki)
